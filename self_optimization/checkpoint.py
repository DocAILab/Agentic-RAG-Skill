"""Durable Task transactions for bounded Slurm allocations; no scoring policy."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .workspace import WorkspaceError, _RUN_ID_PATTERN, _default_run_id, _filesystem_path, _reject_symlinks, _tree_hash


class CheckpointError(WorkspaceError):
    pass


class TaskBoundaryExit(SystemExit):
    """Exit 75 means the committed workspace is ready for another allocation."""
    def __init__(self):
        super().__init__(75)


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == 'posix':
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sync_tree(root):
    """Flush committed files before publishing the checkpoint on every platform."""
    paths = list(Path(root).rglob('*'))
    for path in paths:
        if path.is_file():
            # Windows FlushFileBuffers requires a writable handle. All of these
            # files belong to the run's writable copy, never the source Skills.
            mode = 'r+b' if os.name == 'nt' else 'rb'
            with path.open(mode) as stream:
                os.fsync(stream.fileno())
    # Windows has no portable directory-fsync equivalent in the standard library.
    if os.name != 'posix':
        return
    directories = [path for path in paths if path.is_dir()] + [Path(root), Path(root).parent]
    for path in reversed(directories):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@contextmanager
def _file_lock(path):
    path = _filesystem_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if os.name == 'posix':
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CheckpointError('Another process holds this workflow run lock') from exc
        else:
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise CheckpointError('Another process holds this workflow run lock') from exc
        try:
            yield
        finally:
            if os.name == 'posix':
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            else:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def exclusive_workflow(function):
    @wraps(function)
    def wrapped(config, **kwargs):
        from .task_config import TaskWorkflowConfig
        if not isinstance(config, TaskWorkflowConfig):
            # Let the public runner report its configuration error before any I/O.
            return function(config, **kwargs)
        run_id = kwargs.get('run_id') or config.self_optimization.run_id
        if kwargs.get('resume') and not run_id:
            raise CheckpointError('--resume requires an explicit stable run-id')
        run_id = run_id or _default_run_id()
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise CheckpointError('Invalid run-id')
        kwargs['run_id'] = run_id
        root = Path(config.self_optimization.runs_root).resolve()
        with _file_lock(root / '.workflow-locks' / (run_id + '.lock')):
            return function(config, **kwargs)
    return wrapped


def workflow_identity(config, selection, freeze_skills):
    """Persist hashes, not config contents or credentials."""
    configuration = {
        'self_optimization': asdict(config.self_optimization),
        'tasks': [asdict(task) for task in config.tasks],
        'max_process_examples': config.max_process_examples,
        'validation_examples_per_task': config.validation_examples_per_task,
        'qa_execution': config.qa_execution.to_dict(),
    }
    canonical = json.dumps(configuration, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8')
    repository = Path(__file__).resolve().parent.parent
    source = hashlib.sha256()
    for package in ('framework', 'self_optimization'):
        for path in sorted((repository / package).rglob('*.py')):
            source.update(path.relative_to(repository).as_posix().encode('utf-8'))
            source.update(path.read_bytes())
    return {
        'schema_version': 1, 'selection': selection, 'freeze_skills': bool(freeze_skills),
        'configuration_sha256': hashlib.sha256(canonical).hexdigest(),
        'framework_source_sha256': source.hexdigest(),
        'task_files': {str(task.data_path.resolve()): _hash_file(task.data_path) for task in config.tasks},
        'group_manifest_sha256': _hash_file(config.group.manifest_path) if config.group else None,
        'corpus_fingerprint': config.group.corpus_fingerprint if config.group else None,
        'shuffle_seed': config.group.shuffle_seed if config.group else None,
    }


class TaskCheckpoint:
    def __init__(self, workspace, identity, *, resume=False):
        self.workspace = workspace
        self.root = workspace.run_root / 'checkpoints'
        self.path = self.root / 'state.json'
        self.report_path = workspace.run_root / 'task-flow.report.json'
        self.recovery = []
        self.completed_report = None
        self.identity = identity
        if resume:
            if not self.path.is_file():
                raise CheckpointError('This run has no recoverable Task checkpoint')
            self.state = json.loads(self.path.read_text(encoding='utf-8'))
            if self.state.get('identity') != identity:
                raise CheckpointError('Resume protocol/configuration/source/data/arm differs from original run')
            self._recover()
        else:
            if self.root.exists():
                raise CheckpointError('Checkpoint directory already exists')
            self.root.mkdir()
            self.state = {'schema_version': 1, 'identity': identity, 'completed_tasks': 0,
                          'committed_skill_hash': workspace.working_hash(), 'committed_report': None,
                          'active_task': None, 'attempt_counts': {}, 'recoveries': []}
            self._save()

    @property
    def completed_tasks(self):
        return self.state['completed_tasks']

    def _save(self):
        _atomic_json(self.path, self.state)

    def _inside(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()) or path == self.root.resolve():
            raise CheckpointError('Checkpoint path escapes this run')
        return path

    def _recover(self):
        committed = self.state['committed_report']
        if committed:
            path = self._inside(committed['path'])
            if _hash_file(path) != committed['sha256']:
                raise CheckpointError('Committed report hash mismatch')
            self.completed_report = json.loads(path.read_text(encoding='utf-8'))
            expected = [task['task_id'] for task in self.identity['selection']['tasks'][:self.completed_tasks]]
            actual = [task['task_id'] for task in self.completed_report['tasks']]
            if actual != expected or len(actual) != self.completed_tasks:
                raise CheckpointError('Committed Tasks are not the original selected prefix')
        elif self.completed_tasks:
            raise CheckpointError('Completed Task count has no committed report')
        active = self.state['active_task']
        if active:
            snapshot = self._inside(active['snapshot'])
            _reject_symlinks(snapshot)
            if _tree_hash(snapshot) != active['skill_hash'] or active['skill_hash'] != self.state['committed_skill_hash']:
                raise CheckpointError('Task start snapshot does not match last committed Skills')
            current = self.workspace.skill_root
            if current.resolve() != (self.workspace.run_root / 'skills').resolve() or current.is_symlink():
                raise CheckpointError('Unsafe workspace restore target')
            archive = self._inside('interrupted/' + active['attempt_id'])
            if current.exists():
                _reject_symlinks(current)
                if not archive.exists():
                    shutil.copytree(current, archive)
                shutil.rmtree(current)
            shutil.copytree(snapshot, current)
            _sync_tree(current)
            if self.workspace.working_hash() != active['skill_hash']:
                raise CheckpointError('Restored Task start snapshot hash mismatch')
            recovery = {'task_id': active['task_id'], 'attempt_id': active['attempt_id'],
                        'restored_skill_hash': active['skill_hash'], 'interrupted_skills': str(archive),
                        'uncommitted_revision_ids': [path.name for path in sorted(self.workspace.revisions_root.glob('revision-*'))
                            if path.is_dir() and int(path.name.split('-')[1]) > active.get('revision_number_before', 0)],
                        'utc': datetime.now(UTC).isoformat()}
            self.recovery.append(recovery)
            self.state['recoveries'].append(recovery)
            self.state['active_task'] = None
            self._save()
        elif self.workspace.working_hash() != self.state['committed_skill_hash']:
            raise CheckpointError('Committed working Skills changed outside an active Task')
        self._publish_committed_report()

    def _publish_committed_report(self):
        desired = self.state['committed_report']
        if self.report_path.exists() and (not desired or _hash_file(self.report_path) != desired['sha256']):
            archive = self._inside('report-history/' + _hash_file(self.report_path) + '.json')
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists():
                shutil.copyfile(self.report_path, archive)
            if not desired:
                self.report_path.unlink()
        if desired:
            _atomic_json(self.report_path, self.completed_report)

    def recover_events(self, event_log):
        rows = []
        if event_log.path.exists():
            raw = event_log.path.read_bytes()
            for line in raw.decode('utf-8', errors='replace').splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
            if raw and not raw.endswith(b'\n'):
                with event_log.path.open('ab') as stream:
                    stream.write(b'\n')
                event_log.write('event_log_partial_line_preserved')
        completion = self.state.get('last_completion')
        if completion and not any(isinstance(row, dict) and row.get('event') == 'task_completed' and row.get('checkpoint_commit_id') == completion['checkpoint_commit_id'] for row in rows):
            event_log.write('task_completed', **completion, recovered_from_durable_commit=True)
        for recovery in self.recovery:
            event_log.write('task_attempt_recovered', **recovery)

    def begin_task(self, task_id, task_index):
        if self.state['active_task'] is not None or task_index != self.completed_tasks + 1:
            raise CheckpointError('Cannot begin Task outside the committed sequence')
        if self.identity['selection']['tasks'][task_index - 1]['task_id'] != task_id:
            raise CheckpointError('Task ID differs from selected sequence')
        if self.workspace.working_hash() != self.state['committed_skill_hash']:
            raise CheckpointError('Working Skills do not match committed checkpoint')
        attempt = self.state['attempt_counts'].get(task_id, 0) + 1
        attempt_id = f'task-{task_index:04d}-attempt-{attempt:04d}'
        snapshot = self._inside('starts/' + attempt_id + '/skills')
        if snapshot.exists():
            # A crash while copying the not-yet-active snapshot left an owned orphan.
            attempt += 1
            while self._inside(f'starts/task-{task_index:04d}-attempt-{attempt:04d}/skills').exists():
                attempt += 1
            attempt_id = f'task-{task_index:04d}-attempt-{attempt:04d}'
            snapshot = self._inside('starts/' + attempt_id + '/skills')
        shutil.copytree(self.workspace.skill_root, snapshot)
        _sync_tree(snapshot)
        snapshot_hash = _tree_hash(snapshot)
        if snapshot_hash != self.state['committed_skill_hash']:
            raise CheckpointError('Task start snapshot copy failed integrity check')
        self.state['attempt_counts'][task_id] = attempt
        self.state['active_task'] = {'task_id': task_id, 'task_index': task_index,
            'attempt_id': attempt_id, 'snapshot': snapshot.relative_to(self.root).as_posix(), 'skill_hash': snapshot_hash,
            'revision_number_before': self.workspace._revision_number}
        self._save()
        return attempt_id

    def commit_task(self, report, event_log, completion):
        active = self.state['active_task']
        if not active or len(report['tasks']) != self.completed_tasks + 1 or report['tasks'][-1]['task_id'] != active['task_id']:
            raise CheckpointError('Invalid Task commit sequence')
        self.workspace.assert_source_unchanged()
        _sync_tree(self.workspace.skill_root)
        _sync_tree(self.workspace.revisions_root)
        commit_id = active['attempt_id']
        path = self._inside('commits/' + commit_id + '.json')
        _atomic_json(path, report)
        completion = {**completion, 'attempt_id': active['attempt_id'], 'checkpoint_commit_id': commit_id}
        self.state.update(completed_tasks=len(report['tasks']), active_task=None,
            committed_skill_hash=self.workspace.working_hash(),
            committed_report={'path': path.relative_to(self.root).as_posix(), 'sha256': _hash_file(path)},
            last_completion=completion)
        self._save()  # The only authoritative commit point. Recovery never reruns it.
        self.completed_report = report
        self._publish_committed_report()
        event_log.write('task_completed', **completion)


def restore_metrics(tasks):
    from framework.evaluation.evaluator import ExampleMetrics, RetrievalMetrics, GenerationMetrics
    retrieval_names = {'f1': 'F1', 'f1_at_1': 'F1@1', 'mrr': 'MRR', 'hit_at_1': 'Hit@1',
                       'hit_at_10': 'Hit@10', 'map': 'MAP', 'ndcg': 'NDCG', 'dcg': 'DCG', 'idcg': 'IDCG'}
    generation_names = {'chrf': 'ChrF', 'chrf_pp': 'ChrF++', 'meteor': 'METEOR', 'r1': 'R1',
                        'r2': 'R2', 'rl': 'RL', 'ppl': 'PPL', 'cer': 'CER', 'wer': 'WER'}
    return [ExampleMetrics(
        retrieval=RetrievalMetrics(**{field: example['metrics']['retrieval'][key] for field, key in retrieval_names.items()}),
        generation=GenerationMetrics(**{field: example['metrics']['generation'][key] for field, key in generation_names.items()}))
        for task in tasks for example in task['examples']]
