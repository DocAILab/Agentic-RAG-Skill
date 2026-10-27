"""Skill 仓库副本自优化框架的公共接口。"""

from framework.data_groups import DataGroup, GroupDataError, GroupQA, GroupTask, load_group

from .config import (
    OptimizationSettings,
    SelfOptimizationConfig,
    SelfOptimizationConfigError,
    create_optimizer_model,
    load_self_optimization_config,
)
from .contracts import OptimizationProposal, RevisionResult, SkillEdit
from .optimizer import (
    OptimizationError,
    OptimizationNoChange,
    SkillOptimizer,
    parse_optimization_proposal,
)
from .runner import SelfOptimizationError, run_self_optimization as run_sample_optimization
from .task_config import (
    RAGTaskConfig,
    TaskWorkflowConfig,
    TaskWorkflowConfigError,
    load_task_workflow_config,
)
from .task_flow import TaskWorkflowError, run_task_workflow, validate_task_workflow
from .workspace import RevisionRejected, SkillWorkspace, WorkspaceError

# The package API and all CLI entrypoints execute the same Task/Group workflow.
run_self_optimization = run_task_workflow

__all__ = [
    "DataGroup",
    "GroupDataError",
    "GroupQA",
    "GroupTask",
    "OptimizationError",
    "OptimizationNoChange",
    "OptimizationProposal",
    "OptimizationSettings",
    "RAGTaskConfig",
    "RevisionRejected",
    "RevisionResult",
    "SelfOptimizationConfig",
    "SelfOptimizationConfigError",
    "SelfOptimizationError",
    "SkillEdit",
    "SkillOptimizer",
    "SkillWorkspace",
    "TaskWorkflowConfig",
    "TaskWorkflowConfigError",
    "TaskWorkflowError",
    "WorkspaceError",
    "create_optimizer_model",
    "load_self_optimization_config",
    "load_group",
    "load_task_workflow_config",
    "parse_optimization_proposal",
    "run_self_optimization",
    "run_sample_optimization",
    "run_task_workflow",
    "validate_task_workflow",
]
