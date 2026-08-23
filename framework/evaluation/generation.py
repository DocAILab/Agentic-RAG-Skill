"""复刻 XRAG 生成指标所使用的 Jury、Evaluate 与 GPT-2 计算口径。"""

from __future__ import annotations

import re
import string
from collections.abc import Callable, Mapping, Sequence
from threading import Lock
from typing import Any


class GenerationMetricError(RuntimeError):
    """表示生成指标依赖缺失、资源加载或计算失败。"""


class XRAGGenerationEvaluator:
    """计算 XRAG 的九项生成指标，并复用进程内 PPL 模型。"""

    def __init__(
        self,
        *,
        perplexity_model_id: str = "openai-community/gpt2",
        perplexity_calculator: Callable[[str], float] | None = None,
    ) -> None:
        """保存 PPL 模型标识，并允许测试注入等价的 PPL 计算器。"""
        self.perplexity_model_id = perplexity_model_id
        self._perplexity_calculator = perplexity_calculator
        self._perplexity_model: Any = None
        self._perplexity_tokenizer: Any = None
        self._perplexity_device: Any = None
        self._model_lock = Lock()

    def evaluate(
        self,
        prediction: str,
        references: str | Sequence[str],
    ) -> dict[str, float]:
        """按 XRAG 的尺度、参数与多参考聚合规则计算生成指标。"""
        prediction_text = _validate_prediction(prediction)
        reference_texts = _normalize_references(references)
        chrf = _chrf_score(prediction_text, reference_texts, word_order=0)
        chrf_pp = _chrf_score(prediction_text, reference_texts, word_order=2)
        rouge = _rouge_scores(prediction_text, reference_texts)
        ppl = self._perplexity(prediction_text)
        if int(ppl) > 1600:
            ppl = 0.0
        return {
            "chrf": chrf,
            "chrf++": chrf_pp,
            "meteor": _meteor_score(prediction_text, reference_texts),
            "r1": rouge["rouge1"],
            "r2": rouge["rouge2"],
            "rl": rouge["rougeL"],
            "ppl": ppl,
            "cer": _character_error_rate(prediction_text, reference_texts),
            "wer": _word_error_rate(prediction_text, reference_texts),
        }

    def _perplexity(self, prediction: str) -> float:
        """调用注入计算器或复用 GPT-2 模型计算单条预测的 PPL。"""
        if self._perplexity_calculator is not None:
            return float(self._perplexity_calculator(prediction))
        self._ensure_perplexity_model()
        return self._compute_cached_perplexity(prediction)

    def _ensure_perplexity_model(self) -> None:
        """按需加载并缓存 XRAG 指定的因果语言模型与 tokenizer。"""
        if self._perplexity_model is not None:
            return
        with self._model_lock:
            if self._perplexity_model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = AutoModelForCausalLM.from_pretrained(
                    self.perplexity_model_id
                ).to(device)
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained(self.perplexity_model_id)
                if tokenizer.pad_token is None:
                    existing_special_tokens = list(
                        tokenizer.special_tokens_map_extended.values()
                    )
                    if not existing_special_tokens:
                        raise GenerationMetricError(
                            "PPL tokenizer has no reusable special token for padding"
                        )
                    tokenizer.add_special_tokens(
                        {"pad_token": existing_special_tokens[0]}
                    )
            except Exception as exc:
                raise GenerationMetricError(
                    "Cannot load XRAG perplexity model "
                    f"'{self.perplexity_model_id}': {exc}"
                ) from exc
            self._perplexity_model = model
            self._perplexity_tokenizer = tokenizer
            self._perplexity_device = device

    def _compute_cached_perplexity(self, prediction: str) -> float:
        """复刻 Hugging Face Evaluate PPL 的交叉熵与 BOS 处理。"""
        try:
            import torch
            from torch.nn import CrossEntropyLoss

            tokenizer = self._perplexity_tokenizer
            model = self._perplexity_model
            device = self._perplexity_device
            encodings = tokenizer(
                [prediction],
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)
            encoded_texts = encodings["input_ids"]
            attention_masks = encodings["attention_mask"]
            if not torch.all(torch.ge(attention_masks.sum(1), 1)):
                raise GenerationMetricError(
                    "PPL prediction must contain at least one model token"
                )
            if tokenizer.bos_token_id is not None:
                bos_tokens = torch.tensor(
                    [[tokenizer.bos_token_id]],
                    device=device,
                )
                encoded_texts = torch.cat([bos_tokens, encoded_texts], dim=1)
                attention_masks = torch.cat(
                    [torch.ones_like(bos_tokens), attention_masks],
                    dim=1,
                )
            with torch.no_grad():
                logits = model(
                    encoded_texts,
                    attention_mask=attention_masks,
                ).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = encoded_texts[..., 1:].contiguous()
            shift_masks = attention_masks[..., 1:].contiguous()
            losses = CrossEntropyLoss(reduction="none")(
                shift_logits.transpose(1, 2),
                shift_labels,
            )
            perplexity = torch.exp(
                (losses * shift_masks).sum(1) / shift_masks.sum(1)
            )
            return float(perplexity[0].item())
        except GenerationMetricError:
            raise
        except Exception as exc:
            raise GenerationMetricError(f"Cannot compute XRAG PPL: {exc}") from exc


def _chrf_score(
    prediction: str,
    references: Sequence[str],
    *,
    word_order: int,
) -> float:
    """使用 SacreBLEU CHRF，并按 Jury/XRAG 将百分制结果除以 100。"""
    try:
        from sacrebleu import CHRF

        reference_streams = [[reference] for reference in references]
        score = CHRF(word_order=word_order).corpus_score(
            [prediction],
            reference_streams,
        ).score
    except Exception as exc:
        raise GenerationMetricError(f"Cannot compute ChrF: {exc}") from exc
    return float(score) / 100.0


def _meteor_score(prediction: str, references: Sequence[str]) -> float:
    """使用 Jury 的小写去标点 tokenizer 和 NLTK METEOR 多参考规则。"""
    try:
        return _compute_meteor_score(prediction, references)
    except LookupError:
        _download_meteor_resources()
        try:
            return _compute_meteor_score(prediction, references)
        except Exception as exc:
            raise GenerationMetricError(f"Cannot compute METEOR: {exc}") from exc
    except Exception as exc:
        raise GenerationMetricError(f"Cannot compute METEOR: {exc}") from exc


def _compute_meteor_score(prediction: str, references: Sequence[str]) -> float:
    """在资源已就绪时执行一次 NLTK METEOR 计算。"""
    from nltk.translate import meteor_score

    prediction_tokens = _jury_tokens(prediction)
    reference_tokens = [_jury_tokens(reference) for reference in references]
    return float(
        meteor_score.meteor_score(
            references=reference_tokens,
            hypothesis=prediction_tokens,
            alpha=0.9,
            beta=3,
            gamma=0.5,
        )
    )


def _download_meteor_resources() -> None:
    """按新版 Jury 的资源列表下载 METEOR 所需 NLTK 数据。"""
    try:
        import nltk

        for resource in ("wordnet", "punkt", "omw-1.4"):
            if not nltk.download(resource, quiet=True):
                raise GenerationMetricError(
                    f"Cannot download NLTK resource '{resource}'"
                )
    except GenerationMetricError:
        raise
    except Exception as exc:
        raise GenerationMetricError(
            f"Cannot prepare METEOR NLTK resources: {exc}"
        ) from exc


def _jury_tokens(text: str) -> list[str]:
    """复刻 Jury DefaultTokenizer 的 ASCII 标点清理、小写和空白折叠。"""
    pattern = rf"[{re.escape(string.punctuation)}]"
    normalized = " ".join(re.sub(pattern, " ", text).split()).lower()
    return normalized.split()


def _rouge_scores(
    prediction: str,
    references: Sequence[str],
) -> dict[str, float]:
    """计算无词干 ROUGE F-measure，并按 Jury 对参考答案逐项取最大值。"""
    try:
        from rouge_score import rouge_scorer

        metric_names = ("rouge1", "rouge2", "rougeL")
        scorer = rouge_scorer.RougeScorer(metric_names, use_stemmer=False)
        scores = [scorer.score(reference, prediction) for reference in references]
    except Exception as exc:
        raise GenerationMetricError(f"Cannot compute ROUGE: {exc}") from exc
    return {
        metric: max(score[metric].fmeasure for score in scores)
        for metric in metric_names
    }


def _word_error_rate(prediction: str, references: Sequence[str]) -> float:
    """计算 JiWER WER，并复刻 Jury 对多参考答案取最大错误率的规则。"""
    try:
        import jiwer

        return float(
            max(jiwer.process_words(reference, prediction).wer for reference in references)
        )
    except Exception as exc:
        raise GenerationMetricError(f"Cannot compute WER: {exc}") from exc


def _character_error_rate(prediction: str, references: Sequence[str]) -> float:
    """计算 JiWER CER，并复刻 Jury 对多参考答案取最大错误率的规则。"""
    try:
        import jiwer

        return float(
            max(
                jiwer.process_characters(reference, prediction).cer
                for reference in references
            )
        )
    except Exception as exc:
        raise GenerationMetricError(f"Cannot compute CER: {exc}") from exc


def _validate_prediction(prediction: str) -> str:
    """校验预测答案为非空字符串。"""
    if not isinstance(prediction, str):
        raise TypeError("prediction must be a string")
    if not prediction.strip():
        raise ValueError("prediction must not be empty")
    return prediction


def _normalize_references(references: str | Sequence[str]) -> tuple[str, ...]:
    """将单参考或多参考答案统一为非空字符串元组。"""
    if isinstance(references, str):
        normalized = (references,)
    elif isinstance(references, (bytes, bytearray)) or not isinstance(
        references, Sequence
    ):
        raise TypeError("references must be a string or a sequence of strings")
    else:
        normalized = tuple(references)
    if not normalized:
        raise ValueError("references must not be empty")
    if not all(isinstance(reference, str) for reference in normalized):
        raise TypeError("references must contain only strings")
    if any(not reference.strip() for reference in normalized):
        raise ValueError("references must contain non-empty strings")
    return normalized


_DEFAULT_GENERATION_EVALUATOR: XRAGGenerationEvaluator | None = None
_DEFAULT_EVALUATOR_LOCK = Lock()


def get_default_generation_evaluator() -> XRAGGenerationEvaluator:
    """返回进程级默认生成评测器，使 GPT-2 PPL 权重只加载一次。"""
    global _DEFAULT_GENERATION_EVALUATOR
    if _DEFAULT_GENERATION_EVALUATOR is not None:
        return _DEFAULT_GENERATION_EVALUATOR
    with _DEFAULT_EVALUATOR_LOCK:
        if _DEFAULT_GENERATION_EVALUATOR is None:
            _DEFAULT_GENERATION_EVALUATOR = XRAGGenerationEvaluator()
    return _DEFAULT_GENERATION_EVALUATOR


def validate_generation_scores(scores: Mapping[str, float]) -> dict[str, float]:
    """校验自定义生成评测器返回完整的 XRAG 指标集合。"""
    required = ("chrf", "chrf++", "meteor", "r1", "r2", "rl", "ppl", "cer", "wer")
    missing = [name for name in required if name not in scores]
    if missing:
        raise GenerationMetricError(
            f"Generation evaluator is missing metrics: {missing}"
        )
    try:
        return {name: float(scores[name]) for name in required}
    except (TypeError, ValueError) as exc:
        raise GenerationMetricError(
            "Generation evaluator metrics must be numeric"
        ) from exc
