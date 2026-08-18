# Retrieval experiment results

All frozen retrieval experiments are complete. The selected defaults are BM25F
B3 and BGE V2. HotpotQA and 2WikiMultihopQA provide strong supporting-fact
labels. TriviaQA uses answer-alias matches as weak labels and is interpreted
separately.

## Selected defaults

| Retriever | Selected configuration |
|---|---|
| BM25F | B3; `k1=1.2`, `b=0.5`, `title_b=0.75`, `title_boost=3.0`, `top_k=10` |
| BGE | V2; instructed query, `title + text` passages, normalized vectors, `BAAI/bge-large-en-v1.5`, `batch_size=8`, `top_k=10` |

The framework-level request default remains `top_k=3`; an explicit request
value overrides it. The component and benchmark defaults above describe the
validated retrieval configuration.

## Experiment coverage

| Phase | Data | Status |
|---|---|---|
| BM25F tuning | 10,000 HotpotQA train + 10,000 2Wiki train examples | Complete |
| BGE screening | 2,000 HotpotQA train + 2,000 2Wiki train examples | Complete |
| HotpotQA full validation | 7,405 examples | Complete |
| 2Wiki full validation | 12,576 examples | Complete |
| TriviaQA full validation | 17,944 examples; 17,832 weak-labelled | Complete |
| Original RRFusion-chain compatibility | BM25F and BGE V2 with fake embeddings | Complete |

## BM25F full validation

Candidate-minus-baseline effects use 10,000 deterministic paired-bootstrap
resamples. The practical-effect threshold is one percentage point.

| Dataset | Metric | B0 | B3 | Difference | 95% CI | Practical gain |
|---|---|---:|---:|---:|---:|---|
| HotpotQA | AllSupport@5 | 64.54% | 68.66% | +4.12 pp | [+3.48, +4.78] | Yes |
| HotpotQA | Recall@5 | 81.33% | 83.78% | +2.45 pp | [+2.10, +2.81] | Yes |
| HotpotQA | MRR | 0.8641 | 0.8977 | +3.36 pp | [+2.99, +3.74] | Yes |
| 2Wiki | AllSupport@5 | 53.32% | 60.52% | +7.20 pp | [+6.70, +7.71] | Yes |
| 2Wiki | Recall@5 | 78.12% | 82.62% | +4.50 pp | [+4.23, +4.76] | Yes |
| 2Wiki | MRR | 0.8519 | 0.9464 | +9.45 pp | [+9.09, +9.82] | Yes |
| TriviaQA weak | AllSupport@5 | 62.80% | 62.78% | -0.02 pp | [-0.05, 0.00] | No |
| TriviaQA weak | Recall@5 | 84.92% | 84.91% | -0.01 pp | [-0.02, 0.00] | No |
| TriviaQA weak | MRR | 0.9955 | 0.9952 | -0.03 pp | [-0.07, +0.00] | No |

BM25F B3 produces clear, practically meaningful gains on both strong-label
datasets. The weak-label TriviaQA comparison is effectively unchanged.

## BGE full validation

V0 embeds a raw query and text-only passage. V2 adds the official BGE query
instruction and embeds each passage as `title + text`.

| Dataset | Metric | V0 | V2 | Difference | 95% CI | Practical gain |
|---|---|---:|---:|---:|---:|---|
| HotpotQA | AllSupport@5 | 88.74% | 89.94% | +1.20 pp | [+0.74, +1.66] | Yes |
| HotpotQA | Recall@5 | 94.19% | 94.82% | +0.63 pp | [+0.40, +0.87] | No |
| HotpotQA | MRR | 0.9519 | 0.9579 | +0.59 pp | [+0.38, +0.81] | No |
| 2Wiki | AllSupport@5 | 73.28% | 76.01% | +2.73 pp | [+2.34, +3.13] | Yes |
| 2Wiki | Recall@5 | 89.11% | 90.43% | +1.33 pp | [+1.15, +1.51] | Yes |
| 2Wiki | MRR | 0.9848 | 0.9871 | +0.23 pp | [+0.09, +0.36] | No |
| TriviaQA weak | AllSupport@5 | 62.85% | 62.86% | +0.01 pp | [-0.02, +0.04] | No |
| TriviaQA weak | Recall@5 | 84.94% | 84.94% | +0.01 pp | [-0.01, +0.03] | No |
| TriviaQA weak | MRR | 0.9964 | 0.9963 | -0.01 pp | [-0.05, +0.03] | No |

V2 improves multi-document support retrieval on both strong-label datasets. No
reliable or practical V0/V2 difference appears under TriviaQA weak labels.

## Efficiency and compatibility

On the validation GPU, batch size 8 was the fastest measured setting at 15.60
texts/second. Batch sizes 16, 32, and 64 achieved 15.20, 3.13, and 0.47
texts/second. Batch size 64 did not run out of memory, but it was not efficient.

The original RRFusion framework path discovers, binds, and executes both updated
retrievers without changing `RetrievalRequest`, `RetrievalResult`, or the
component `run(inputs, context)` interface. The final suite passes 87 tests; the
raw HotpotQA demo and real-model BGE integration tests are skipped unless their
external data or opt-in environment is available.

## Interpretation limits

- TriviaQA labels are `weak_answer_alias`, not supporting-fact annotations.
  Their metrics must not be pooled with HotpotQA or 2Wiki.
- Top-10 metrics saturate on the strong-label candidate-document tasks, so
  AllSupport@5 is the primary discriminator.
- These experiments validate retrieval ranking and framework compatibility.
  They do not establish end-to-end answer quality or context-token reduction.
- Raw JSONL results, checkpoints, summaries, and model weights remain ignored.
  Runs record their code commit, and the frozen full BGE runs use commit
  `68aae3b5142af9097a621ffb623404f20412d109`.
