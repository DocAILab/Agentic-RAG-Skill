# TriviaQA 向量化实验

本目录承接 `data/TriviaQA/` 的文本子集，提供从“标准化文本 JSON 子集”到“稠密向量索引”
的向量化脚本。仓库只提交代码与文档，不提交原始数据、模型权重和向量库。

## 脚本

- `scripts/build_vectors.py`：读取加载脚本输出的 `*_subset_*.json`，去重文档、可选按
  token 切块，用 sentence-transformers 编码并复用 framework 的向量索引格式持久化为
  `<output>/<cache_key>/manifest.json + vectors.npy`，同时写出 `corpus.jsonl`。

## 本地小样本测试

先本地跑通加载脚本（见 `data/TriviaQA/README.md`），再向量化小样本：

```powershell
python -B "data/TriviaQA/加载脚本.py" --max-query-samples 10 100
python -B experiments/triviaqa/scripts/build_vectors.py --verify
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input` | `data/TriviaQA/outputs` | 子集 JSON 文件或目录（目录扫描 `*_subset_*.json`） |
| `--output` | `data/TriviaQA/outputs/vector_index` | 索引根目录（`outputs/` 已被 git 忽略） |
| `--model` | `BAAI/bge-large-en-v1.5` | sentence-transformers 模型名 |
| `--device` | 自动 | `cpu` / `cuda` / 留空自动选择 |
| `--batch-size` | 32 | embedding 批大小 |
| `--chunk-tokens` | 512 | 按 token 切块；`0` 表示整篇编码（会被模型截断到 512 token） |
| `--verify` | 关 | 构建后对前 3 条 query 做检索自检 |

构建产物：

- `<output>/<cache_key>/manifest.json`、`vectors.npy`：与 framework Vector Retriever
  完全兼容的索引（首次构建后，若 `framework/settings.yaml` 的
  `runtime.vector_index.cache_dir` 指向该索引根目录，框架运行会直接命中磁盘缓存，不再重新
  编码；默认配置指向 hotpotqa 缓存，需显式切换）。
- `<output>/corpus.jsonl`：去重（含切块）后的文档语料，供检索结果 ID 反查正文使用。

## 服务器跑批（AutoDL）

流程原则：本地先把链路跑通，确认无误后把代码、数据传到服务器，用服务器算力做全量向量化。

1. 创建实例（按量付费），4090 24G 或 CPU 实例均可；数据盘建议 50G+（原始 tar ~2.5G、
   解压证据、模型权重 ~1.3G、向量库）。先开**无卡模式**做下载与装环境。
2. 连接后安装依赖：
   ```bash
   pip install -r requirements.txt
   pip install -e .
   pip install -e ".[embedding]"
   ```
3. 上传代码（git clone 或 scp）与原始数据（`data/raw/triviaqa/`，布局见
   `data/TriviaQA/README.md`）；模型权重首次运行会自动下载（可开学术资源加速）。
4. 服务器上跑加载脚本生成子集，再跑向量化：
   ```bash
   python -B data/TriviaQA/加载脚本.py --max-query-samples 100 800 5000
   python -B experiments/triviaqa/scripts/build_vectors.py --device cuda --batch-size 64
   ```
5. 产物 `scp` 回本地或保留在数据盘；**跑完立即关机**（关机后 GPU 不再计费，数据盘按
   容量计费；长时间不用请备份后释放实例，避免空转扣费）。

## 已知限制

- 整篇编码（`--chunk-tokens 0`）时，超过模型最大序列长度（bge 约 512 token）的正文会被
  截断；建议保持默认 512 切块以获得可用的检索粒度。
- 本目录产物只供 framework 向量索引使用；检索基准（`experiments/retrieval`，已支持
  `--dataset triviaqa`）从 Hugging Face 原始数据流式读取，不消费本目录的子集 JSON。
- 本目录只负责向量化；检索、生成与评测请使用 framework（Vector Retriever 会按相同的
  缓存键直接加载本脚本产出的索引）。
