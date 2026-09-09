# ENEAS（speridlabs/eneas）

> 来源归档

- **标题：** ENEAS
- **类型：** repo
- **来源：** SperidLabs
- **链接：** <https://github.com/speridlabs/eneas>
- **论文：** <https://arxiv.org/abs/2609.03756>
- **项目页：** <https://speridlabs.com/research/eneas>
- **演示：** <https://huggingface.co/spaces/speridlabs/eneas>
- **许可：** Apache 2.0
- **入库日期：** 2026-09-09
- **一句话说明：** 文本/点提示实例跟踪与语义类别分割；CLI `eneas unique_instance` / `generic_category`；SeC-4B + grounding HF 自动拉取。
- **沉淀到 wiki：** [`wiki/entities/paper-eneas.md`](../../wiki/entities/paper-eneas.md)

---

## 仓库入口（README）

| 组件 | 说明 |
|------|------|
| 安装 | `uv sync` 或 `pip install -e .`；Python 3.10–3.12；CUDA GPU |
| 唯一实例（文本） | `eneas unique_instance -i ./frames --text "…" -o ./output` |
| 唯一实例（点） | `eneas unique_instance -i ./frames -p x,y -f frame.jpg` |
| 语义发现 | `eneas generic_category -i ./frames --category "chair"`（需 **Ollama**） |
| Python API | `UniqueInstanceSegmenter().segment(frames_path=…, text=…)` |
| 模型 | SeC-4B 分割 + grounding；`HF_HOME` 可改缓存路径 |
| 编码器档位 | `-s long-small` 等速度与精度权衡 |

## 开源边界（截至 2026-09-09）

- **已开源**：核心库、CLI、Gradio UI（`uv sync --extra ui`）、pytest/ruff 开发依赖。
- **权重**：首次运行自动从 Hugging Face 下载。
- **Generic 模式**：依赖本机 Ollama VLM 做语义验证；部署需单独装 Ollama。
- **上游**：扩展 SeC 架构；评测对齐 SAM 3 SA-Co/VEval 协议。
