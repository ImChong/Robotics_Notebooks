# caipeng328/TeleOCR

> 来源归档（repo）

- **名称：** TeleOCR
- **类型：** repo / document-parsing / vlm / ocr
- **URL：** <https://github.com/caipeng328/TeleOCR>
- **论文：** [arXiv:2608.12898](../papers/teleocr_arxiv_2608_12898.md)
- **权重：** <https://huggingface.co/StarDoc-AI/TeleOCR>
- **在线体验：** <https://www.teleai.com.cn/docparse/documentParsing> — [`sources/sites/teleai-docparse.md`](../sites/teleai-docparse.md)
- **机构：** 中国电信人工智能（TeleAI）
- **许可证：** 见仓库 LICENSE（入库日以 README 为准）
- **入库日期：** 2026-09-30
- **一句话说明：** 轻量文档 VLM 官方推理栈：`pip install -e .` + `infer.py` 批处理 PDF/图像；默认 **vLLM 异步引擎** 加载 `StarDoc-AI/TeleOCR`；`LAYOUT_MODE=Detection|Segmentation`。

## 运行入口（README）

| 步骤 | 命令 / 模块 |
|------|-------------|
| 克隆 | `git clone https://github.com/caipeng328/TeleOCR.git` |
| 环境 | Python **3.10+**，CUDA；`conda create -n teleocr python=3.10` |
| 安装 | `pip install -e .`（可编辑安装） |
| 批量推理 | `python infer.py --image_sub_path ... --result_save_path ... --use_async --override model_path="StarDoc-AI/TeleOCR" BACKEND="vllm-async-engine" LAYOUT_MODE="Detection"` |
| HF 单页 | `AutoProcessor` + `AutoModel.from_pretrained("StarDoc-AI/TeleOCR", trust_remote_code=True)` |

## 关键代码路径

| 路径 | 作用 |
|------|------|
| `infer.py` | CLI：遍历输入目录 → `do_parse` / `aio_do_parse` |
| `TeleOCR/engine.py` | `do_parse`、`aio_do_parse` 文档解析主流程 |
| `TeleOCR/tools/read_file.py` | `read_fn` 读 PDF/图像页 |
| `TeleOCR/config.py` | 运行时 `CONFIG.update(override)` |
| `TeleOCR-vllm/` | vLLM 后端集成 |
| `TeleOCR/` | 核心解析模块与工具 |

## 配置要点（README）

| 参数 | 说明 |
|------|------|
| `BACKEND` | `vllm-engine` / `vllm-async-engine` |
| `LAYOUT_MODE` | `Detection`（默认）或 `Segmentation`；Wild/PureDoc **真实退化** 轨建议 Segmentation |
| `model_path` | HF 模型 id 或本地路径 |
| `PDF_TOOLS` | `PyMuPDF` / `pypdfium2` |

## 开源边界（2026-09-30）

- **已开源：** 推理代码、配置、`infer.py`、HF 权重、技术报告（arXiv）
- **未随仓发布：** 完整训练数据引擎伪标与四阶段训练脚本（README 侧重数据引擎 **概念** 与评测）
- **社区：** GGUF / llama.cpp（NaviDC-OCR-GGUF）、Ascend 910B 部署经验（知乎等第三方）
