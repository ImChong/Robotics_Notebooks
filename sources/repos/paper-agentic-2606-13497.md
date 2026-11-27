# intuitive-robots/sparc

> 来源归档

- **标题：** SPARC（CoRL 2026 官方空间标注流水线）
- **类型：** repo
- **组织：** intuitive-robots（Karlsruhe Institute of Technology 等）
- **代码：** <https://github.com/intuitive-robots/sparc>
- **论文：** <https://arxiv.org/abs/2606.13497>
- **项目页：** <http://intuitive-robots.github.io/sparc-labeling>
- **默认分支：** `main`
- **许可证：** GitHub API 元数据未标注顶层 `license` 字段；内嵌推理组件（AllTracker、SAM2、RobotSeg 等）遵循各自上游开源许可（详见仓库 `docs/vendor-scope.md`）
- **入库日期：** 2026-11-27
- **一句话说明：** 实现 SPARC 三阶段机器人演示视频空间标注流水线，支持任务物体缓存预提取、多 GPU 异步推理标注及基于 `selection.score` 的质量过滤。

## 已核对目录与主要入口

依据官方仓库 `files_sample` 与 `README.md` 核对（静态代码核对，未本地运行）：

| 路径 / 入口 | 用途 |
|---|---|
| `extract_task_objects.py` | 阶段 1 预提取：连接兼容 OpenAI 接口的 VLM 服务（默认 `qwen3-vl-30b`，配 `configs/vllm/qwen3_vl_30b.yaml`），结合 `sparc/llm/semantic_task_parsing.py` 与 `sparc/llm/prompts/prompts.py` 解析指令与夹爪相位，写出可复用的 `task_objects.pkl` 缓存 |
| `annotate.py` | 主标注入口：读取 `configs/annotation_pipeline.yaml` 与 `configs/dataset/*.yaml`（支持 `droid_lerobot`、`bridge_lerobot`、`agibot_world`、`oxe_lerobot`、`libero`、`egolive`、`robocoin`、`robomind2` 等），调度检测、分割、跟踪与 3D 评分，输出 `annotations.jsonl`、`annotations_artifacts/`（HDF5 分片）及可选 `debug/` 图像 |
| `filter_annotations.py` | 阶段 3 过滤入口：读取 `annotations.jsonl`，按 `selection.score`（推荐起始阈值 `--threshold 0.95`，双臂样本要求双臂均达标）筛选生成 `annotations_filtered.jsonl` |
| `sparc/detectors/` | 检测、分割与跟踪封装：`llmdet_hf.py`（LLMDet）、`sam2/`（SAM2.1 实例分割与视频预测）、`alltracker/`（AllTracker 稠密像素跟踪） |
| `sparc/perception/` | 几何与评分辅助模块：`ann_utils.py`、`bbox_ops.py`、`detection_utils.py`、`tracking_models.py` |
| `sparc/data/` | 多格式数据集加载器与相位工具：`dataset_loaders.py`、`mobile_dataset_loaders.py`、`egolive_grasp.py`、`egolive_phases.py`、`starvla_bridge.py`、`cache.py` |
| `setup/prepare_robotseg.py` | RobotSeg 权重准备与 `robotseg_numpy.patch` 补丁脚本 |
| `slurm/` | 集群部署脚本：`serve_vlm.sh`（启动 VLM 服务）与 `launch_annotation.sh`（多节点并发标注） |
| `requirements.txt` / `requirements-vllm.txt` | 分离的标注环境（Python 3.12 + CUDA 扩展，安装需 `--no-deps`）与 vLLM 推理服务环境 |

## 复现边界

- **核对方式：** 静态核对 `README.md` 与仓库文件列表，**未本地运行**。
- **硬件与外部权重依赖：** 运行流水线需 Linux、Python 3.12、NVIDIA GPU、FFmpeg 与 C/C++ 编译器；除手动下载 `robotseg.pt` 与运行 `sparc/detectors/sam2/checkpoints/download_ckpts.sh` 外，LLMDet、AllTracker 与 MoGe 权重在首次运行时下载，且任务解析需独立部署 VLM 推理端点。
- **阈值语义：** `filter_annotations.py` 的 `--threshold 0.95` 是基于 `selection.score` 排序截断的起始操作点，不代表校准后的 95% 概率或 95% 正确率，且有别于基线字段 `baselines.detector.*.detector_score`。

## 关联发布资源（Hugging Face）

- 集合入口：<https://huggingface.co/collections/irl-kit/sparc>
- 模型权重：<https://huggingface.co/irl-kit/SPARC-Qwen3.5-4B>
- 数据集与基准：<https://huggingface.co/datasets/irl-kit/SPARC-VQA>、<https://huggingface.co/datasets/irl-kit/IA-Bench>
- 预生成空间标注：<https://huggingface.co/datasets/irl-kit/sparc-droid-annotations>（BridgeData、AgiBotWorld、OXE 标注在 README 中标注为 Coming soon）

## 对 wiki 的映射

- 论文摘录：[`sources/papers/paper-agentic-2606-13497.md`](../papers/paper-agentic-2606-13497.md)
- 详情页：[`wiki/entities/paper-agentic-2606-13497.md`](../../wiki/entities/paper-agentic-2606-13497.md)
