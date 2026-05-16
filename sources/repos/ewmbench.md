# EWMBench（AgibotTech 官方实现）

> 来源归档

- **标题：** EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models
- **类型：** repo + benchmark toolkit + HF weights/data
- **组织：** AgiBot（AgibotTech）等（论文作者单位见 [sources/papers/ewmbench.md](../papers/ewmbench.md)）
- **代码：** <https://github.com/AgibotTech/EWMBench>
- **论文：** <https://arxiv.org/abs/2505.09694>
- **Hugging Face：** 权重集合 [agibot-world/EWMBench-model](https://huggingface.co/agibot-world/EWMBench-model)；数据集 [agibot-world/EWMBench](https://huggingface.co/datasets/agibot-world/EWMBench)
- **入库日期：** 2026-05-16
- **一句话说明：** 开源 **具身世界模型（EWM）** 视频生成评测管线：统一初始化（初始帧 + 语言指令 + 可选 6D 末端轨迹）→ 候选模型自回归生成未来帧 → 在 **场景一致性、动作轨迹、语义对齐与多样性** 三轴上对照真值；附 `config.yaml` 驱动的预处理与评测脚本，以及 HF 上的微调 DINOv2 / YOLO-World 等权重入口。
- **沉淀到 wiki：** [EWMBench（具身世界模型生成评测）](../../wiki/entities/ewmbench.md)

---

## README 归纳（环境、数据、权重）

1. **环境：** README 推荐 **CUDA 11.8**、**Python 3.10+**、`git clone --recursive` 后 `pip install -r requirements.txt`，并额外 `pip install git+https://github.com/openai/CLIP.git`。
2. **权重：** 文档要求自备 **Qwen2.5-VL-7B-Instruct** 与 **qwen-vl-utils** 适配、**CLIP ViT-B/32** 与 **clip-vit-base-patch16**，以及仓库 README 指向的 **微调 DINOv2 与 YOLO-World**（HF：`EWMBench-model`）；各路径写入 **`config.yaml`**。
3. **数据：** 从 HF 下载后放入 `./data`；真值目录约定为 `gt_dataset/`，生成结果目录名需以 `_dataset` 结尾；首次运行需按 README 完成 **ground truth detection** 等预处理（`bash processing.sh ./config.yaml`）。
4. **命名说明：** README 正文中将框架缩写写作 **「EWMBM」**（与论文题名 **EWMBench** 并存）；以论文与 arXiv 题名为准。

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 为「像素域 / 视频域世界模型」提供 **面向操作任务** 的可复现评测坐标，而非仅 VBench 类通用视频审美指标 |
| [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) | 当把视频生成当作具身推演接口时，EWMBench 显式检查 **场景静态元素守恒、末端轨迹与指令逻辑** 等 embodied 特有问题 |
| [Manipulation](../../wiki/tasks/manipulation.md) | 当前 v1 基准聚焦 **机械臂操作** 十类任务、每类多 episode；论文讨论未来扩展到导航与移动操作 |

---

## 对 wiki 的映射

- 新建 **`wiki/entities/ewmbench.md`**：基准 + 工具链实体页（流程 mermaid、三轴指标、与生成式世界模型/视频即仿真互链）。
- 轻量交叉更新 **`wiki/methods/generative-world-models.md`**、**`wiki/concepts/video-as-simulation.md`**：各补一条指向该实体页的关联，避免孤岛页。

---

## 外部参考（便于复核）

- Hu et al., *EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models*, [arXiv:2505.09694](https://arxiv.org/abs/2505.09694)
- [AgibotTech/EWMBench（GitHub）](https://github.com/AgibotTech/EWMBench)
