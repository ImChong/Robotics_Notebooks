# DriveTeach-VLA（ShivaTeam/DriveTeach-VLA 官方仓库）

- **标题**: Teaching Vision-Language-Action Models What to See and Where to Look
- **论文**: <https://arxiv.org/abs/2607.01658>（**ECCV 2026**）
- **代码**: <https://github.com/ShivaTeam/DriveTeach-VLA>
- **类型**: code-release（与 `sources/papers/driveteach_vla_arxiv_2607_01658.md` 分工：本文件聚焦仓库、数据引擎与训练入口）
- **机构**: 北京航空航天大学（BUAA）、清华大学智能产业研究院（AIR）、滴滴、中国传媒大学
- **License**: **Apache-2.0**
- **首次入库**: 2026-09-21

## 一句话摘要

ECCV 2026 官方实现：**DVD** 预训练注入驾驶视觉先验，**2D-TGP** 提供轨迹对齐空间 prompt，**TGP-guided SFT + GRPO** 训练 Qwen2.5-VL 驾驶 VLA；含可配置 **data_engine**、LLaMA-Factory 补丁与 Google Drive 标注数据。

## 仓库结构（README 摘要，截至 2026-09-21）

| 路径 | 作用 |
|------|------|
| `data_engine/` | NAVSIM v2.0.0 → SampleIR → Enrich（2D-TGP 投影等）→ Render（LLaMA-Factory 数据） |
| `data_engine/configs/pipelines/` | `poutine_label` / `prompter` / `planner` 三 pipeline |
| `dvd/` | Driving-aware Vision Distillation 预训练 + 文档 |
| `sft/` | LLaMA-Factory SFT；支持 `enable_dvd`、`dvd_tgp_weight`、`dvd_ema` 等 YAML 参数 |
| `assets/` | 方法 overview 图 |

## 发布清单（README Release）

- [x] Configurable Data Engine
- [x] DVD code
- [x] LLaMA-Factory SFT configuration
- [x] RL code → [Curious-VLA](https://github.com/Mashiroln/curious_vla)（外置）
- [x] Annotations / Datasets → [Google Drive](https://drive.google.com/drive/folders/1oOz6EVfsrxGvOXvYkwxNhcgqjYP-LqSa?usp=drive_link)

## 可复现入口

```bash
# 数据引擎示例
python data_engine/main.py --config data_engine/configs/pipelines/planner.yaml

# DVD / SFT：见 dvd/README.md 与 sft/ 下 LLaMA-Factory YAML
```

## 推理架构（双模型）

1. **TGP-Prompter** — 前视图像 → 预测 **2D-TGP** 关键点序列
2. **TGP-Planner** — 前视 + 2D-TGP → **BEV 轨迹**

## 与机器人 / 驾驶 VLA 栈的关系

- **问题轴：** 相对纯 CoT/VQA 预训练的驾驶 VLA，本文在 **视觉编码器 + 空间 prompt** 上显式注入 **交通先验与可行轨迹 grounding**。
- **评测：** NAVSIM + nuScenes；与 [S²-VLA](../../wiki/entities/paper-s-squared-vla.md)（双流解耦）、[Depth-Wise Probing Driving VLA](../../wiki/entities/paper-depth-wise-probing-driving-vla.md)（层探针剪枝）同属 **2026 驾驶 VLA** 研究簇。
- **RL：** TGP-guided GRPO 实现外置 Curious-VLA，与 [GRPO 方法页](../../wiki/methods/grpo.md) 交叉。

## 对 Wiki 的映射

- **`wiki/entities/paper-driveteach-vla.md`**：论文实体与方法归纳。
- **`sources/papers/driveteach_vla_arxiv_2607_01658.md`**：论文级摘录。
