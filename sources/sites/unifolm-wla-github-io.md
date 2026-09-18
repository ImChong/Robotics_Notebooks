# unigen-x.github.io/unifolm-wla.github.io（UnifoLM-WLA-1.0 项目页）

- **标题：** UnifoLM-WLA-1.0: One Model Drives All, Whole-Body Coordination
- **类型：** site / project-page
- **URL：** <https://unigen-x.github.io/unifolm-wla.github.io/>
- **代码：** <https://github.com/unitreerobotics/unifolm-wla>
- **模型：** <https://huggingface.co/collections/unitreerobotics/unifolm-wla-10>
- **数据集：** <https://huggingface.co/collections/unitreerobotics/unifolm-wla-10>
- **机构：** 宇树科技（Unitree）
- **入库日期：** 2026-09-18

## 一句话摘要

UnifoLM 系列 **6B 通用人形基础模型** 官方项目页：以 **UnifoLM-ER-1** 具身推理为起点，经 **动态区域预测** 与 **离散动作 RVQ** 得到 **UnifoLM-ER-Flow**，再叠 **MMDiT Action Expert** 训练 **UnifoLM-WLA-1.0**；约 **2,500 小时** 真机数据、**64 任务**（10 全身 + 54 桌面），支持二指夹爪与多种五指灵巧手。

## 开源状态（步骤 2.5，2026-09-18）

| 资源 | 状态 |
|------|------|
| 项目页演示 / 方法说明 | **已发布** |
| GitHub 仓库 | **已链出** — [unitreerobotics/unifolm-wla](https://github.com/unitreerobotics/unifolm-wla)（README + 动作/状态处理规范文档） |
| **后训练代码** | **待发布** — Open-Source Plan 中 Post-Train Code 未勾选 |
| Hugging Face 权重 | **部分发布** — `UnifoLM-ER-1`、`UnifoLM-ER-Flow` 已发布；`UnifoLM-WLA-Base` 待发布 |
| 开源数据集 | **部分发布** — `UniBot-V1 Challenge Dataset` 已发布；`Unitree-WBT-Dataset`、`Unitree-Manipulation-Dataset` 待发布 |

## 架构要点（编译自项目页）

1. **UnifoLM-ER-1（Embodied Reasoning）**：基于 Qwen3-VL-4B，500 万+ 样本联合通用图文与具身空间任务（点预测、检测、多图推理、2D/3D 轨迹、空间 QA 等）；16 项 benchmark 中 7 项开源领先。
2. **Dynamic Region Prediction**：光流提取未来动态区域 → VQ-VAE 离散 mask token；VLM 条件预测未来交互区域变化。
3. **Discrete Action Learning（→ UnifoLM-ER-Flow）**：统一动作空间拆为 EEF 位姿、末端关节、下肢关节三路 **RVQ** 离散化，与 mask token 联合对齐视觉–语言–动作。
4. **UnifoLM-WLA-1.0**：在 ER-Flow 多模态骨干上接入 **MMDiT Action Expert**，约 2,500h 真机数据（Unitree Open Datasets、BitRobot-HIW-500 等）训练；联合语言预测、离散动作 token 与连续 flow 动作解码。

## 关联资料

- 代码归档：[`sources/repos/unifolm-wla.md`](../repos/unifolm-wla.md)
- 沉淀实体：[`wiki/entities/unifolm-wla.md`](../../wiki/entities/unifolm-wla.md)
- 同族 VLA / WMA：[`wiki/entities/unifolm-vla.md`](../../wiki/entities/unifolm-vla.md) · [`wiki/entities/unifolm-world-model-action.md`](../../wiki/entities/unifolm-world-model-action.md)
