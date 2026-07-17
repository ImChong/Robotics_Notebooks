# LingBot-VLA：A Pragmatic VLA Foundation Model

> 来源归档

- **标题：** A Pragmatic VLA Foundation Model（LingBot-VLA 1.0）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2601.18692>
- **PDF（仓库内链）：** <https://github.com/robbyant/lingbot-vla/blob/main/assets/LingBot-VLA.pdf>
- **项目页：** <https://technology.robbyant.com/lingbot-vla>
- **代码：** <https://github.com/robbyant/lingbot-vla>
- **入库日期：** 2026-07-17
- **一句话说明：** Robbyant **务实 VLA 基础模型 1.0**：**2 万小时** 九类双臂真机预训练 + **Qwen2.5-VL-3B** 动作头，在仿真与 GM-100 真机 generalist 设定下报告优于 π₀.₅；强调 **训练效率**（相对既有 VLA 代码库约 **1.5–2.8×** 加速）与 **LeRobot 后训练** 范例。

## 核心摘录（策展）

### 1) 大规模双臂真机预训练

- **规模：** **20,000 h** 真机数据，覆盖 **9** 种主流双臂机器人构型。
- **务实取向：** 相对同期工作更强调 **数据规模 + 工程化训练栈**，而非单点算法 trick。

→ wiki 映射：

- [LingBot-VLA](../../wiki/entities/lingbot-vla.md)

### 2) 4B 模型与深度蒸馏变体

- **骨干：** **Qwen2.5-VL-3B-Instruct** + flow 去噪动作头。
- **变体：** **w/o depth** 与 **w/ depth**（LingBot-Depth 教师蒸馏）两路公开权重。
- **后训练：** RoboTwin 2.0 五任务范例 + **GM-100** generalist 联合训练 checkpoint。

→ wiki 映射：

- [LingBot-VLA](../../wiki/entities/lingbot-vla.md) — 权重表与部署
- [LeRobot](../../wiki/entities/lerobot.md) — v3.0 数据格式

### 3) 评测与对照

- **仿真：** RoboTwin 平均 SR **88.56% / 86.68%**（clean/rand，depth 变体）vs π₀.₅ **82.74% / 76.76%**。
- **真机 GM-100：** 项目页与 README 给出 progress/success 对照（AgileX Cobot Magic 等）；2.0 技术报告将其作为 **1.0 基线**。

→ wiki 映射：

- [LingBot-VLA 2.0](../../wiki/entities/lingbot-vla-v2.md) — 对照表中的 1.0 列
- [Manipulation](../../wiki/tasks/manipulation.md)

## 对 wiki 的映射

- [x] `wiki/entities/lingbot-vla.md` 新建
- [x] 与 `wiki/entities/lingbot-vla-v2.md` 区分代际
- [x] 修正 [loco_manip_161_survey_152_lingbot-vla.md](loco_manip_161_survey_152_lingbot-vla.md) 误链至 2.0 页
