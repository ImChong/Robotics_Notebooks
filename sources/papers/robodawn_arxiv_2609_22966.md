# RoboDawn（arXiv:2609.22966）

> 论文来源归档

- **标题：** Transferring the Intelligence of VLMs to Robotic Control
- **类型：** paper / vla / agentic / manipulation / frozen-policy
- **机构：** 清华大学（Tsinghua）；腾讯混元（Tencent Hunyuan）
- **链接：** <https://arxiv.org/abs/2609.22966> · [PDF](https://arxiv.org/pdf/2609.22966)
- **项目页：** <https://robodawn.top> · [全部评测轨迹](https://robodawn.top/results)
- **代码：** <https://github.com/Hugo-AGI/RoboDawn>（MIT，2026-09-22 发布）
- **入库日期：** 2026-09-24
- **一句话说明：** 冻结 agentic VLM + 人类直觉语义动作接口（GIP 平移/旋转/夹爪）+ ICL 演示；零训练机器人参数；RoboTwin 2.0 C2R 53.2%→73.6%（1-shot），RoboDojo 35.67%→47.17%；Franka 真机零样本 9/10。

## 核心摘录（面向 wiki 编译）

### 1) 语义动作接口 + 闭环

- **要点：** GIP（指尖中点）上的离散 `move/rotate/point/gripper/home/wait/done`；单步 ≤20 cm / 90°；每命令规划为完整运动至静止；Observe→Reason→Act→Adapt，无特权物体位姿。
- **对 wiki 的映射：** [`wiki/entities/paper-robodawn.md`](../../wiki/entities/paper-robodawn.md)

### 2) ICL：command primer + 任务演示

- **要点：** $D = D_{\mathrm{prim}} \oplus D_{\mathrm{task}}$；专家轨迹转写成语义命令序列 + VLM  rationale；0/1/N-shot；1-shot 增益最大，8-shot 略降（长上下文）。
- **对 wiki 的映射：** 同上；对比 [`paper-harness-vla`](../../wiki/entities/paper-harness-vla.md)（冻结 VLA 原语 + 记忆编排）

### 3) 评测与开源资产

- **RoboTwin 2.0 C2R：** 50 任务 ×10；1-shot GPT-6 Astra **73.6%** > π₀.₅ **46.0%**（全量后训练）。
- **RoboDojo：** 42 任务；GPT-6 Astra 1-shot **47.17%**（zero **35.67%**）。
- **真机：** Franka block-in-basket / stacking，Gemini 3.8 Flash 零样本。
- **开源：** harness、prompts、128 ICL demos、evaluation seeds、**710** 可回放 episode（[results 浏览器](https://robodawn.top/results)）。

## 当前提炼状态

- [x] 项目页 + GitHub 开源核查（2026-09-24）
- [x] wiki 实体 + 交叉 RoboTwin / RoboDojo / VLA
