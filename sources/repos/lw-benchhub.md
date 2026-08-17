# LW-BenchHub（光轮官方仓）

> 来源归档

- **标题：** LW-BenchHub
- **类型：** repo
- **链接：** https://github.com/LightwheelAI/LW-BenchHub
- **机构：** 光轮科技（Lightwheel）
- **许可：** Apache-2.0（README 徽章与 LICENSE 声明；GitHub API `license` 字段为空，以仓内文件为准）
- **Stars：** ~192（2026-08-17）
- **文档：** https://docs.lightwheel.net/lw_benchhub
- **项目页：** https://lightwheel.ai/lightwheel-platform
- **入库日期：** 2026-08-17
- **一句话说明：** 光轮在 Isaac Lab-Arena 上的具身仿真评测底座：多本体厨房任务、遥操作采数、RL/IL 与 EnvHub 策略评测。
- **交叉归档：** [LW BENCHHUB TOUR](lw_benchhub_tour.md)、[Lightwheel Platform](../sites/lightwheel-platform.md)
- **沉淀到 wiki：** 本条不单独升格；工程闭环见 [lw-benchhub-tour](../../wiki/entities/lw-benchhub-tour.md)

---

## 步骤 2.5

**已开源。** 官方 README 给出 `install.sh`、遥操作 / replay / RL `train.sh` `eval.sh`；数据集在 Hugging Face [LightwheelAI/datasets](https://huggingface.co/LightwheelAI/datasets)。Tour 仓把它当作 **L3 任务内容层**（机器人 + 场景 + 任务），经 EnvHub 接到 `lerobot-eval`。

---

## README 规模（2026-08 快照）

| 项 | 数字 |
|----|------|
| 任务 | 268（LIBERO 130 + RoboCasa 138） |
| 本体族 | 7 类、27 变体（含 Double Piper、G1、Franka、SO100/101 等） |
| 厨房布局 × 风格 | 10 × 10 |
| 演示集 | 219 任务 × 4 本体，21,500 episode / 20,537,015 帧 |

Tour 复现钉的是 Isaac Lab **2.3.2** + Isaac Sim **5.1**；官方 README 徽章写 Isaac Lab **5.0.0**——版本线不要混用。
