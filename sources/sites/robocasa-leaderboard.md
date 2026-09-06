# RoboCasa365 Leaderboard

- **标题：** RoboCasa365 Leaderboard
- **类型：** site / 公开排行榜
- **链接：** https://robocasa.ai/leaderboard.html
- **代码：** https://github.com/robocasa/robocasa（评测栈已开源）
- **快照日期：** 2026-09-01（页面标注 Updated 09/01/2026）
- **入库日期：** 2026-09-06
- **一句话说明：** 50 任务多任务学习公开榜：Overall = Atomic-Seen + Composite-Seen + Composite-Unseen 三拆分成功率均值；Human300 预训练后评 50 target 任务。
- **沉淀到 wiki：** 是 → [`wiki/entities/robocasa.md`](../../wiki/entities/robocasa.md)

---

## 榜规模（页面头部）

- **50 Tasks** 多任务 benchmark
- **13 Models** 已评测（截至快照日）
- **3 Evaluation Splits**

---

## Overall 排名（2026-09-01）

| Rank | Policy | Overall | Atomic-Seen | Composite-Seen | Composite-Unseen | Open Source |
|------|--------|---------|-------------|----------------|----------------|-------------|
| 1 | Xiaomi-Robotics-1 | 57.4 | 80.2% | 57.1% | 32.1% | ✓ |
| 2 | ABot-M0.6 | 46.6 | 79.4% | 48.3% | 7.9% | |
| 3 | ABot-M0.5 | 40.3 | 75.6% | 37.7% | 3.3% | |
| 4 | PRTS | 39.6 | 66.3% | 30.3% | 18.8% | ✓ |
| 5 | RLDX-1 | 36.0 | 67.6% | 27.9% | 8.5% | ✓ |
| 6 | WorldDreamer | 35.3 | 66.3% | 26.7% | 9.0% | ✓ |
| 7 | GR00T N1.5 | 23.9 | 50.7% | 14.8% | 2.7% | ✓ |
| 8 | GR00T N1.6 | 21.9 | 51.1% | 9.4% | 1.7% | ✓ |
| 9 | GigaWorld-Policy 0.1 | 20.7 | 44.4% | 11.8% | 2.9% | ✓ |
| 10 | π0.5 | 16.9 | 39.6% | 7.1% | 1.2% | ✓ |
| 11 | π0 | 14.8 | 34.6% | 6.1% | 1.1% | ✓ |
| 12 | Azero-Robotics-1 | 12.6 | 30.3% | 3.8% | 1.6% | |
| 13 | Diffusion Policy | 6.1 | 15.7% | 0.2% | 1.3% | ✓ |

**注：** GR00T N1.5 按 RoboCasa **v1.0.1**（horizon 1.5×）重评；训练配置仅供透明，**不可跨架构直接比**。

---

## 评测协议摘要

- **训练：** Human300 预训练集（300 任务 × 2500 预训练厨房）
- **评测：** 50 target 任务于预训练厨房
- **Atomic-Seen / Composite-Seen：** 预训练见过
- **Composite-Unseen：** 预训练**未见过**的复合任务 → 零样本泛化
- **Overall：** 三拆分任务成功率 published average

首发聚焦 Diffusion Policy、π₀、π₀.₅、GR00T N1.5 四大家族；持续接受用户提交审核上榜。

---

## 对 wiki 的映射

- [RoboCasa](../../wiki/entities/robocasa.md)
