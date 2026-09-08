# VLA-Precision（arXiv:2609.04355）

> 来源归档（ingest）

- **标题：** VLA-Precision: Asymmetric Co-Bootstrapping for Efficient Real-World Online RL of Vision-Language-Action Models
- **简称：** VLA-Precision
- **类型：** paper / vla / online-rl / precision-manipulation / real-robot
- **arXiv：** <https://arxiv.org/abs/2609.04355>
- **PDF：** <https://arxiv.org/pdf/2609.04355>
- **项目页：** <https://vla-precision.github.io/> — 归档见 [`sources/sites/vla-precision-github-io.md`](../sites/vla-precision-github-io.md)
- **代码：** <https://github.com/scy-v/VLA-Precision> — 归档见 [`sources/repos/scy-v-vla-precision.md`](../repos/scy-v-vla-precision.md)
- **机构：** 中国科学技术大学（USTC）自动化系
- **入库日期：** 2026-09-08
- **一句话说明：** 真机 VLA 在线 RL：ACoB 非对称共自举 + ACoB-Stream 闭环架构；九项精密化学任务、四机型平均成功率 98.3%，吞吐最高 10.9× 基线。

## 开源状态（步骤 2.5，2026-09-08）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线，链 GitHub |
| GitHub | **已开源** Apache-2.0；Stage I OpenPI 微调 + Stage II ACoB 在线 RL 全栈 |
| 遥操作 | 配套 UR/Franka LeRobot 分支（README 表格） |

**结论：已开源** — 训练、部署、评测入口完整。

## 核心摘录

### 摘录 1：ACoB 算法

- 真机在线 RL 两大瓶颈：不可靠 value 致策略漂移；大 VLA 吞吐低。
- ACoB 跨时间尺度非对称共自举：早期干预引导行为学习快速提升；经验积累后全局 return + 局部偏好排序校准 value，reference-regularized 策略改进抑制漂移。

**对 wiki 的映射：** [paper-vla-precision](../../wiki/entities/paper-vla-precision.md)

### 摘录 2：ACoB-Stream 与实验

- invariant-state decoupling + on-demand streaming → 吞吐/算效 **最高 10.9×**。
- 九项精密化学操纵（移液枪、比色皿、试管刷等），四类接触属性 × 四机器人 embodiment。
- 平均成功率 **98.3%**，**45.8 min/task** 训练预算；episode **27.6 s**，速度为 VLA/RL 基线 **1.2× / 1.8×**。

**对 wiki 的映射：** [paper-vla-precision](../../wiki/entities/paper-vla-precision.md)

## 当前提炼状态

- [x] 项目页、GitHub README 核查（2026-09-08）
- [x] wiki 映射：`wiki/entities/paper-vla-precision.md`
