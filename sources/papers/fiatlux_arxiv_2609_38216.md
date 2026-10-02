# Fiatlux：人形攀梯换灯长时程基准

- **标题：** Fiatlux: A Long-Horizon Benchmark for Humanoid Ladder Climbing and Light-Bulb Replacement
- **类型：** paper
- **论文：** <https://arxiv.org/abs/2609.38216>（2026-09-27，v1）
- **机构：** 夏威夷大学马诺阿分校 HawAII、独立研究者、普渡大学。
- **项目页：** <https://fiatlux-bench.github.io/>；[项目页核查](../sites/fiatlux.md)
- **代码：** <https://github.com/haw-ai-i/fiatlux>；[仓库归档](../repos/fiatlux.md)
- **遥操作数据：** <https://huggingface.co/datasets/haw-ai-i/fiatlux-teleoperation>；[数据归档](../datasets/fiatlux-teleoperation.md)
- **入库日期 / 最后更新：** 2026-10-02
- **一句话说明：** Isaac Lab 中的 G1 长时程维护基准，连接搬梯、攀爬、换灯与旧灯处置；攀爬可达性仍未完整证明。

## 核心摘录

### 1. 从单技能到整段任务

完整任务 `FIATLUX-Replace-v0` 分成 S01–S12 十二个子环境，覆盖平地移动、操作与攀爬；子环境可单独重置，不能把单项成功等同于连续全程成功。

**对 wiki 的映射：** [Fiatlux](../../wiki/entities/paper-fiatlux.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)。

### 2. 评分与安全边界

难度加权门控评分允许部分进度；需同时报告成功、干净成功、掉落和破损。标准观测与仿真特权观测分离。灯泡固定机制为状态机与外力保持，尚非真实螺纹连接器模型。

**对 wiki 的映射：** [Fiatlux](../../wiki/entities/paper-fiatlux.md)、[具身评测入口](../../wiki/overview/hub-embodied-eval-benchmark.md)。

### 3. 当前成果与未完成项

论文/项目页报告八个非攀爬子任务共 107 个遥操作 takes，其中 80 个满分；四个攀爬子任务 S02/S04/S10/S12 缺少通过门控的示范。已发布 GR00T N1.7 零样本、zero、random 三组基线在每个子任务的成功率均为零，非零分数来自部分进度。PPO 训练入口存在，但不是成功解决整链的证据。

当前 HF 数据卡另外列出 18 个攀爬尝试，总计 **125 episodes / 3.7 GB**；这与论文的 107 个非攀爬 takes 是不同统计范围。数据为仿真遥操作记录，不是权重或真机数据。

**对 wiki 的映射：** [Fiatlux](../../wiki/entities/paper-fiatlux.md)。

## 开源核查

2026-10-02 打开项目页并检查官方 README：代码、仿真资产、遥操作记录、基线 rollout 已提供公开入口；真机 SDK 适配器仍为 future work。详见[项目页核查](../sites/fiatlux.md)。
