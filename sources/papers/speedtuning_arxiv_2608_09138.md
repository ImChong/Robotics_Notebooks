# SpeedTuning: Speeding Up Policy Execution with Lightweight Reinforcement Learning（arXiv:2608.09138）

> 来源归档（ingest）

- **标题：** SpeedTuning: Speeding Up Policy Execution with Lightweight Reinforcement Learning
- **缩写 / 框架：** **SpeedTuning**
- **类型：** paper / imitation-learning / lightweight-rl / manipulation
- **arXiv：** <https://arxiv.org/abs/2608.09138>
- **会议：** ICRA 2025（DOI [10.1109/ICRA55743.2025.11128753](https://doi.org/10.1109/ICRA55743.2025.11128753)）
- **项目页：** <https://daivdyuan.github.io/speed-tuning/>（归档见 [`sources/sites/speed-tuning-github-io.md`](../sites/speed-tuning-github-io.md)）
- **代码：** <https://github.com/DaivdYuan/SpeedTuning>（归档见 [`sources/repos/speedtuning.md`](../repos/speedtuning.md)）
- **作者：** David D. Yuan、Tony Z. Zhao、Kaylee Burns、Chelsea Finn
- **机构：** 斯坦福大学（Stanford）
- **入库日期：** 2026-08-17
- **一句话说明：** 冻结模仿基座策略，用轻量 RL 只预测动作执行的速度倍率；不额外采数据，倒/抛/取等任务上超过 2.4× 加速并保持足够成功率。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-17）：** [daivdyuan.github.io/speed-tuning](https://daivdyuan.github.io/speed-tuning/) 有摘要、示意与 ICRA 视频；配套静态站仓 `DaivdYuan/speed-tuning` 指向代码仓。
- **代码仓：** [DaivdYuan/SpeedTuning](https://github.com/DaivdYuan/SpeedTuning)（MIT）是 **仿真复现发布**：脚本化基座策略 + Rainbow DQN 速度策略；入口 `scripts/train_speed_policy.py`、`scripts/eval_speed_policy.py`、`scripts/run_sim.py`。
- **结论：** **已开源、可运行仿真训练/评测**。真机 ACT 集成脚本 `act_integration.py` 在仓内，完整真机数据不随仓发布。

## 摘录 1：问题与接口

模仿策略速度被示教者与采集硬件钉死；全局固定倍速插值会在接触关键帧牺牲成功率。SpeedTuning 把速度当成 **独立控制维**：基座继续出动作，速度策略只选倍率。

## 摘录 2：方法

轻量 RL（仿真复现用 Rainbow DQN、离散倍率）吃当前机器人/任务观测，输出 speed multiplier。关键交互（抓茶包）降到约 2×，过渡段可到 4×。不重新采集示教。

## 摘录 3：数字

- 论文：动态与精细任务上 **>2.4×** 速度提升，相对原策略与固定插值仍保持足够成功率。
- 仿真任务预设：`scripted-pick-and-place` / `scripted-insertion` / `scripted-tea-bag`（另有随机位姿变体）。

**对 wiki 的映射：** [`wiki/entities/paper-speedtuning.md`](../../wiki/entities/paper-speedtuning.md)；交叉 [模仿学习](../../wiki/methods/imitation-learning.md)、[强化学习](../../wiki/methods/reinforcement-learning.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（仿真仓可运行）
