# Learning While Deploying（LWD，arXiv:2605.00416）

> 来源归档（paper）

- **标题：** Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2605.00416>
- **PDF：** <https://arxiv.org/pdf/2605.00416v1>
- **项目页：** <https://finch.agibot.com/research/lwd>（论文首页给出；未打开，仓库未见代码链接）
- **作者与机构（论文首页）：** Yi Wang 等；Shanghai Innovation Institute、AGIBOT Finch、Columbia University；通讯作者 Jianlan Luo
- **入库日期：** 2026-11-12
- **一句话说明：** 把 16 台双臂机器人的自主 rollout 与人工干预汇入共享回放，用 DIVL 学分布式价值、用 QAM 抽取策略，对 flow 式 VLA（π0.5 架构）做离线到在线的持续 RL 后训练。
- **核对范围：** 2026-11-12 下载 arXiv v1 PDF 并用 pdftotext 抽取全文，逐段读完正文（摘要、第 I 至 VII 节）、参考文献区与附录 A（A1 离散化、A2 命题证明、A3 直接反传分析）、B（离线数据、超参、检查点初始化）、C（基线实现、完整消融 Table V、DIVL 定性结果）、D（分布式数据基础设施、Table VI）；图 1、3、4、6、9、10 的图像本身未目视，仅读了图注与抽取出的图内文字和柱状图数值；未打开项目页、未见补充视频，未复现任何实验。
- **通读：** 已完成

## 核心摘录

1. **任务（Sec. V-A）：** 8 个真机任务，4 个杂货补货（二值成功率）与 4 个 3–5 分钟长程任务（功夫茶、果汁、鸡尾酒、装鞋盒，按 5–8 个子步骤打 1 / 0.5 / 0 分取平均，由人工评分员按预设细则打分）。平台为 AgiBot G1 双臂（每臂 7 自由度，3 个 RGB 相机，30 Hz 关节位置控制）。
2. **算法（Sec. IV）：** DIVL 用 C51 式分类分布（K=201 个原子，区间 [-0.1, 1.1]，附录 A1）表示状态条件下回放动作 Q 值分布，取 τ 分位作为 TD 自举目标，τ 由归一化熵自适应（式 17–18）；QAM 把 critic 动作梯度转成沿参考 flow 的逐步回归目标（式 8–10）。离线与在线用同一 RL 目标。
3. **主结果（Table I）：** 8 任务平均：SFT 0.76、RECAP 0.85、HG-DAgger 0.85、LWD 离线 0.88、LWD 在线 0.95；长程任务平均 0.91 对 SFT 0.68、RECAP 0.77、HG-DAgger 0.73、LWD 离线 0.79（Sec. V-B）。
4. **预算（Sec. V-A）：** 在线实验每个方法 4 小时墙钟、约 60 机器人小时；16 台机器人中 4 台做杂货、每个长程任务 3 台；每 50 个训练步向机队广播一次策略。
5. **消融（Table II/III/V）：** DIVL 对标量 expectile 回归，长程在线 0.78 到 0.91；自适应 τ 对常数 τ=0.52，离线平均 0.84 到 0.88。
6. **数据（Table IV）：** 离线缓冲 652.5 小时，其中示范 336.6、成功 rollout 88.8、失败 rollout 39.2、play 187.9 小时。
7. **作者自陈局限（Sec. VI）：** 在线更新调度较朴素；长程任务只用一条短指令；未显式建模执行安全。

## 对 wiki 的映射

- [paper-lwd-fleet-scale-rl](../../wiki/entities/paper-lwd-fleet-scale-rl.md)
