# HumanTracker 论文通读（运控评测指标视角）

> 来源归档（原文通读 · 支撑 `wiki/concepts/motion-control-policy-evaluation-metrics.md`）

- **标题：** HumanTracker: Towards Comprehensive and Human-Aligned Motion Tracking Benchmark
- **作者：** Dairu Liu, Zekun Qi, Jiayu Zeng 等（Nankai / Tsinghua / Galbot / SJTU / PKU / Shanghai Qi Zhi Institute；通讯 He Wang, Li Yi）
- **类型：** paper / benchmark / humanoid-motion-tracking
- **原文：** <https://arxiv.org/abs/2608.13555>（v1，2026-08-13；通过 arXiv PDF 获取全文）
- **一句话说明：** 提出约 153 h、四个动作族的光学动捕跟踪基准，并用专家成对偏好训练轨迹级奖励模型 HumanScore，指出运动学误差与人的观感脱节。
- **核对范围：** 通读正文 §1–§5（引言、相关工作、基准构建、标准化评测、偏好数据、HumanScore、实验、敏感性分析、结论）及附录 A–G（实现细节、特征、数据规模与格式、覆盖度分析、对齐率置信区间、讨论与局限）；参考文献列表仅浏览，未逐篇核对被引论文。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题（§1）：** 运动学误差逐帧平均，漏掉接触、支撑、平衡与闭环误差累积；两条 rollout 的 MPJPE 相近，观察者仍稳定偏好接触干净、支撑稳的那条。常用 AMASS 测试集仅 140 条序列，且常只给单一聚合分。
- **基准（§3.1，表 2、附录 D）：** 24 位专业表演者、光学动捕，经 GMR 重定向并剔除悬浮 / 穿地 / 接触不连续片段；四族 Daily 89 h / 9.7k、Highly Dynamic 11 h / 2.7k、Interaction 48 h / 10.9k、Ground 5 h / 1.6k，合计约 153 h、25K 条；族按暴露的失败机制划分，分布反映采集频率而非均衡，结果须按族与总体同时报；9:1 切分训练 / 测试，同源动作不跨分区；轨迹 50 Hz，qpos 为 36 维、29 个主动关节。
- **标准化评测（§3.2）：** 各 tracker 保留自身观测与动作解码，评测器统一动作列表、机器人模型、参考索引、rollout 记账与指标实现；成功判据沿用 SONIC：骨盆 / 双踝 / 双腕垂向误差 >0.25 m、骨盆旋转误差 >1 rad 或 qpos / qvel 非有限即失败；Succ 为完成回合比例，MPJPE 为 29 个主动关节角的平均绝对误差（rad），只在已执行片段上取平均；另有关节速度误差、关键点位置误差、足接触一致性、加速度与 jerk 等诊断量。
- **偏好数据（§3.3、附录 C）：** 仅用训练划分动作；GMT / Humanoid-GPT / SONIC / TWIST2 对同一参考各出 rollout，切成 250 帧（5 s）窗口，均匀抽样并均衡六种 tracker 配对；6 位机器人方向博士标注，先判是否失败 / 失衡，再依次比较抖动、脚滑、行走一致性、整体自然度；标签含 strict / Similar / Cannot compare；6000 对原始轨迹对镜像后得 12000 条（24K 条运动），按 motion_id 8:2 切分。
- **HumanScore（§3.4–3.5、附录 A）：** 逐帧 539 维 token（当前参考 70 维 + rollout 469 维，含接触力 / 足速度等特权仿真量），4 层 8 头 Transformer（d=256），masked mean pooling + MLP 得标量；Bradley–Terry 损失 + Similar 对的对称约束；推理时窗口奖励经 sigmoid，再按真实帧数加权得 0–100 分。
- **结果（§4.2，表 3）：** 零样本、测试划分。Humanoid-GPT 整体最强（Daily Succ 94.4 / MPJPE 0.046 / HumanScore 54.7）；SONIC 在 Interaction 的 Succ 最高、在 Ground 的 HumanScore 最高；GMT、TWIST2 在 Ground 的 Succ 为 0.0，SONIC 20.1，Humanoid-GPT 32.9。传统指标与 HumanScore 排名并不总一致。
- **与人类偏好的对齐（§4.3，表 4、附录 F 表 9）：** HumanScore 90.83%（95% CI 87.36–93.83），高于 MPJVE 84.04、关键点位置 MAE 84.05、MPJPE 80.49、足接触准确率 78.82、关节 jerk 72.32、加速度 69.33（按族平均后再等权平均）。
- **敏感性（§4.4，图 5）：** 去掉接触特征在 Ground 族退化最明显（0.759 vs 基线 0.879）；加入未来参考残差略差于基线；可用时间上下文从 1 s 增至 5 s，对齐率单调上升。
- **局限（附录 G）：** HumanScore 仅测试了对未见动作的泛化，不含未见机器人 / 仿真器 / 控制器；输入含真机不可得的特权量，上真机需可观测特征集或状态估计；每对仅一个主要判断，未量化重复标注不确定性；Ground 族样本较少；单一 29-DoF 人形、MuJoCo；仅用于评测，直接当 RL 奖励优化可能被钻空子；应与 Succ、运动学误差并列报，不能取代。

## 对 wiki 的映射

- [`wiki/concepts/motion-control-policy-evaluation-metrics.md`](../../wiki/concepts/motion-control-policy-evaluation-metrics.md)
