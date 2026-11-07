# neural_motion_retargeting_nmr

> 来源归档（ingest）

- **标题：** Make Tracking Easy: Neural Motion Retargeting for Humanoid Whole-body Control
- **类型：** paper
- **来源：** arXiv abs / arXiv HTML
- **原始链接：**
  - <https://arxiv.org/abs/2603.22201>
  - <https://arxiv.org/html/2603.22201v3>
- **项目主页：** <https://nju3dv-humanoidgroup.github.io/nmr.github.io/>
- **作者：** Qingrui Zhao, Kaiyue Yang, Xiyu Wang, Shiqi Zhao, Yi Lu, Xinfang Zhang, Qiu Shen, Xiao-Xiao Long, Xun Cao（南京大学等）
- **官方代码：** <https://github.com/3DV-HumanoidGroup/MakeTrackingEasy>（2026-11-05 核对：推理代码与 checkpoint 已发布；CEPR 数据集与训练代码未发布；HF Space `RayZhao/NMR`）
- **入库日期：** 2026-05-08；2026-11-07（全文通读更新）
- **核对范围：** 通读 arXiv:2603.22201 v3 正文 §I–§V 与附录 A 的非凸性分析大意（摘要、引言、相关工作、非凸性命题 1、CEPR 三步数据管线、网络与两阶段训练、AMASS 测试集实验与表 III/IV、上游噪声修正、结论与局限）；附录 Hessian 推导仅浏览，未逐式核对。未运行官方代码（推理代码与 checkpoint 状态以本档案开头 2026-11-05 核对为准）。
- **通读：** 已完成
- **最后更新：** 2026-05-08
- **一句话说明：** 提出 **NMR（Neural Motion Retargeting）**：用「分布映射」替代逐帧几何优化，配合 **CEPR** 分层数据管线（聚类 + 并行 RL 专家 + 物理仿真精修）自动生成人机配对监督，训练 CNN–Transformer 式非自回归网络，在 Unitree G1 上显著抑制关节跳变与自碰撞并加速下游全身跟踪策略收敛。

## 核心摘录

### 1) Make Tracking Easy: Neural Motion Retargeting for Humanoid Whole-body Control（Zhao 等，2026）
- **链接：** <https://arxiv.org/abs/2603.22201>
- **问题：** 经典「先重定向、再跟踪」流水线里，基于 IK / 微分优化的重定向在 Hessian 意义下**高度非凸**，易陷局部最优，产生关节急跳、自穿模、脚滑等伪影；同时 SMPL 等人体估计噪声会被几何优化**机械放大**（garbage in, garbage out）。
- **核心思路：** 把重定向从「逐帧求最优位形」改写为「从人体运动分布到机器人**可行运动流形**上的学习映射」；为避免用劣质优化结果当监督，引入物理仿真里的 RL 跟踪专家生成**伪真值**配对数据。
- **CEPR（Clustered-Expert Physics Refinement）三阶段数据构造：**
  1. **物理感知人体数据筛选**：剔除过大 jerk、质心相对支撑基过远、脚–地接触不足等片段（对齐 PHUMA 类过滤思想）。
  2. **运动学重定向 + 硬阈值过滤**：用 **GMR** 得到初始机器人序列；再按关节速度峰值、MuJoCo 自碰撞帧占比、脚平均离地高度等规则剔除失败段。
  3. **基于物理的人形轨迹精修**：用 **TMR** 训练的运动–文本检索编码器提取序列特征，**K-Means（余弦距离）** 将运动库聚成语义相近簇；每簇单独训练对称 Actor–Critic + **PPO** 的全身跟踪专家（大规模并行仿真），对参考做跟踪 rollout，记录仿真中真实可达的机器人状态，与输入 SMPL 配对；论文报告约 **3 万**条高质量配对序列。
- **NMR 网络：** 人体侧采用类似 MotionMillion 的 **272 维**序列表示（根平面速度、根 6D 旋转、局部关节位置/速度等），机器人侧增加关节 DoF；**1D ResNet 编码 → 全连接自注意力 Transformer（非因果，并行预测整段）→ 上采样与 1D Conv 解码**；损失为全序列 **L1**。
- **两阶段训练：** 先在**大规模运动学**重定向数据上预训练（覆盖广、含残余物理伪影），再用 **CEPR 物理配对**子集微调，把输出分布拉向动力学可行区域；消融表明两阶段缺一不可（仅运动学则物理约束弱；仅物理数据则过拟合、泛化差）。
- **实验：** Unitree G1 上多种动态技能（武术、舞蹈等），相对强基线显著降低关节不连续、自碰撞与限位违反；NMR 生成的参考还加速下游全身控制策略训练。
- **对 wiki 的映射：**
  - 新建 [NMR（神经运动重定向与人形全身控制）](../../wiki/methods/neural-motion-retargeting-nmr.md)
  - 在 [Motion Retargeting](../../wiki/concepts/motion-retargeting.md)、[GMR](../../wiki/methods/motion-retargeting-gmr.md) 中补充「学习式重定向 + 用 RL 仿真修补监督」的定位与交叉引用

## 当前提炼状态

- [x] arXiv 摘要与方法主线（CEPR、网络、两阶段训练、G1 实验）已摘录
- [x] wiki 方法页与 Mermaid 流程图已落盘
- [x] 与 motion-retargeting / GMR 概念页交叉引用已补

## 全文核对要点（2026-11-07，均出自论文）

- **非凸性论证：** 单末端位姿误差的 Hessian = Gauss-Newton 半正定项 + 曲率修正项；命题 1：存在可行目标位姿与构型使 Hessian 有严格负方向曲率（来源：正向运动学的递归三角复合曲率 + SO(3)/SE(3) 对数映射非线性）。因此逐帧 IK / 微分 IK（线性化成凸 QP）对初值与权重敏感，无全局收敛保证。
- **CEPR 三步（§III-B）：** ①沿用 PHUMA 的物理一致性过滤（过大 jerk、CoM 远离支撑域、足地接触不足）；②用 **GMR** 做运动学重定向，再按硬阈值过滤——关节峰值速度超过硬件饱和速度、自相交帧占比超过 0.05、最低脚点平均离地超过 0.10 m 的序列被丢弃；③物理精修：用 **TMR**（文本-运动检索模型）的运动编码器提特征，**K-Means（余弦距离）** 聚类，每簇训练一个 PPO 跟踪专家（对称 actor-critic、观测维度共 509，奖励以跟踪项为主、仅"动作变化率"与"不期望接触"两项惩罚，$\sigma$ 从大到小线性收紧），rollout 得到约 **30K** 对 (SMPL 序列, 物理一致机器人序列)。注意：摘要和结论写"VAE 聚类"，方法节实际用 TMR + K-Means，原文前后表述不一致。
- **网络（§III-C/D）：** 人体用 272 维 MotionMillion 表示，机器人在此基础上加关节 DoF；1D-ResNet 编码器 + LLaMA 式 Transformer，**全自注意力非自回归**（人机时间严格一一对应），1D 上采样解码，L1 损失。两阶段：①在大规模运动学重定向数据上预训练（batch 128，lr $2\times10^{-4}$，500 epoch，余弦退火）；②用 CEPR 数据微调（lr $10^{-5}$，50 epoch）。消融：只预训练得到运动学合理但物理不受约束的输出；只用物理数据会过拟合 RL 语料覆盖的动作。
- **评测（Unitree G1，AMASS 82 段共 119K 帧 @120 Hz，按动作复杂度 / 长度分层）：** 基线 GMR、PHUMA。表 III（关节跳变 / 关节极限 / 自碰撞，按原文文字陈述解读）：NMR 关节跳变 0（GMR 56，约 0.11%）；自碰撞帧比 GMR 少 54%（0.87% vs 1.91%）；关节极限帧占比 16.80%（GMR 43.21%，约少 61%，PHUMA 21.32%）。注：PDF 文本抽取使列顺序不易辨认，此处以正文 / 结论的叙述为准。下游 BeyondMimic 跟踪（表 IV，成功率 / MPJPE / W-MPJPE，短 / 中 / 长序列）NMR 均最好（如短序列成功率 NMR 31/33，另两种方法为 26/33 与 25/33；原文表格未在抽取文本中清晰标注哪列对应哪种基线），PHUMA 在短序列 W-MPJPE 0.660 m（NMR 0.237 m），作者解释为序列级优化扭曲了短动作。训练曲线：NMR 参考使 episode 更长、奖励更高。
- **上游噪声：** 训练数据经物理过滤，SMPL 抖动被视作分布外，网络倾向输出连续可行序列；双向注意力利用异常帧前后信息做隐式插值；GMR 逐帧处理会把噪声原样传下去。
- **局限（原文）：** CEPR 是**机型特定**的，换平台要重新生成数据管线；形态条件化的架构留待未来。论文**未给出推理耗时数字**（"无需物理仿真器即可高效推理"），旧版 wiki 的"<10 ms/段"没有原文依据。
