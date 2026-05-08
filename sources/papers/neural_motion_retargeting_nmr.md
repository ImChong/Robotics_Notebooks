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
- **入库日期：** 2026-05-08
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
