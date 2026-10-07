# Humanoid Rickshaw Pulling 来源归档

- **论文：** https://arxiv.org/abs/2610.04238
- **HTML v1：** https://arxiv.org/html/2610.04238v1
- **PDF：** https://arxiv.org/pdf/2610.04238
- **演示视频：** https://youtu.be/eqnAlQLjZF8
- **作者：** Yangzhi Yang, Xiansheng Lin, Zhaoming Xie, Xiaobin Xiong
- **机构：** Legged AI Lab, Shanghai Innovation Institute
- **代码：** arXiv v1 未列独立 GitHub 代码仓库

## 方法摘录

G1 以双手持续握把与被动两轮车耦合。三阶段管线：S0 用车辆状态、交互力和负载参数训练特权教师；S1 用本体历史学生预测交互 latent 并蒸馏动作；S2 用 PPO 微调适应 latent 估计误差。策略 50 Hz 输出 29 维关节位置目标，由 PD 跟踪。

训练采用 MuJoCo mjlab、8,192 并行环境、两张 NVIDIA H200，约 11 小时。实机使用定制 6061 铝末端执行器；车把反力协助推进、转向及身体平衡。

## 实验口径

- 仿真质量扫描 20–120 kg、速度命令 0.6–2.0 m/s；随机化训练质量范围为 20–60 kg。
- 实机装载后车辆总质量 60、90、115 kg，同一策略执行起步、持续牵引、转向和停止。
- 115 kg 指 loaded rickshaw mass，不是额外手提重量。
- Cost-of-Transport 是基于关节功率的代理指标，不是电池端实测能耗。

## 边界

实验从已经双手握住把手开始；抓握获取/恢复、导航和电能测量列为后续工作。此归档引用论文与作者视频，不将其误标为已开放代码仓库。
