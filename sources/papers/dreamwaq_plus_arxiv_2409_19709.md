# DreamWaQ++: Obstacle-Aware Quadrupedal Locomotion With Resilient Multi-Modal Reinforcement Learning（arXiv:2409.19709）

> 来源归档（ingest）

- **标题：** DreamWaQ++: Obstacle-Aware Quadrupedal Locomotion With Resilient Multi-Modal Reinforcement Learning
- **缩写：** **DreamWaQ++**
- **类型：** paper / 四足 locomotion / 多模态 RL / sim2real
- **arXiv：** <https://arxiv.org/abs/2409.19709>（PDF：<https://arxiv.org/pdf/2409.19709>）
- **期刊：** IEEE Transactions on Robotics（T-RO），vol. 42, pp. 819–836, **2026**
- **项目页：** <https://dreamwaqpp.github.io/>
- **演示视频：** <https://youtu.be/DECFbMdpfps>
- **前序会议版：** DreamWaQ（ICRA 2023，<https://arxiv.org/abs/2301.10602>）
- **作者：** I Made Aswin Nahrendra†, Byeongho Yu, Minho Oh, Dongkyu Lee, Seunghyun Lee, Hyeonwoo Lee, Hyungtae Lim, Hyun Myung（KAIST Urban Robotics Lab / KRAFTON / URobotics / MIT LIDS）
- **入库日期：** 2026-05-30
- **一句话说明：** 在 **非对称 Actor–Critic + PPO** 框架下，用 **分层外感知记忆 + PointNet 置信滤波 + MLP-Mixer 多模态融合 + 对比/VAE 辅助损失**，单阶段训练出 **本体+外感知（3D 点云）** 四足控制器，在楼梯/陡坡/大障碍与传感器失效下保持鲁棒。

## 摘要级要点

- **问题：** 纯本体盲走需 **足端碰撞** 才能感知障碍；纯外感知又依赖 **精确时序地图** 且与控制环 **频率异步**（外感知约 10 Hz vs 本体/控制 50–200 Hz）。
- **主张：** **弹性多模态 RL**——在仿真中单阶段联合训练感知管线与控制策略；真机 **Go1** 等平台上 **零微调** 部署；外感知输入为 **传感器无关的 3D 点云**（深度相机或 LiDAR）。
- **外感知记忆：** 将最近 $K$ 帧点云经 **IMU + 估计体速积分** 的 $SE(3)$ 对齐到当前机体坐标系，在 **不做 U-Net 场景重建** 的前提下获得 **时序稠密** 外感知（论文式 (2)(3)）。
- **外感知编码：** **PointNet** 骨干 + **置信滤波层**（用点云统计方差抑制离群 max-pool 特征）→ 外感知上下文 $\mathbf{z}^e_t$。
- **本体编码：** 基于 DreamWaQ 的 **CENet**，FC 换 **MLP-Mixer**（$H{=}5$ 帧 @ 50 Hz）；**$\beta$-VAE** 随机潜变量 + **体速估计** $\hat{\mathbf{v}}_t$（供记忆对齐与 bootstrap）。
- **多模态融合：** 分模态 LayerNorm 后 **MLP-Mixer** 融合 → $\mathbf{z}^{pe}_t$；**约束重参数化** 限制 $\sigma$ 范围以稳定早期训练。
- **学习目标：** PPO 主损失 + $\mathcal{L}_{est}$ + 本体/外感知 **VAE 重建**（高度扫描 $\mathbf{h}_t$）+ **自适应 $\beta$ 调度** + **对比损失**（对齐 actor 观测潜变量与 critic 特权潜变量）+ **versatility gain**（鼓励探查/技能发现，如 **probing**）。
- **控制接口：** 策略 **50 Hz** 输出关节位置目标，**200 Hz PD** 跟踪力矩；**非对称 AC**：actor 用部分观测 + 多模态上下文，critic 用仿真特权状态。
- **项目页亮点（相对盲走 DreamWaQ）：** 50 级楼梯竞速 **35 s / 30 m 水平 / 7.4 m 爬升**；困难楼梯成功率 **97.8%**；训练坡度 **10°** 外推至 **35°** 爬行步态；四平台（Go1+RealSense/Ouster/Livox 等）与仿真 **Go1 / ANYmal-C / Hound**；大障碍 **0.6–1.5 m** 与 **41 cm 沙发** 实机。

## 对 wiki 的映射

- 沉淀实体页：[`wiki/entities/dreamwaq-plus.md`](../../wiki/entities/dreamwaq-plus.md)（交叉引用 Privileged Training、Terrain Adaptation、Locomotion、Sim2Real、State Estimation、Unitree 等见该实体页）
- 前序工作索引：`privileged_training.md` § DreamWaQ（ICRA 2023）
