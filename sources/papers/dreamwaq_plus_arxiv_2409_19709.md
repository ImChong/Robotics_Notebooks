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
- **入库日期：** 2026-05-30（2026-11-07 按 arXiv v2 全文通读复核）
- **版本：** arXiv:2409.19709v2（2026-02-26，T-RO 版），pdftotext 提取全文
- **核对范围：** 通读摘要、§I 引言、§II 相关工作、§III 方法（外感知记忆式 2–3、置信滤波式 4、本体编码器、多模态 Mixer、约束重参数化式 5、学习目标式 6–15）、§IV 实现细节（仿真、PD、域随机化表 I、噪声与对齐偏置表 II、特权状态、奖励课程表 III）、§V 实验（楼梯竞速、affordance、probing、OOD、跨平台、大障碍）、§VI 消融（表 IV、表 V、图 13–18）、§VII 结论；参考文献仅浏览。项目页视频与交互内容未重新逐条核对（项目页另见 `sources/sites/dreamwaqpp-github-io.md`）；未运行代码。
- **通读：** 已完成
- **一句话说明：** 在 **非对称 Actor–Critic + PPO** 框架下，用 **分层外感知记忆 + PointNet 置信滤波 + MLP-Mixer 多模态融合 + 对比/VAE 辅助损失**，单阶段训练出 **本体+外感知（3D 点云）** 四足控制器，在楼梯/陡坡/大障碍与传感器失效下保持鲁棒。

## 摘要级要点

- **问题：** 纯本体盲走需 **足端碰撞** 才能感知障碍；纯外感知又依赖 **精确时序地图** 且与控制环 **频率异步**（外感知约 10 Hz vs 本体/控制 50–200 Hz）。
- **主张：** **弹性多模态 RL**——在仿真中单阶段联合训练感知管线与控制策略；真机 **Go1（R1/R3/R4）与一台无外感知的 A1（R2，仅作盲走对照）** 上 **零微调** 部署；外感知输入为 **传感器无关的 3D 点云**（深度相机或 LiDAR）。
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

## 2026-11-07 通读复核补充（出自论文原文，v2）

- **平台：** R1 = Go1 + RealSense D435f（向下倾 45°、15 Hz 输出，Jetson Xavier NX，约 +0.5 kg 防护罩）；R2 = Unitree **A1**（无外感知，只跑盲走 DreamWaQ 作对照）；R3 = Go1 + NUC + Ouster OS-01（约 +3.0 kg）；R4 = Go1 + NUC + 双 Livox Mid-360（约 +2.5 kg）。仿真另训练 ANYmal-C 与 Hound。
- **频率与记忆：** 外感知取 10 Hz，故 $K=5$；本体 $H=5$ @ 50 Hz（窗口 100 ms）；本体 200 Hz 采样、外感知 10 Hz、控制器 50 Hz 综合；SE(3) 位姿由 IMU 姿态 + 估计体速欧拉积分得到，每 $K$ 步随新外感知重置，因此漂移不显著。
- **损失超参：** $eta=5.0$（β-VAE），自适应 β 调度式 (10)(11)；对比损失用编码后的真值高度扫描作正锚点、$U[-1,1]$ 随机潜变量作负锚点；versatility gain 取 $\lambda_e=0.1$（大障碍实验提到 0.2，并把速度跟踪奖励从 1.0 降到 0.1）；约束重参数化 $\sigma_{min}=0,\ \sigma_{max}=5$；MLP / Mixer 两隐层各 256 单元、ELU。
- **低层控制：** 策略 50 Hz 输出目标关节角，200 Hz PD，$K_p=25,\ K_d=0.7$；Pybind 接 Unitree SDK。
- **训练：** Isaac Gym Preview 3 + Legged Gym，3,500 个并行体，A5000 上约 11 小时。
- **域随机化（表 I）：** 负载 [−1,2] kg、Kp/Kd/电机强度因子 [0.9,1.1]、质心偏移 ±50 mm、摩擦 [0.2,1.25]、系统延迟 0–15 ms；外感知噪声分低 / 中 / 高三档（0–0.03 m / 0.03–0.1 m / 0.1–0.3 m），训练占比 30% / 50% / 20%；另有外感知安装位姿偏置（roll ±0.2 rad 等，表 II）。另用 Roll-Drop。
- **奖励：** 沿用 DreamWaQ 全部奖励；对力矩 / 关节速度 / 加速度 / 动作变化率 / 平滑项做指数退火 $w_{i+1}=0.998\,w_i$（表 III）。
- **楼梯竞速：** 50 级台阶，R1（DreamWaQ++）35 s 完成约 30.03 m 水平、7.38 m 爬升，指令速度约 1.0 m/s；R2（DreamWaQ，指令 1.2 m/s）同一时刻 20.05 m / 5.44 m；R3（内置感知控制器，起点前移一级）仅 6.38 m / 2.44 m 且失败。
- **表 IV 消融（仿真 1000 机，成功率 Stairs-easy / Stairs-hard / Discrete）：** 完整 99.5 / 97.8 / 99.8；无记忆 98.7 / 85.1 / 99.6；隐式记忆（LSTM）99.1 / 92.5 / 99.5；无潜变量融合 90.2 / 60.7 / 94.3；无对比损失 98.4 / 93.1 / 99.2；无 versatility gain 98.2 / 89.4 / 99.3。小腿碰撞数完整版最少（10.5 / 18.3 / 5.8）。
- **与 ViNL 式基线（图 7）：** 成功率比带真值高度图的 ViL-teacher 基线高 20–40%（作者归因于 versatility gain）。
- **表 V：** MLP-Mixer 体速估计误差 0.0363 / 0.0331 / 0.0385 m/s（vx/vy/vz），重建误差 0.0674 m，优于 MLP（0.1978）与 RNN（0.1253）；预测未来关节位置上与 Transformer 相当但更轻量（图 18）。
- **OOD：** 训练坡度仅 10°，实测 35° 坡后足力矩约为 DreamWaQ 的 1/1.5；小车被踢走时支撑多边形扩大约 20.12%；传感器给白噪声时出现「卡脚反射」，相机脱落后用脚和膝触地保持稳定（图 16）。
- **大障碍：** Go1 / ANYmal-C / Hound 分别训练到 0.6 / 1.0 / 1.5 m（图 12 题注；正文另有 ANYmal-C「可达 1.5 m」的表述，与题注不一致）；实机 Go1 带 2.5 kg 爬 41 cm 沙发。
- **论文自述未来方向：** 给相机装主动俯仰机构，联合学习运动与相机倾转；latent modulation 的关键维度会随训练种子变化（§VI-C 自述局限）；重建精度低于真值地图（§VI-F 三条原因）。
