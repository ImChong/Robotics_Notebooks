---
type: entity
tags: [hardware, humanoid, platform, unitree]
status: complete
updated: 2026-09-04
related:
  - ./humanoid-robot.md
  - ./rek.md
  - ./unitree.md
  - ./unitree-unistore.md
  - ./paper-synthetic-video-humanoid-tasks.md
  - ./unitree-ros.md
  - ./unitree-ros2.md
  - ./unitree-g1-software-stack.md
  - ./grove-g1.md
  - ./humanoid-system-curriculum.md
  - ./botlab-motioncanvas.md
  - ./paper-adp.md
  - ./paper-humoslope-physics-guided-slope-locomotion.md
  - ./paper-uni-lavira.md
  - ./paper-pac-man-perceptive-cbf-rl.md
  - ./paper-fddc.md
  - ./paper-agile-humanoid-loco-manipulation.md
  - ./htd-decoupled-wbc.md
  - ./paper-p3.md
  - ./paper-wm-loco.md
  - ./paper-safe-stop-humanoid.md
  - ./paper-smpc2rl-loco-manipulation.md
  - ./paper-roboreact.md
  - ./paper-zest.md
  - ./paper-humanoidvln.md
  - ./paper-fail-passive-gap.md
  - ./paper-notebook-vb-com-learning-vision-blind-composite-humanoid.md
  - ../roadmaps/humanoid-control-roadmap.md
  - ./paper-umr-unified-motion-retargeting.md
sources:
  - ../../sources/papers/humanoid_hardware.md
  - ../../sources/papers/adp_arxiv_2607_03454.md
  - ../../sources/papers/humoslope_arxiv_2607_07830.md
  - ../../sources/papers/uni_lavira_arxiv_2605_27582.md
  - ../../sources/papers/pac_man_perceptive_cbf_rl_arxiv_2607_28623.md
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
  - ../../sources/papers/fddc_arxiv_2608_00500.md
  - ../../sources/papers/roboreact_arxiv_2608_03387.md
  - ../../sources/papers/zest.md
  - ../../sources/papers/humanoidvln_arxiv_2608_12860.md
  - ../../sources/papers/fail_passive_gap_arxiv_2608_02809.md
  - ../../sources/papers/p3_arxiv_2607_25541.md
  - ../../sources/repos/unitree_ros2.md
summary: "Unitree G1 是一款由宇树科技推出的入门级教育科研用人形机器人，以其极高的性价比、高集成度以及对仿真学习框架的良好支持而备受关注。"
---

# Unitree G1 (人形机器人)

**Unitree G1** 是宇树科技 (Unitree) 在 H1 之后推出的一款量产型、高性价比的人形机器人平台。其设计初衷是降低人形机器人研究的门槛，使其能够大规模进入实验室、高校和家庭场景。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| G1 | Unitree G1 Humanoid | 宇树入门级教育科研人形平台 |
| WBC | Whole-Body Control | 全关节力控，适配全身协调控制 |
| RL | Reinforcement Learning | 常见在 Isaac Lab / legged_gym 训练 |
| LiDAR | Light Detection and Ranging | 机载 3D 激光，支撑地形感知 |
| SD-AMP | Selective Domain AMP | 走跑起身统一策略的代表工作线 |
| PILOT | Perceptive Loco-Manipulation | 感知移动操作 LLC 在 G1 上的验证 |

## 核心特性

1. **高集成度与便携性**：G1 的体型较 H1 更小，支持折叠收纳，单人即可搬运和部署。
2. **丰富的感知方案**：集成了 3D 激光雷达 (LiDAR) 和深度相机，原生支持 [地形自适应](../concepts/terrain-adaptation.md)。
3. **力控能力**：全关节支持高带宽力控，极其适配 [WBC](../concepts/whole-body-control.md) 与强化学习。
4. **生态支持**：完美适配 [Isaac Lab](../entities/nvidia-omniverse.md)、[robot_lab](../entities/robot-lab.md) 和 `legged_gym`；同时作为 NVIDIA [MotionBricks](../methods/motionbricks.md) 生成式运动框架的首批验证硬件。
5. **高效数据生成**：支持 [CLAW](../methods/claw.md) 等合成数据管线，通过网页交互快速生成带语言标签的全身动作数据。
6. **足球技能研究**：作为 [PAiD](../methods/paid-framework.md) 框架的主要实验平台，证明了其在执行类人化踢球动作方面的卓越物理特性。
7. **敏捷技能切换**：[Switch](../methods/switch-framework.md) 框架在 G1 上实现了 100% 的跨技能切换成功率与极强的抗扰动能力。
8. **视频生成驱动的零样本控制**：作为 [ExoActor](../methods/exoactor.md)（BAAI, 2026）的端到端验证平台，把第三人称视频生成 + 通用动作跟踪整合为零真实数据的交互行为生成系统；[RoboReact](./paper-roboreact.md)（arXiv:2608.03387）则用 **egocentric 生成视频** 蒸馏物体中心关键帧技能，G1 长程双臂均值 SR **81.3%**（未开源）。
9. **浏览器侧策略–仿真编排（生态周边）**：地瓜机器人 [BotLab / MotionCanvas](./botlab-motioncanvas.md) 在网页中提供面向 G1 / Go2 的 ONNX + MuJoCo 节点图实验台，便于对照训练侧 obs 堆叠语义与推理同步策略。
10. **无机器人全身示范部署**：[BifrostUMI](./paper-bifrost-umi.md)（BAAI Aether, 2026）在 G1 上验证 Pico + 双腕夹爪采集数据经扩散策略与 SKR 的杂乱桌面与桌下全身操作。
11. **统一走跑起身（SD-AMP）**：[SD-AMP](./paper-unified-walk-run-recovery-sdamp.md)（HKU, arXiv:2605.18611）在 G1 真机用三条 LAFAN1 参考 + 双 AMP 判别器实现 recovery→walk→run 无部署模式切换。
12. **感知跑酷（PHP）**：[Perceptive Humanoid Parkour](./paper-hrl-stack-22-perceptive_humanoid_parkour.md)（arXiv:2602.15827，RSS 2026）在 G1 上仅用机载深度与 2D 速度指令完成 1.25 m 攀墙与长程多障碍跑酷。
13. **显式楼梯几何爬梯**：[Explicit Stair Geometry Conditioning](./paper-explicit-stair-geometry-humanoid-locomotion.md)（arXiv:2605.09944）在 G1 上零样本部署 BEV 点云 → 楼梯几何 token → PPO，户外连续 33 级上楼。
14. **多姿态起身（HoST）**：[HoST](./paper-host-humanoid-standingup.md)（arXiv:2502.08378，RSS 2025 系统论文 finalist）在 G1 上从零 RL 学习跨地面/平台/墙/坡及俯仰卧、室内外场景的起身，官方 [InternRobotics/HoST](https://github.com/InternRobotics/HoST) 开源。
15. **感知 loco-manipulation LLC（PILOT）**：[PILOT](./paper-pilot-perceptive-loco-manipulation.md)（arXiv:2601.17440，上海交大）在 G1 上用 LiDAR 高程图 + MoE 单阶段全身策略完成楼梯/高台等非结构化 **边走边操作**（VR 遥操作 + 分层 RL）。
16. **扩散生成 + RL 全身感知移动**：[Learning Whole-Body Humanoid Locomotion](./paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md)（arXiv:2604.17335，ETH RSL）在 G1 上全 onboard 部署地形条件扩散参考生成与 RL 跟踪器，完成 75 cm 箱攀、跨栏、楼梯与混合地形穿越。
17. **感知梯子攀爬 + 梯上操作（LadderMan）**：[LadderMan](./paper-ladderman-humanoid-perceptive-ladder-climbing.md)（arXiv:2606.05873，Amazon FAR 等）在 G1 上用机载深度 + VFM 零样本攀爬多样梯子，梯顶 PICO VR 双智能体遥操作（调画、换灯泡、递箱）。
18. **格斗体育联赛（REK）**：[REK](./rek.md) 以 **VR 遥操作 G1** 进行全接触擂台赛，并提供 G1 U2 活动/教育租赁——消费级人形的娱乐产品化出口，与科研 teleop 数据采集形成对照。
19. **低噪行走（QuietWalk）**：[QuietWalk](./paper-quietwalk-humanoid-locomotion.md)（NIMTE / Westlake, arXiv:2604.23702）在 G1 上用 **PINN 估计竖直 GRF** 作 RL 冲击惩罚，真机 **1.2 m/s** 相对基线 RL 平均降噪 **7.17 dB**，验证 **赤脚 / 滑板鞋 / 运动鞋 / 高跟鞋** 跨鞋型泛化。
20. **官方技能商店（UniStore）**：[UniStore](./unitree-unistore.md) 在 **2026-05-07** 全面开放后，G1 用户可通过 **Unitree Explore App（≥ 1.9.0）** 与 **OTA（≥ 1.4.8）** 从云端一键安装舞蹈、武术等成品动作包，与自研 RL / 模仿学习管线形成「平台技能 vs 实验室策略」对照。
21. **DimOS agent 集成（beta）**：[DimOS（Dimensional）](./dimensionalos-dimos.md) 提供 `dimos --simulation run unitree-g1-sim`（MuJoCo）及 README 列 **beta** 级 G1 平台支持，用 Python Blueprint + MCP 做导航/感知/agent 编排，**无需 ROS 即可起步**。
22. **官方开源研发栈**：在 [Unitree / unitreerobotics](./unitree.md) 组织下，G1 常用入口包括 `unitree_rl_gym` / `unitree_rl_lab` / `unitree_rl_mjlab`（RL）、`xr_teleoperate`（XR 遥操作）、`unitree_lerobot`（IL）与 UnifoLM VLA/WMA；ROS 2 真机桥见 [`unitree_ros2` v0.3.0](./unitree-ros2.md)（双臂 / Dex3 / Arm SDK）。组织地图见 [sources/repos/unitree.md](../../sources/repos/unitree.md)。
23. **工业功能安全缺口（Fail-Passive Gap）**：[Fail-Passive Gap](./paper-fail-passive-gap.md)（西门子，arXiv:2608.02809）在 G1 EDU 半封闭抓放单元上用认证光幕/F-PLC/无线 PROFIsafe 定位：切电对行走双足是危害，机侧平衡站住目前评不了 PL。

## 在具身智能中的作用

G1 的出现极大地加速了大规模数据的采集。由于其成本低廉，研究者可以构建“机器人机房（Robot Farms）”，利用海量实体机器人通过 [自动化标注](../methods/auto-labeling-pipelines.md) 或利用 [CLAW](../methods/claw.md) 等仿真合成手段快速生成训练基础策略（Foundation Policies）所需的真实数据。

## 关联页面

- [G1 软件服务栈](./unitree-g1-software-stack.md) — SDK2/DDS 与仿真桥接口（课程 1.4）
- [unitree_ros2](./unitree-ros2.md) — v0.3.0 G1 双臂 / Dex3 / Arm SDK 官方 ROS 2 入口
- [人形系统课程策展](./humanoid-system-curriculum.md) — 深蓝学院 G1 系统课八章地图
- [AGILE（论文实体）](./paper-agile-humanoid-loco-manipulation.md) — NVIDIA Isaac Lab 人形 RL 工作流；G1 速度/高度/stand-up/舞蹈/pick&place（arXiv:2603.20147）
- [Blind Dexterity](./paper-blind-dexterity.md) — 纯本体 G1 全身操作：足球/滑板/手提箱与无 IMU 推抗行走（arXiv:2608.29487；代码待发布）
- [HTD 解耦 WBC](./htd-decoupled-wbc.md) — HTD 开源下肢+腰 RL 控制器；G1 零样本部署
- [P³](./paper-p3.md) — VAE-PPO 边缘似然；G1 踏石/楼梯/缺口真机（arXiv:2607.25541，已开源）
- [WM-LOCO](./paper-wm-loco.md) — 单深度 RSSM+PPO；沟/踏石/楼梯机载平均 93.3%（arXiv:2609.02542；代码待发布）
- [Safe-Stop](./paper-safe-stop-humanoid.md) — 可停止性双估计急停；OOD 停止 96.4%（arXiv:2609.02358；代码待发布）
- [VB-Com](./paper-notebook-vb-com-learning-vision-blind-composite-humanoid.md) — 视觉/盲策略复合，G1/H1 感知缺失下缺口与动态障碍（ICRA 2026；代码 coming soon）
- [SMPC-to-RL](./paper-smpc2rl-loco-manipulation.md) — 稀疏奖励全身推箱；SMPC 专家 + FastTD3（arXiv:2608.12063；截至 2026-08-17 未开源）
- [ZEST](./paper-zest.md) — RAI × BD：视频爬箱/芭蕾与 MoCap 侧手翻、乒乓球零样本（Science Robotics 2026；未开源）
- [RoboReact](./paper-roboreact.md) — 生成 egocentric 视频蒸馏全身操作；G1 + HOMIE，四任务均值 SR 81.3%（arXiv:2608.03387；未开源）
- [HumanoidVLN](./paper-humanoidvln.md) — Isaac 人形 VLN 基准与 G1 DualVLN 20 条 sim–real 试点（arXiv:2608.12860；待开源）
- [Teleopit](./paper-teleopit.md) — PICO VR 全身+连续灵巧手+主动视觉遥操作（G1 29 DoF；arXiv:2608.01834）
- [FDDC](./paper-fddc.md) — 可部署动态 CoM 单腿平衡；G1 ONNX 50 Hz 无蒸馏真机（arXiv:2608.00500）
- [POT-VLA](./paper-pot-vla.md) — 在 G1 上用持久 3D 对象 token 实现可验证的闭环移动操作
- [smp](../methods/smp.md) (基于得分匹配的运动先验，已在 G1 完成验证)
- [NCKU 合成视频人形任务](./paper-synthetic-video-humanoid-tasks.md) — 生成视频→GMR→仿真 RL 跟踪（无真机结果；arXiv:2607.21648）
- [X-Morph](./paper-xmorph.md) — 人体运动先落到 G1 表示，再跨形态到 Go2/六足/B2-Z1（arXiv:2606.30290）
- [人形机器人 (Humanoid Robot)](./humanoid-robot.md)
- [ParcelStow](./paper-parcelstow.md) — Isaac Lab 上 G1 L6 灵巧手模仿策略的时间鲁棒性评测（arXiv:2609.01453，已开源）
- [ADAPT（文本驱动）](./paper-adapt-text-driven-humanoid.md) — ETH 端到端扩散先验，G1 50 Hz 在线换 prompt（arXiv:2609.00677，未开源）
- [UMR](./paper-umr-unified-motion-retargeting.md) — 学习点云对应重定向；G1 跟踪/接触/真机（arXiv:2609.02134，待发布）
- [Unitree 品牌主页](./unitree.md)
- [UniStore（宇树应用平台）](./unitree-unistore.md)
- [robot_lab (IsaacLab 扩展框架)](./robot-lab.md)
- [PAiD Framework (足球技能学习)](../methods/paid-framework.md)
- [Humanoid Soccer (足球任务)](../tasks/humanoid-soccer.md)
- [CLAW (宇树 G1 全身动作数据生成管线)](../methods/claw.md)
- [CLIFT](./paper-clift-closed-loop-iterative-finetuning.md) — 在 G1 上做接触丰富双臂任务的闭环迭代微调（装箱 / 插杯 / 双臂交接）
- [LEGS（论文实体）](./paper-legs-embodied-gaussian-splatting-vla.md) — G1 上 3DGS 合成 loco-manip VLA 数据（arXiv:2606.01458）
- [Humanoid Control Roadmap](../roadmaps/humanoid-control-roadmap.md)
- [ExoActor](../methods/exoactor.md) — G1 上的视频生成驱动的零样本交互控制系统。
- [PAC-MAN](./paper-pac-man-perceptive-cbf-rl.md) — G1 机载深度零样本躲避球（CBF-RL，19/20）
- [BifrostUMI（论文实体）](./paper-bifrost-umi.md) — 无机器人示范 + SKR + WBC 的全身 visuomotor 管线。
- [BotLab / MotionCanvas](./botlab-motioncanvas.md) — 浏览器内 G1 相关策略与 MuJoCo 可视化编排入口。
- [显式楼梯几何条件化（论文实体）](./paper-explicit-stair-geometry-humanoid-locomotion.md) — G1 楼梯几何 token + PPO（arXiv:2605.09944）。
- [PILOT（论文实体）](./paper-pilot-perceptive-loco-manipulation.md) — G1 感知统一 loco-manipulation 低层控制器（arXiv:2601.17440）。
- [ResMimic（论文实体）](./paper-resmimic.md) — G1 上 GMT+残差全身 loco-manipulation，4.5–5.5 kg 载荷（arXiv:2510.05070）。
- [RuN（论文实体）](./paper-notebook-run-residual-policy-for-natural-humanoid-locomot.md) — G1 上 CMG 运动先验 + 轻量残差策略的 0–2.5 m/s 自然走跑切换（arXiv:2509.20696）。
- [Residual Policy Learning（方法页）](../methods/residual-policy-learning.md) — base + 残差谱系总览（含 G1 上 RuN/ResMimic 定位）。
- [Whole-Body Locomotion（论文实体）](./paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md) — G1 扩散运动生成 + RL 全身跟踪（arXiv:2604.17335）。
- [REK](./rek.md) — G1 VR 格斗联赛与机器人租赁品牌。
- [QuietWalk（论文实体）](./paper-quietwalk-humanoid-locomotion.md) — G1 PINN-GRF 低噪行走（arXiv:2604.23702）。
- [ADP（论文实体）](./paper-adp.md) — G1 上动力学对抗先验抗扰 locomotion（arXiv:2607.03454；代码待发布）。
- [CMP（论文实体）](./paper-cmp.md) — 模拟 G1 上上下文感知 AMP 适配（arXiv:2608.03234；Dribbling 294→467、达阈 10.1→2.9×10⁸；代码未开源）。
- [HumoSlope（论文实体）](./paper-humoslope-physics-guided-slope-locomotion.md) — G1 盲穿户外草地坡至 32.1°（局部平面 ZMP + BSGA；arXiv:2607.07830；代码未开源）。
- [Uni-LaViRA（论文实体）](./paper-uni-lavira.md) — G1 真机零样本 VLN/ObjectNav/EQA 部署之一（arXiv:2605.27582）。
- [DimOS（Dimensional）](./dimensionalos-dimos.md) — G1 MuJoCo 仿真与 beta 级 agent/导航集成栈。
- [Grove-G1](./grove-g1.md) — ROS 2 Humble 自主栈：Nav2 + MoveIt + BehaviorTree 端到端 pick-place（[Adyansh04/grove-g1](https://github.com/Adyansh04/grove-g1)）。
- [人形机器人并联关节解算](../concepts/humanoid-parallel-joint-kinematics.md) — G1 踝部闭链 IK/FK/雅可比参考实现（[Parallel_Ankle_Joint](https://github.com/feidedao/Parallel_Ankle_Joint)）。
- [Fail-Passive Gap](./paper-fail-passive-gap.md) — G1 EDU 工业保护停可行性：外部链可评、机侧反应链不可评（arXiv:2608.02809）。

## 参考来源

- [RL Sim2Sim 在线演示：G1 AMP Walk/Run/Getup](https://imchong.github.io/RL_Sim2Sim_Demo_Website/index.html)
- Unitree G1 官方规格书。
- [sources/papers/exoactor.md](../../sources/papers/exoactor.md) — ExoActor 在 G1 上的端到端实现。
- [sources/papers/roboreact_arxiv_2608_03387.md](../../sources/papers/roboreact_arxiv_2608_03387.md) — RoboReact：G1 生成视频蒸馏全身操作（arXiv:2608.03387）。
- [sources/papers/bfm_humanoid_arxiv_2509_13780.md](../../sources/papers/bfm_humanoid_arxiv_2509_13780.md) — BFM：在 G1 上做 motion tracking / VR 遥操作 / locomotion 的统一条件生成策略。
- [sources/papers/bifrost_umi_arxiv_2605_03452.md](../../sources/papers/bifrost_umi_arxiv_2605_03452.md) — BifrostUMI：G1 真机全身 loco-manipulation（arXiv:2605.03452）。
- [sources/papers/php_parkour_arxiv_2602_15827.md](../../sources/papers/php_parkour_arxiv_2602_15827.md) — PHP：G1 感知跑酷（arXiv:2602.15827）。
- [sources/papers/p3_arxiv_2607_25541.md](../../sources/papers/p3_arxiv_2607_25541.md) — P³：G1 VAE-PPO 边缘似然与踏石/楼梯/缺口真机（arXiv:2607.25541）。
- [sources/papers/pilot_arxiv_2601_17440.md](../../sources/papers/pilot_arxiv_2601_17440.md) — PILOT：G1 感知 loco-manipulation LLC（arXiv:2601.17440）。
- [sources/papers/resmimic_arxiv_2510_05070.md](../../sources/papers/resmimic_arxiv_2510_05070.md) — ResMimic：G1 GMT→残差 loco-manipulation（arXiv:2510.05070）。
- [sources/papers/eth-g1-diffusion.md](../../sources/papers/eth-g1-diffusion.md) — ETH RSL：G1 扩散运动生成 + RL 全身感知 locomotion（arXiv:2604.17335）。
- [sources/papers/uni_lavira_arxiv_2605_27582.md](../../sources/papers/uni_lavira_arxiv_2605_27582.md) — Uni-LaViRA：G1 等四本体零样本统一导航（arXiv:2605.27582）。
- [sources/papers/fail_passive_gap_arxiv_2608_02809.md](../../sources/papers/fail_passive_gap_arxiv_2608_02809.md) — 西门子：G1 EDU 功能安全 fail-passive gap（arXiv:2608.02809）。
- [sources/sites/rek-com.md](../../sources/sites/rek-com.md) — REK 官网：G1 VR 格斗联赛与租赁。
- [sources/repos/unitree_ros2.md](../../sources/repos/unitree_ros2.md) — v0.3.0 G1 双臂 / Dex3 / Arm SDK ROS 2 入口。
