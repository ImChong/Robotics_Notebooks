---
type: entity
tags: [hardware, humanoid, platform, unitree]
status: complete
updated: 2026-09-19
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
  - ./paper-pgmt.md
  - ./paper-tango-vla.md
  - ./paper-vibe.md
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
summary: "Unitree G1 是宇树推出的量产型高性价比人形机器人：可折叠、全关节力控、自带 3D LiDAR 与深度相机；本页按行走地形、起身安全、全身操作、遥操作采数、动作生成、导航 VLA 六条主线归纳其上的研究工作与上手入口。"
---

# Unitree G1 (人形机器人)

## 一句话定义

**Unitree G1** 是宇树科技在 H1 之后推出的量产型高性价比人形机器人：小型、可折叠、全关节力控、自带 3D LiDAR 与深度相机，配合官方 RL / 遥操作开源栈，已成为学术界人形运动与全身操作研究**事实上的公共验证平台**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| G1 | Unitree G1 Humanoid | 宇树入门级教育科研人形平台 |
| DoF | Degree of Freedom | 自由度；G1 论文常见 23 / 29 两种控制配置 |
| WBC | Whole-Body Control | 全关节力控，适配全身协调控制 |
| RL | Reinforcement Learning | 常见在 Isaac Lab / legged_gym 训练 |
| LiDAR | Light Detection and Ranging | 机载 3D 激光，支撑地形感知 |
| SDK2 | Unitree SDK v2 | 官方 C++/Python 开发包（经 CycloneDDS） |
| SR | Success Rate | 本页表格中的真机/仿真成功率 |

## 为什么重要

- **门槛低到可以「按台买」**：体型小于 H1、可折叠、单人搬运，使高校与中小实验室能批量部署，进而做机器人机房式的规模化真机数据采集（配合 [自动化标注](../methods/auto-labeling-pipelines.md) 与 [CLAW](../methods/claw.md) 这类合成管线）。
- **成了横向对照的公共基准**：本页收录的数十项工作多数在同一本体上做真机验证（少数仅到仿真，表内已标注），读论文时可以直接比「同一台 G1 上谁跑得更稳」。
- **软硬件生态齐全**：官方同时提供 RL 训练、XR 遥操作、模仿学习与 ROS 2 桥，第三方还有仿真、浏览器编排与成品技能商店，从「跑通一个策略」到「装一个舞蹈包」都有现成入口。

## 平台速览

| 维度 | 要点 |
|------|------|
| 定位 | 入门级教育 / 科研人形；小型、可折叠、单人可搬运部署 |
| 感知 | 机载 3D LiDAR + 深度相机，原生支撑 [地形自适应](../concepts/terrain-adaptation.md) |
| 控制 | 全关节高带宽力控，适配 [WBC](../concepts/whole-body-control.md) 与 RL 策略直出 |
| 常见控制配置 | 论文里多为 **23 DoF**（不含腕）或 **29 DoF**（含腰/腕）；重定向类页面按全身 43 DoF（无手指）计 |
| 机械特殊点 | 踝部为并联闭链，IK/FK 需专门解算，见 [人形并联关节解算](../concepts/humanoid-parallel-joint-kinematics.md) |
| 软件入口 | [G1 软件服务栈](./unitree-g1-software-stack.md)（SDK2 / CycloneDDS）、[unitree_ros2 v0.3.0](./unitree-ros2.md) |
| 仿真生态 | [Isaac Lab](./nvidia-omniverse.md)、[robot_lab](./robot-lab.md)、`legged_gym`、MuJoCo |
| 成品技能 | [UniStore](./unitree-unistore.md) 云端动作包（2026-05-07 全面开放） |

> **读规格时注意**：本页不复制官方参数表，整机尺寸、重量、电池与价格以宇树官方规格书为准；仓库内不同页面对 DoF 的口径（23 / 29 / 43）取决于该工作控不控腰腕与手指，**跨论文比较前先对齐配置**。

## 用 G1 做研究：六条主线

下面把仓库里所有 G1 相关工作按**读者想解决的问题**归类，而不是按收录时间罗列。每行括注为 arXiv 编号与开源状态。

### 1. 行走、地形与跑酷

问「G1 能走过什么」的读者从这里开始。

| 工作 | 在 G1 上做到什么 | 开源 |
|------|------------------|------|
| [PHP 感知跑酷](./paper-hrl-stack-22-perceptive_humanoid_parkour.md) | 仅机载深度 + 2D 速度指令，1.25 m 攀墙与长程多障碍（RSS 2026，2602.15827） | — |
| [ETH 扩散 + RL 全身](./paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md) | 地形条件扩散参考生成 + RL 跟踪，全 onboard；75 cm 箱攀、跨栏、楼梯（2604.17335） | — |
| [显式楼梯几何条件化](./paper-explicit-stair-geometry-humanoid-locomotion.md) | BEV 点云 → 楼梯几何 token → PPO，户外连续 33 级上楼零样本（2605.09944） | — |
| [LadderMan](./paper-ladderman-humanoid-perceptive-ladder-climbing.md) | 深度 + VFM 零样本爬多样梯子，梯顶 VR 双智能体作业（2606.05873） | — |
| [HumoSlope](./paper-humoslope-physics-guided-slope-locomotion.md) | 盲穿户外草地坡至 32.1°（局部平面 ZMP + BSGA，2607.07830） | 未开源 |
| [P³](./paper-p3.md) | VAE-PPO 边缘似然；踏石 / 楼梯 / 缺口真机（2607.25541） | **已开源** |
| [WM-LOCO](./paper-wm-loco.md) | 单深度 RSSM + PPO；沟 / 踏石 / 楼梯机载平均 93.3%（2609.02542） | 待发布 |
| [VB-Com](./paper-notebook-vb-com-learning-vision-blind-composite-humanoid.md) | 视觉 / 盲策略复合，感知缺失下过缺口与动态障碍（ICRA 2026） | coming soon |
| [PGMT](./paper-pgmt.md) | 感知通用动作跟踪，零样本跨 37 cm 障碍（2609.08511） | 未开源 |
| [SD-AMP](./paper-unified-walk-run-recovery-sdamp.md) | 三条 LAFAN1 参考 + 双 AMP 判别器，recovery→walk→run 无部署模式切换（2605.18611） | — |
| [RuN](./paper-notebook-run-residual-policy-for-natural-humanoid-locomot.md) | CMG 运动先验 + 轻量残差，0–2.5 m/s 自然走跑切换（2509.20696） | — |
| [ADP](./paper-adp.md) | 动力学对抗先验抗扰 locomotion（2607.03454） | 待发布 |
| [QuietWalk](./paper-quietwalk-humanoid-locomotion.md) | PINN 估竖直 GRF 作冲击惩罚；1.2 m/s 降噪 7.17 dB，跨鞋型泛化（2604.23702） | — |

### 2. 起身、抗扰与安全停

摔倒后能不能自己起来、急停时会不会砸到人，是从演示走向可用的分水岭。

| 工作 | 在 G1 上做到什么 | 开源 |
|------|------------------|------|
| [HoST](./paper-host-humanoid-standingup.md) | 从零 RL 学跨地面 / 平台 / 墙 / 坡、俯仰卧、室内外多姿态起身（2502.08378，RSS 2025 系统论文 finalist） | **已开源**（[InternRobotics/HoST](https://github.com/InternRobotics/HoST)） |
| [FDDC](./paper-fddc.md) | 可部署动态 CoM 单腿平衡；ONNX 50 Hz 无蒸馏直接上真机（2608.00500） | — |
| [Safe-Stop](./paper-safe-stop-humanoid.md) | 可停止性双估计急停，OOD 场景停止率 96.4%（2609.02358） | 待发布 |
| [PAC-MAN](./paper-pac-man-perceptive-cbf-rl.md) | CBF-RL，机载深度零样本躲避球 19/20（2607.28623） | — |
| [Switch](../methods/switch-framework.md) | 100% 跨技能切换成功率与强抗扰动 | — |
| [Fail-Passive Gap](./paper-fail-passive-gap.md) | 西门子在 G1 EDU 抓放单元上定位工业功能安全缺口：切电对行走双足本身是危害（2608.02809） | — |

### 3. 全身操作（loco-manipulation）

边走边干活——把腿和手当成一个系统来控。

| 工作 | 在 G1 上做到什么 | 开源 |
|------|------------------|------|
| [PILOT](./paper-pilot-perceptive-loco-manipulation.md) | LiDAR 高程图 + MoE 单阶段全身策略，楼梯 / 高台等非结构化场景边走边操作（2601.17440） | — |
| [ResMimic](./paper-resmimic.md) | GMT + 残差全身 loco-manipulation，4.5–5.5 kg 载荷（2510.05070） | — |
| [SteadyTray](./paper-notebook-steadytray.md) | 托盘 ReST-RL 残差平衡；96.9% 变速跟踪 / 74.5% 抗扰零样本 sim-to-real（2603.10306） | **已开源** |
| [Blind Dexterity](./paper-blind-dexterity.md) | 纯本体感知全身操作：足球 / 滑板 / 手提箱与无 IMU 推抗行走（2608.29487） | 待发布 |
| [SMPC-to-RL](./paper-smpc2rl-loco-manipulation.md) | 稀疏奖励全身推箱；SMPC 专家 + FastTD3（2608.12063） | 未开源 |
| [HTD 解耦 WBC](./htd-decoupled-wbc.md) | 开源下肢 + 腰 RL 控制器，G1 零样本部署 | **已开源** |
| [CLIFT](./paper-clift-closed-loop-iterative-finetuning.md) | 接触丰富双臂任务闭环迭代微调（装箱 / 插杯 / 双臂交接） | — |
| [ParcelStow](./paper-parcelstow.md) | Isaac Lab 上 L6 灵巧手模仿策略的时间鲁棒性评测（2609.01453） | **已开源** |
| [AGILE](./paper-agile-humanoid-loco-manipulation.md) | NVIDIA Isaac Lab 人形 RL 工作流：速度 / 高度 / 起身 / 舞蹈 / pick&place（2603.20147） | — |

### 4. 遥操作与数据采集

G1 的另一半价值在于「采数据的工具」，不只是「跑策略的靶子」。

| 工作 | 在 G1 上做到什么 |
|------|------------------|
| [Teleopit](./paper-teleopit.md) | PICO VR 全身 + 连续灵巧手 + 主动视觉遥操作（29 DoF，2608.01834） |
| [BifrostUMI](./paper-bifrost-umi.md) | Pico + 双腕夹爪无机器人示范，经扩散策略与 SKR 做杂乱桌面 / 桌下全身操作 |
| [CLAW](../methods/claw.md) | 网页交互快速生成带语言标签的全身动作数据 |
| [LEGS](./paper-legs-embodied-gaussian-splatting-vla.md) | 3DGS 合成 loco-manipulation VLA 数据（2606.01458） |
| [REK](./rek.md) | VR 遥操作 G1 全接触擂台赛与 U2 活动 / 教育租赁——娱乐产品化出口，与科研 teleop 采集形成对照 |

官方遥操作入口为 `xr_teleoperate`，模仿学习入口为 `unitree_lerobot`，见下文「上手路径」。

### 5. 动作生成、重定向与运动先验

「从哪来的参考动作」这条线上，G1 既是重定向目标，也是验证终点。

| 工作 | 在 G1 上做到什么 | 开源 |
|------|------------------|------|
| [ExoActor](../methods/exoactor.md) | 第三人称视频生成 + 通用动作跟踪，零真实数据的交互行为生成（BAAI, 2026） | — |
| [RoboReact](./paper-roboreact.md) | egocentric 生成视频蒸馏物体中心关键帧技能；长程双臂均值 SR 81.3%（2608.03387） | 未开源 |
| [ZEST](./paper-zest.md) | RAI × BD：视频爬箱 / 芭蕾与 MoCap 侧手翻、乒乓球零样本（Science Robotics 2026） | 未开源 |
| [ADAPT](./paper-adapt-text-driven-humanoid.md) | ETH 端到端扩散先验，50 Hz 在线换 prompt 的文本驱动控制（2609.00677） | 未开源 |
| [UMR](./paper-umr-unified-motion-retargeting.md) | 学习点云对应做重定向；跟踪 / 接触 / 真机（2609.02134） | 待发布 |
| [X-Morph](./paper-xmorph.md) | 人体运动先落到 G1 表示，再跨形态到 Go2 / 六足 / B2-Z1（2606.30290） | — |
| [MotionBricks](../methods/motionbricks.md) | NVIDIA 生成式运动框架的首批验证硬件 | — |
| [SMP](../methods/smp.md) | 基于得分匹配的运动先验，已在 G1 完成验证 | — |
| [CMP](./paper-cmp.md) | 仿真 G1 上上下文感知 AMP 适配；Dribbling 294→467（2608.03234） | 未开源 |
| [NCKU 合成视频人形任务](./paper-synthetic-video-humanoid-tasks.md) | 生成视频 → GMR → 仿真 RL 跟踪（**无真机结果**，2607.21648） | — |
| [PAiD](../methods/paid-framework.md) | 足球技能学习主力平台，验证类人化踢球动作（另见 [Humanoid Soccer](../tasks/humanoid-soccer.md)） | — |

### 6. 导航、语言与 VLA

| 工作 | 在 G1 上做到什么 | 开源 |
|------|------------------|------|
| [HumanoidVLN](./paper-humanoidvln.md) | Isaac 人形 VLN 基准 + G1 DualVLN 20 条 sim–real 试点（2608.12860） | 待开源 |
| [Uni-LaViRA](./paper-uni-lavira.md) | G1 真机零样本 VLN / ObjectNav / EQA 部署之一（2605.27582） | — |
| [TANGO](./paper-tango-vla.md) | 全身 VLA 杂乱室内导航，G1 零样本（CoRL 2026，2609.09158） | 未开源 |
| [POT-VLA](./paper-pot-vla.md) | 持久 3D 对象 token 实现可验证的闭环移动操作 | — |
| [ViBe](./paper-vibe.md) | 预训练视觉编码器 + LoRA 的感知全身后训练；路缘 / 跑酷 / 物体操作 / 躲避球零样本 sim2real（2609.09918） | 未开源 |
| [Grove-G1](./grove-g1.md) | ROS 2 Humble 自主栈：Nav2 + MoveIt + BehaviorTree 端到端 pick-place | **已开源** |
| [DimOS](./dimensionalos-dimos.md) | `dimos --simulation run unitree-g1-sim`（MuJoCo），Python Blueprint + MCP 编排，**无需 ROS 起步**；G1 支持为 beta | **已开源** |

## 上手路径

| 你想做什么 | 从哪进 |
|------------|--------|
| 先看真机能跑成什么样 | [RL Sim2Sim 在线演示：G1 AMP Walk/Run/Getup](https://imchong.github.io/RL_Sim2Sim_Demo_Website/index.html) |
| 读写关节 / 订阅传感器 | [G1 软件服务栈](./unitree-g1-software-stack.md)（SDK2 + CycloneDDS） |
| 用 ROS 2 接真机 | [`unitree_ros2` v0.3.0](./unitree-ros2.md)（双臂 / Dex3 / Arm SDK）；ROS1 + Gazebo 路线见 [unitree_ros](./unitree-ros.md) |
| 训 RL 策略 | 官方 `unitree_rl_gym` / `unitree_rl_lab` / `unitree_rl_mjlab`；扩展框架 [robot_lab](./robot-lab.md)、[Isaac Lab](./nvidia-omniverse.md)（组织地图见 [sources/repos/unitree.md](../../sources/repos/unitree.md)） |
| 做遥操作 / 模仿学习 | 官方 `xr_teleoperate`、`unitree_lerobot` 与 UnifoLM VLA/WMA，见 [Unitree 品牌主页](./unitree.md) |
| 只想装成品动作 | [UniStore](./unitree-unistore.md)：Explore App ≥ 1.9.0 + OTA ≥ 1.4.8 一键安装舞蹈 / 武术包 |
| 在浏览器里比对 obs / 推理同步 | [BotLab / MotionCanvas](./botlab-motioncanvas.md)：ONNX + MuJoCo 节点图实验台 |
| 系统性地学一遍 | [人形系统课程策展](./humanoid-system-curriculum.md)（深蓝学院八章地图）、[Humanoid Control Roadmap](../roadmaps/humanoid-control-roadmap.md) |

## 局限与风险

- **规格口径要自己对齐**：23 / 29 / 43 DoF 分别对应控不控腕、腰与手指，不同配置的成功率不可直接横比（例：[HUSKY](./paper-amp-survey-14-husky.md) 只控 23 DoF，与 29 DoF 设定的工作不同口径）；整机参数以官方规格书为准。
- **大量工作未开源**：上表中标「未开源 / 待发布」的占比不低，复现前先确认代码与权重是否真的可得。
- **工业安全尚未闭环**：[Fail-Passive Gap](./paper-fail-passive-gap.md) 指出「切电即安全」对行走双足不成立，机侧平衡站住目前评不出性能等级（PL），**不能按工业机械臂的保护停思路做安全设计**。
- **负载量级偏小**：仓库内 G1 全身操作工作报告的载荷多在个位数公斤级（如 [ResMimic](./paper-resmimic.md) 的 4.5–5.5 kg）；把这个量级当作规划上界是**基于现有论文的归纳，不是官方指标**，实际承载以官方规格书为准。
- **踝部并联闭链**：直接套串联模型的 IK/FK 会出错，见 [人形并联关节解算](../concepts/humanoid-parallel-joint-kinematics.md)。

## 关联页面

- [人形机器人 (Humanoid Robot)](./humanoid-robot.md) — 上位概念页
- [Unitree 品牌主页](./unitree.md) — 整机与开源组织总览
- [G1 软件服务栈](./unitree-g1-software-stack.md) — SDK2/DDS 与仿真桥接口
- [unitree_ros2](./unitree-ros2.md) — v0.3.0 G1 双臂 / Dex3 / Arm SDK 官方 ROS 2 入口
- [UniStore（宇树应用平台）](./unitree-unistore.md) — 云端成品技能分发
- [Humanoid Control Roadmap](../roadmaps/humanoid-control-roadmap.md) — 人形控制学习路线
- [人形系统课程策展](./humanoid-system-curriculum.md) — 深蓝学院 G1 系统课八章地图
- [Whole-Body Control](../concepts/whole-body-control.md) — G1 力控能力对应的控制范式
- [地形自适应](../concepts/terrain-adaptation.md) — 感知 locomotion 的概念底座
- [人形机器人并联关节解算](../concepts/humanoid-parallel-joint-kinematics.md) — G1 踝部闭链 IK/FK/雅可比参考实现
- [Residual Policy Learning](../methods/residual-policy-learning.md) — base + 残差谱系（含 G1 上 RuN / ResMimic 定位）
- [Humanoid Soccer](../tasks/humanoid-soccer.md) — G1 足球任务线
- [自动化标注管线](../methods/auto-labeling-pipelines.md) — 规模化真机数据的下游处理
- [robot_lab](./robot-lab.md) / [Isaac Lab](./nvidia-omniverse.md) — 常用训练框架
- [REK](./rek.md) — G1 VR 格斗联赛与机器人租赁品牌
- [DimOS（Dimensional）](./dimensionalos-dimos.md) — G1 MuJoCo 仿真与 beta 级 agent 集成栈
- [BotLab / MotionCanvas](./botlab-motioncanvas.md) — 浏览器内策略与 MuJoCo 可视化编排

## 参考来源

- Unitree G1 官方规格书（整机参数以官方发布为准）。
- [RL Sim2Sim 在线演示：G1 AMP Walk/Run/Getup](https://imchong.github.io/RL_Sim2Sim_Demo_Website/index.html)
- [sources/repos/unitree.md](../../sources/repos/unitree.md) — 官方开源组织地图（RL / 遥操作 / IL / VLA 入口）。
- [sources/repos/unitree_ros2.md](../../sources/repos/unitree_ros2.md) — v0.3.0 G1 双臂 / Dex3 / Arm SDK ROS 2 入口。
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
- [sources/papers/adp_arxiv_2607_03454.md](../../sources/papers/adp_arxiv_2607_03454.md) — ADP：G1 动力学对抗先验抗扰 locomotion（arXiv:2607.03454）。
- [sources/papers/humoslope_arxiv_2607_07830.md](../../sources/papers/humoslope_arxiv_2607_07830.md) — HumoSlope：G1 户外草地坡 locomotion（arXiv:2607.07830）。
- [sources/papers/pac_man_perceptive_cbf_rl_arxiv_2607_28623.md](../../sources/papers/pac_man_perceptive_cbf_rl_arxiv_2607_28623.md) — PAC-MAN：G1 机载深度零样本躲避球（arXiv:2607.28623）。
- [sources/papers/fddc_arxiv_2608_00500.md](../../sources/papers/fddc_arxiv_2608_00500.md) — FDDC：G1 动态 CoM 单腿平衡（arXiv:2608.00500）。
- [sources/papers/zest.md](../../sources/papers/zest.md) — ZEST：视频与 MoCap 驱动的 G1 零样本技能。
- [sources/papers/humanoidvln_arxiv_2608_12860.md](../../sources/papers/humanoidvln_arxiv_2608_12860.md) — HumanoidVLN：人形 VLN 基准与 G1 sim–real 试点（arXiv:2608.12860）。
- [sources/papers/fail_passive_gap_arxiv_2608_02809.md](../../sources/papers/fail_passive_gap_arxiv_2608_02809.md) — 西门子：G1 EDU 功能安全 fail-passive gap（arXiv:2608.02809）。
- [sources/courses/shenlan_humanoid_system_theory_practice.md](../../sources/courses/shenlan_humanoid_system_theory_practice.md) — 深蓝学院人形系统课程。
- [sources/sites/rek-com.md](../../sources/sites/rek-com.md) — REK 官网：G1 VR 格斗联赛与租赁。
- [sources/papers/humanoid_hardware.md](../../sources/papers/humanoid_hardware.md) — 人形硬件横向资料。

## 推荐继续阅读

- [Unitree 官方 GitHub 组织](https://github.com/unitreerobotics) — `unitree_rl_gym` / `unitree_rl_lab` / `xr_teleoperate` / `unitree_lerobot` 等一手仓库。
- [InternRobotics/HoST](https://github.com/InternRobotics/HoST) — G1 多姿态起身的可复现开源实现，适合作为第一个真机 RL 项目。
- [Adyansh04/grove-g1](https://github.com/Adyansh04/grove-g1) — ROS 2 Humble 上 Nav2 + MoveIt + BehaviorTree 的 G1 端到端自主栈。
- [Parallel_Ankle_Joint](https://github.com/feidedao/Parallel_Ankle_Joint) — G1 踝部并联闭链 IK/FK/雅可比参考实现。
