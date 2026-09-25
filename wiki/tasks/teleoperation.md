---
type: task
tags: [teleoperation, manipulation, loco-manipulation, data-collection, humanoid]
status: complete
summary: "Teleoperation 让人类通过远程接口直接操作机器人，是数据采集和复杂任务执行的重要桥梁。"
updated: 2026-09-20
sources:
  - ../../sources/blogs/wechat_jushen_qianyan_embodied_data_collection_taxonomy_2026-09-05.md
  - ../../sources/papers/ego_oscar_arxiv_2608_08285.md
  - ../../sources/papers/omega0_arxiv_2608_06375.md
  - ../../sources/papers/immersive_social_vr_llm_humanoids_arxiv_2607_07430.md
  - ../../sources/papers/teledexter_arxiv_2607_11481.md
  - ../../sources/sites/teledexter-project.md
  - ../../sources/blogs/mimicrobotics_m1_u1_full_stack.md
  - ../../sources/sites/rek-com.md
  - ../../sources/sites/engineai-urkl.md
  - ../../sources/sites/urkl-org.md
  - ../../sources/blogs/wechat_urkl_faq_01.md
  - ../../sources/sites/hiw-500-dataset.md
  - ../../sources/papers/ume_exo_arxiv_2606_14218.md
  - ../../sources/papers/rove_arxiv_2606_17011.md
  - ../../sources/blogs/current_robotics_currentworld.md
  - ../../sources/papers/pilot_arxiv_2601_17440.md
  - ../../sources/papers/halomi_arxiv_2606_18772.md
  - ../../sources/papers/hapmorph_arxiv_2509_05433.md
  - ../../sources/papers/teleoperation.md
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/sites/xyzcorp-deux.md
  - ../../sources/sites/x2robot-twindex.md
  - ../../sources/papers/humanoid_touch_dream.md
  - ../../sources/repos/robot-io-rio.md
  - ../../sources/repos/xpad.md
  - ../../sources/repos/microsoft-ui-xaml.md
  - ../../sources/papers/cwi_arxiv_2606_27676.md
  - ../../sources/sites/telegate-project.md
  - ../../sources/papers/telegate_arxiv_2602_09628.md
  - ../../sources/sites/heft-project.md
  - ../../sources/papers/heft_arxiv_2607_02332.md
  - ../../sources/repos/axellwppr_motion_tracking.md
  - ../../sources/sites/teleopit-project.md
  - ../../sources/papers/teleopit_arxiv_2608_01834.md
  - ../../sources/repos/teleopit.md
  - ../../sources/papers/humanoid_surgeon_nature_2026.md
  - ../../sources/papers/subcentimeter_pipeline_inspection_scirobotics_2022.md
  - ../../sources/repos/ssik.md
  - ../../sources/papers/xrobotoolkit_arxiv_2508_00097.md
  - ../../sources/sites/xr-robotics-github-io.md
  - ../../sources/repos/xrobotoolkit.md
  - ../../sources/repos/nvidia_isaac_teleop.md
  - ../../sources/sites/nvidia-isaac-teleop-docs.md
  - ../../sources/datasets/let-base-dataset.md
  - ../../sources/papers/nestdex_arxiv_2608_13362.md
  - ../../sources/sites/aus-bot-nestdex.md
  - ../../sources/papers/spd_corl_2026.md
  - ../../sources/sites/spd-bot.md
---

# Teleoperation（遥操作）

**一句话定义**：操作员通过外部设备实时远程控制机器人完成任务，同时采集高质量示范数据用于后续策略学习。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|---------- |
| Teleop | Teleoperation | 人远程操控机器人采集演示 |
| IL | Imitation Learning | 遥操作数据常用于 BC/扩散策略 |
| VR | Virtual Reality | 全身遥操作与参考生成接口 |
| MoCap | Motion Capture | 与遥操作并列的高质量动作来源 |
| WBC | Whole-Body Control | 低层执行全身跟踪或力控 |

## 为什么重要

遥操作是当前人形机器人获取**高质量演示数据**最可行的路线：

- 真实世界的操作任务（擦桌、开门、折衣物）难以在仿真中自动生成示范
- 大规模 RL 探索在真机上代价过高（执行器损耗、安全风险）
- 人类示范数据 + 模仿学习（BC/ACT/Diffusion Policy）已在多个任务上超过纯 RL

> "You can't imitate what you don't have." — 数据飞轮的起点是高质量遥操作系统。

## 关键挑战

### 1. 运动映射（Motion Retargeting）
人和机器人形态不同：
- 关节 DOF 不同（人手 21 DOF vs 机械手 6-12 DOF）
- 工作空间不同（机器人手臂更短/更长）
- 动力学不同（机器人惯性大、控制带宽有限）

解决方案：基于运动学约束的 IK 重定向 / 学习映射（端到端训练）

### 2. 延迟补偿
端到端延迟（感知→传输→执行）通常 50-200ms，影响精细操作：
- 降低带宽（关节命令压缩、关键帧插值）
- 预测性显示（基于模型预测未来状态）
- 减少不必要的处理步骤（低延迟 ROS2 / EtherCAT）

### 3. 数据质量与一致性
操作员疲劳、环境变化、设备噪声都影响示范质量：
- 失败示范过滤（自动 / 人工标注）
- 示范标准化（速度 / 姿态归一化）
- 多操作员数据融合策略

### 4. 人形干预接管 ≠ 专家示范（deployment-time）
机械臂 + 夹爪上较顺滑的 leader–follower / 3D 鼠标纠正，在 **全身 + 灵巧手人形** 上常出现 **对齐犹豫、回撤、重定向误差**。[ROVE](../entities/paper-rove-humanoid-vla-intervention.md)（arXiv:2606.17011）将 VLA rollout 近失败时的 MoCap 接管拆为 **rollout → adaptation → recovery**，并证明直接 HG-DAgger 式模仿干预会把 **adaptation 噪声** 学进策略；后训练应优先 **价值引导提取** 而非一律当专家 BC。产业侧对照：[CurrentWorld-0](../entities/current-robotics-currentworld.md) 把接管放进 **世界模型**（失败态保存 / 回滚 / 分支），避免在真机反复复现同一失败——与 ROVE 的真机 MoCap 接管互补，不是替代。

### 5. 双臂协调
全身遥操作需要同时控制移动基座 + 双臂：
- 基座运动 vs 手臂运动的优先级
- 碰撞回避（基座移动时避免手臂碰障碍物）
- 双臂时序协调（抓取一件物品时另一臂稳定）

## 主流遥操作系统

按 **操作员接口形态** 分族——选型时最先定下来的通常是「人手上拿/穿什么」，而不是机器人型号。

### Leader–follower 与实体主从臂

主从同构、上手快，是台式双臂精细操作的默认起点。

| 系统 | 机器人 | 输入设备 | 数据规模 | 特点 |
|------|--------|---------|---------|------ |
| ALOHA（Stanford 2023） | 4 臂台式 | Leader Arms | ~50 任务 | 低成本（$20K），精细操作 |
| GELLO（Berkeley 2023） | 多 UR/Franka | Leader Arms | 低成本 | 低成本版 ALOHA |
| **[NestDex](../entities/paper-nestdex.md)**（Usyd / PAIR Lab / Vanderbilt 2026） | Piper Nero + **WujiHand I（20-DoF）** | Leader 臂 + **1-DoF clutch**（内层手技能 copilot） | 六任务真机；外层 20 条/任务 | **嵌套采数**：人控臂与进度，部署卸掉内层；Copilot 采数 **100%**，AnyTeleop 三任务 **0%**；**未开源** |

### 可穿戴 1:1 接口（外骨骼 / 数据手套）

与末端执行器运动学同构，采数即部署，基本不做软件 retarget。

| 系统 | 机器人 | 输入设备 | 数据规模 | 特点 |
|------|--------|---------|---------|------ |
| [mimic U1 / umimic](../entities/mimic-wearable-u1.md)（mimic 2026） | mimic hand M1 | 固定运动学被动外骨骼 | 中层规模化 | 与 M1 **运动学/触觉/腕相机 1:1**；无 retargeting |
| [UME-EXO](../entities/paper-ume-exo.md)（Ant / Stanford 2026） | OpenArm 双臂移动平台等 | 上肢外骨骼 + IMU | 26–157 条/任务 | 实时触觉力矩反馈 + 全身臂形/力矩记录 + 子链重定向；ACT 学主动柔顺 |
| **[TwinDEX](../entities/twindex.md)（自变量 2026-09）** | 同构三指末端（7 主动 + 2 被动） | **可穿戴三指外骨骼**（多视 RGB + 6D 腕 + 关节 + 指尖触觉） | 无机器人采集；宣称 **5.3×** 真机遥操作吞吐 | 采数/部署 **运动学·接触·外观·时序 1:1**，关节直映、无软件 retarget；纯 robot-free 训策略；**未开源** |
| **[DEUX / Glove X](../entities/xyz-deux.md)（XYZ 2026）** | DEUX 半人形双臂服务机器人 | **Glove X**（7 关节 + 3 指尖压 + 双相机；板载 &lt;50 ms） | 真店 proprietary | 手套–三指手 **1:1 接触点**，宣称 **zero-shot retarget、免后处理**；**未开源** |

### 无机器人采集（UMI 系手持 / 头戴）

采集时不需要机器人在场，用统一夹爪 / 头戴相机换取吞吐与可迁移性。

| 系统 | 机器人 | 输入设备 | 数据规模 | 特点 |
|------|--------|---------|---------|------ |
| UMI（Stanford 2024） | 通用 | GoPro + 夹爪 | 可扩展 | 无需专用机器人，数据可迁移 |
| [HandUMI](../entities/handumi.md)（RoboNet 2026） | PiPER / OpenArm / TRLC-DK1 / YAM 等平行夹爪双臂 | HandUMI 手持接口（~$110 tip 可换）+ PICO / Quest | 无机器人采集 | **一次采集、多臂重定向**；Feetech 直测开合；LeRobot v3 + `handumi validate` QA |
| **[HiFi-UMI](../entities/paper-hifi-umi.md)（Simple AI 2026）** | 真机双臂（评测部署） | 头戴 stereo-inertial SLAM + 双手广角（六视角，~3 mm / <40 µs） | **HiFi-UMI-2K 2000 h** 已开源 | **zero-robot 后训练**匹配同域 teleop；数据 CC BY 4.0；采数代码未列 |
| [BifrostUMI](../entities/paper-bifrost-umi.md)（BAAI Aether 2026） | Unitree G1 | Pico 追踪 + 双腕鱼眼夹爪 | 无机器人采集 | UMI 式示范 + 扩散高层 + SKR → 人形全身 WBC |
| [HALOMI](../entities/paper-halomi-humanoid-loco-manipulation.md)（上交 / 萨塞克斯 / 华理 2026） | Unitree G1 + 主动颈 | Pika Sense 双夹爪 + 头盔 egocentric + Vive | 无机器人采集 | UMI+头手追踪 + π₀.₅ VLA + BFM-Zero 流形 WBC |

### XR / VR 全身人形

本页占比最大的一族：头显 + 手柄 / tracker 给全身参考，底层交给 WBC 或 motion tracking 策略。

| 系统 | 机器人 | 输入设备 | 数据规模 | 特点 |
|------|--------|---------|---------|------ |
| OmniH2O（CMU/Tsinghua 2024） | Unitree H1/G1 | VR + 手套 | 全身遥操作 | 全身 DOF 控制，含移动基座 |
| **[xr_teleoperate](../entities/xr-teleoperate.md)（Unitree 官方）** | Unitree G1 / H1 | AVP / PICO / Quest 等 XR | 官方参考实现 | 宇树开源全身遥操作主仓；可与 [unitree_sim_isaaclab](../entities/unitree-sim-isaaclab.md) 同 DDS 仿真采数；组织地图见 [Unitree](../entities/unitree.md) |
| HTD（CMU/Bosch 2026） | 人形 + 灵巧手 | VR + 摇杆 + 分布式触觉 | 5 个真实接触丰富任务 | [解耦 WBC](../entities/htd-decoupled-wbc.md) 稳定下肢已开源；VR 采数与 HTD 策略截至 2026-08-26 仍待发布 |
| [TWIST2](../entities/paper-twist2.md)（Amazon FAR, ICRA 2026） | Unitree G1 | PICO 4 Ultra + 2-DoF 颈 | 真机便携遥操作 | 全身 RL 跟踪 + 扩散 visuomotor 自主；15 min 级百次采集；底层 XR 流常用 XRoboToolkit |
| [PILOT](../entities/paper-pilot-perceptive-loco-manipulation.md)（上海交大 2026） | Unitree G1 | VR 头显 + 手柄 | 长程 loco-manip | 感知 **MoE 全身 LLC** 作底层；楼梯/高台等非结构化场景遥操作 |
| [CWI](../entities/paper-cwi-composite-humanoid-whole-body-imitation.md)（LimX / HKU 等 2026） | LimX Oli | **Meta Quest VR** + 手柄 | 全身 loco-manip | **双手 9D keypoint + 速度/身高** 蒸馏接口，无需全身 MoCap |
| [HEFT](../entities/paper-heft.md)（清华 / RobotEra 2026） | Unitree G1 + **L7**（175 cm 全尺寸） | **VR 全身参考**（部署吃 raw 流） | PMG 配对 VR + SEED 等 | **PMG** 噪声 VR 跟踪 + **WPC** 窗化双手负载；L7 **24 kg** 重载遥操作 + 高动态跟踪 |
| **[Teleopit](../entities/paper-teleopit.md)**（西湖 / 上海创智 2026） | Unitree G1 | **PICO VR** 身体+手+头 | 公开 mocap 子集 + 自采 PICO；**96** 条瓶放置演示 | **全身跟踪 + 连续跨手重定向 + 主动视觉**；History/rewind；五仓开源；ACT/GR00T **90%/95%** |
| [MotionWAM](../entities/paper-motionwam-humanoid-loco-manipulation-wam.md)（Mondo / HKUST 2026） | Unitree G1 | **PICO VR 三点追踪** + SMPL→G1 重定向 | 九项全身 loco-manip | Stage 3 **200 episodes/任务** 全身遥操作演示，供 **WAM** 微调 |
| [Being-M0.7](../entities/paper-being-m07-humanoid-latent-wam.md)（BeingBeyond 2026） | Unitree G1 | **PICO VR** 头显 + 手柄 + 踝 tracker；XRoboToolkit→SMPL→**SONIC** | 四项全身 loco-manip 后训练 | **>1 万 h** 人数据预训练 **latent video-motion 先验** 后接地 |
| [ω-0](../entities/paper-omega-0.md)（NTU / PKU / BAAI / HKUST-GZ 2026） | Unitree G1 + Inspire | **Pico VR** 头显 + 足 tracker + 手持扳机；ZED Mini ego + 房间 ZED depth；**SONIC** 遥操作策略 | ω-HOME **40.3 h / 24 任务** | 潜空间 foresight Joint WAM；评测 11 任务 Omni **SR 81.8%**；代码/数据 WIP |
| **[Immersive Social VR+LLM](../entities/paper-immersive-social-vr-llm-humanoids.md)**（NYUAD 2026） | Unitree H1 + Inspire 手 | **Apple Vision Pro** + 语音 | 多模态遥操作录制（RGB/语音/关节/眼动） | **LLM 语音高层 locomotion** + VR 腕/指操作 + ROS 双向音频社交；新手抓放 **80%** / 社交传方块 **70%**；**系统未开源** |
| **[NEXUS](../entities/nexus-humanoid.md)**（SJTU，Research Preview 2026） | （待论文） | Live human motion | — | **感知型 foundation policy** + **跨域全身遥操作** + **地形感知**（预览楼梯 sim/real）；Paper/Code **Coming Soon** |

### 外部追踪（动捕 / 光学 / 纯视觉）

把追踪算力放到环境侧，换取全局精度或免穿戴。

| 系统 | 机器人 | 输入设备 | 数据规模 | 特点 |
|------|--------|---------|---------|------ |
| [TeleGate](../entities/paper-telegate.md)（USTC / AnyWit 2026） | Unitree G1 | **惯性动捕** 全身关节跟踪 | **2.5 h** 自采六类 | **门控选冻结专家** + VAE 历史→未来先验；避免蒸馏统一策略；跑跳/跌倒恢复 |
| [CLOT](../entities/paper-amp-survey-16-clot.md)（上交 / 上海 AI Lab 2026） | Adam Pro | OptiTrack 全局反馈 | 闭环全局遥操作 | Observation Pre-shift + Transformer + AMP；长时程无漂移 |
| **[TeleDexter](../entities/paper-teledexter.md)**（清华 / BIGAI / 北大 2026） | Franka + SharpaWave / LeapHand | **NOKOV MoCap**（腕 + 指尖 + 物体 6D） | 七任务真机遥操作 + 50 demos/任务 | **hand–object co-tracking** 低层「小脑」；平均 **75.2% SR**；基线运动学/生成先验近失败；**未开源** |
| **[SPD](../entities/paper-spd.md)**（斯坦福 / MIT / Scale AI，CoRL 2026） | 双 YAM Pro + Sharpa Wave（56 DoF） | **仿真：** Quest 3 手跟踪；**真机：** Manus + Quest 手柄 | 仿真 **75 h / 5 人一周**；真机 **1–2 h/任务** | 仿真 on-embodiment 预训练 + 真机微调；五项任务胜过从零 BC；**代码数据待发布** |
| AnyTeleop（UCB 2023） | 多平台 | RGB 相机 | 通用 | 无传感器手套，仅视觉输入 |

### 平台、中间件、数据集与专用领域

不是某一套具体机器，而是采集栈、公开数据或垂直场景。

| 系统 | 机器人 | 输入设备 | 数据规模 | 特点 |
|------|--------|---------|---------|------ |
| **[Isaac Teleop](../entities/isaac-teleop.md)**（NVIDIA，文档 1.5.x） | Franka / G1 / GR1T2 等 Lab 环境；真机 + ROS 2 | AVP / Quest / Pico + 手套/踏板；CloudXR | MCAP / HDF5 / **LeRobot** | **已开源** Apache-2.0 + PyPI；图式 retargeting；**Televiz** XR 合成；第一人称无标记手重建（MANO 门禁）；Lab 3.x XR 主线 |
| **[XRoboToolkit](../entities/paper-xrobotoolkit.md)**（ByteDance PICO / GT / GMU，SII 2026） | UR5 / ARX R5 / Galaxea R1-Lite / Shadow Hand | PICO 4 Ultra / Quest 3（OpenXR） | 跨平台 XR 中间层 | 低延迟立体视觉 + QP-IK + 多模态追踪；全栈开源；π₀ 采数验证 |
| [HIW-500](../entities/hiw-500-dataset.md)（BitRobot / Unitree / HF 2026） | Unitree G1 | 全身遥操作 | **500+ h / 23K+ 集** | 东南亚 **12** 个真实家庭、**10+** 家务任务；开源最大规模人形遥操作集之一 |
| [REK](../entities/rek.md)（2025–） | Unitree G1 / H 系 | VR 头显（REK TEK） | 售票现场赛 | **全接触格斗** 竞技向全身映射；非 IL 数据集导向 |
| [URKL](../entities/urkl.md)（2026–） | EngineAI T800 | **队方自控算法**（非 VR pilot；开幕战自主比例仍待官方规则核实） | 全球算法联赛 | **标准化硬件 + 差异化运控**；开幕展览后 8 月 Top32 实机选拔；与 REK **遥操作** 路线对照 |
| [Humanoid Surgeon](../entities/paper-humanoid-surgeon-in-vivo-laparoscopy.md)（UCSD 2026） | Surgie 人形 | 手术控制台 + 立体视觉 | in vivo 猪模型 | **腹腔镜胆囊切除**；通用器械 + RCM 遥操作；Nature 活体可行性 |

## 工程实践：操作员侧、I/O 与低层执行

上表定的是「人穿/拿什么」，这一节定的是「信号从操作员到关节要过哪些层」。

### 操作员侧可穿戴力反馈（研究参考）

机器人端触觉传感（F/T、GelSight 等）解决的是 **策略输入**；操作员在 VR/遥操作中还需感知 **虚拟或远程物体的尺寸与刚度**。[HapMorph](../entities/paper-hapmorph-pneumatic-haptic-render.md)（arXiv:2509.05433，SSSA）用 **拮抗式织物气动执行器（AFPA）** 在 **21 g** 可穿戴形态下，经双腔压力解耦渲染 **50–104 mm** 尺寸与 **~0.12–4.7 N/mm** 刚度；10 人感知实验对 9 种尺寸×刚度组合的识别准确率达 **89.4%**。与 [UME-EXO](../entities/paper-ume-exo.md) 的外骨骼力矩反馈、[数据手套 vs 视觉遥操作](../comparisons/data-gloves-vs-vision-teleop.md) 中的 **运动采集** 通道正交——HapMorph 属于 **力触觉显示** 而非示教采集。当前原型仍依赖外部气源，尚非可直接部署的遥操作商品件。

### 跨形态实时 I/O（RIO / 手柄 / 操作员 HMI）

**[RIO（Robot I/O）](../entities/robot-io-rio.md)**（CMU / Bosch 等，RSS 2026 接收）把 Spacemouse、手柄、键盘、GELLO、手机与 VR 等遥操作入口，与多相机、机械臂/人形控制、数据记录和 **异步 VLA 推理** 收拢到同一套 **Node + 可切换中间件** 抽象里：换设备组合以 **station 配置** 为主，主循环尽量保持通用。适合作为「多接口采集 + 低延迟闭环」的对照阅读，与上表中以单一系统命名的经典遥操作栈互补。

在 **Linux 工作站** 上，**USB 有线 Xbox 手柄** 通常由内核 **[xpad](../entities/xpad.md)** 驱动暴露为 `/dev/input/js*` 与 evdev 节点，再被 pygame、ROS `joy` 或 RIO 手柄 Node 读取；**蓝牙配对** 的 Xbox 手柄则走通用 HID，不经过 xpad。部署前需分清连接方式，避免「模块已加载但无输入」的误判。

在 **Windows 工控机** 上，游戏手柄经 **XInput / `Windows.Gaming.Input`** 进入应用；若需要 **原生操作员控制台**（多路相机预览、模式切换、急停、状态面板），常见选型是 **[WinUI 3](../entities/winui.md)**（Fluent 控件 + XAML）与 [ONNX Runtime](../entities/onnxruntime.md) C# 推理同栈集成。WinUI 解决 **界面可读性与任务编排**，不替代 [RIO](../entities/robot-io-rio.md) 等跨形态 IO 中间件；GUI 设计取舍可参考 [非专家遥操作 GUI 论文笔记](../entities/paper-notebook-intuitive-gui-for-non-expert-teleoperation-of-hu.md)。宿主 Windows 若需 **debloat 与减遥测**（长时间采数时降低后台 IO），可对照开源 Playbook **[Atlas OS](../entities/atlas-os.md)**（不重打包 ISO，安全选项可审计）。

### 臂部笛卡尔跟踪：解析 IK 参考（ssik）

VR / 手柄每 tick 给出末端 `T_target` 时，常见模式是 **「离当前关节角最近的单解 IK」** 以保持构型连续。**[ssik](../entities/ssik.md)**（UW PRL）对 **6R/7R 机械臂** 提供 **解析全分支** 再按 `q_seed` 排序：`solve(T, max_solutions=1, q_seed=q_current, seed_tolerance=…)`；空结果表示该姿态下无法在容差内平滑延续，应触发重规划。覆盖 **Franka、iiwa、xArm、非 Pieper 6R** 等 EAIK 常拒绝的几何；与 [MoveIt 2](../entities/moveit2.md) 数值 IK、[cuRobo](../entities/curobo.md) GPU IK 可分层组合（解析分支枚举 → 碰撞/规划）。

### 全身人形：视频 / VR 条件 + 低层 tracking

NVIDIA **SONIC** 项目页（[GEAR-SONIC](https://nvlabs.github.io/GEAR-SONIC/)）把遥操作与 **规模化 motion tracking policy** 放在同一套 **统一 token / 控制接口** 下展示：人体视频经 **GEM** 估计姿态后实时跟踪；VR 含 **头 + 双手三点** 驱动上身并由 **运动学规划器** 补全下身，以及 **全身 VR 追踪** 模式。知识库方法页见 [SONIC（规模化运动跟踪人形控制）](../methods/sonic-motion-tracking.md)。[HumanoidArena](../entities/paper-humanoidarena.md) 将 SONIC 与 [TWIST2](../entities/paper-twist2.md) 并列为 **分层全身学习的双 GMT 评测后端**，在共享 **35D 上游参考** 下 stress-test **跨 tracker 迁移**。

## 遥操作到策略学习 Pipeline

```
遥操作采集
    ↓ 示范数据（obs, action 序列）
数据预处理（过滤/标准化）
    ↓
模仿学习训练
  ├─ Behavior Cloning（BC）：直接复制
  ├─ ACT：Action Chunking + CVAE
  ├─ Diffusion Policy：扩散生成动作序列
  └─ IL + RL fine-tune：提升鲁棒性
    ↓
部署（真机 / 仿真验证）
```

## 评估指标

| 指标 | 说明 |
|------|------ |
| 任务成功率 | N 次尝试中完成任务的比例 |
| 完成时间 | 完成任务的平均时长（越短越好） |
| 操作员上手时间 | 新操作员达到可接受质量所需的训练时间 |
| 端到端延迟 | 操作员输入到机器人执行的延迟 |
| 示范数据效率 | 用多少条示范可以训练出成功率 ≥ X% 的策略 |

## 参考来源

### ingest 档案（sources/ 归档）

- [sources/papers/nestdex_arxiv_2608_13362.md](../../sources/papers/nestdex_arxiv_2608_13362.md)、[sources/sites/aus-bot-nestdex.md](../../sources/sites/aus-bot-nestdex.md) — NestDex：内层手技能 copilot 采数 + 独立外层 visuomotor（arXiv:2608.13362；未开源）
- [sources/papers/spd_corl_2026.md](../../sources/papers/spd_corl_2026.md)、[sources/sites/spd-bot.md](../../sources/sites/spd-bot.md) — SPD：仿真 VR 灵巧手预训练 75 h + 真机 1–2 h 微调（CoRL 2026；代码数据待发布）
- [sources/papers/teledexter_arxiv_2607_11481.md](../../sources/papers/teledexter_arxiv_2607_11481.md)、[sources/sites/teledexter-project.md](../../sources/sites/teledexter-project.md) — TeleDexter：hand–object co-tracking 灵巧遥操作（arXiv:2607.11481；未开源）
- [sources/sites/engineai-urkl.md](../../sources/sites/engineai-urkl.md) — URKL：EngineAI 统一 T800 自主算法格斗联赛
- [sources/sites/urkl-org.md](../../sources/sites/urkl-org.md) — URKL 独立导读站（证据链 / 赛程）
- [sources/blogs/wechat_urkl_faq_01.md](../../sources/blogs/wechat_urkl_faq_01.md) — 众擎 URKL 官方 FAQ（开源承诺 / 商业化）
- [sources/sites/rek-com.md](../../sources/sites/rek-com.md) — REK：VR 驱动 G1 全接触格斗联赛与租赁
- [sources/sites/hiw-500-dataset.md](../../sources/sites/hiw-500-dataset.md) — HIW-500：东南亚 12 家庭 G1 全身遥操作 **500+ h / 23K+ 集** 开源数据集
- [sources/sites/x2robot-twindex.md](../../sources/sites/x2robot-twindex.md) — TwinDEX：三指外骨骼–同构手共设计无本体采数（2026-09-02；未开源）
- [sources/blogs/mimicrobotics_m1_u1_full_stack.md](../../sources/blogs/mimicrobotics_m1_u1_full_stack.md) — mimic U1 固定运动学外骨骼 + M1 全栈灵巧平台（2026-07）
- [sources/papers/teleoperation.md](../../sources/papers/teleoperation.md) — ALOHA / OmniH2O / UMI / BifrostUMI / AnyTeleop
- [sources/papers/xrobotoolkit_arxiv_2508_00097.md](../../sources/papers/xrobotoolkit_arxiv_2508_00097.md)、[sources/sites/xr-robotics-github-io.md](../../sources/sites/xr-robotics-github-io.md)、[sources/repos/xrobotoolkit.md](../../sources/repos/xrobotoolkit.md) — XRoboToolkit：OpenXR 跨平台 XR 遥操作套件（arXiv:2508.00097，SII 2026 Best Paper）
- [sources/repos/nvidia_isaac_teleop.md](../../sources/repos/nvidia_isaac_teleop.md)、[sources/sites/nvidia-isaac-teleop-docs.md](../../sources/sites/nvidia-isaac-teleop-docs.md) — Isaac Teleop：Lab 3.x XR 主线 + Televiz + 无标记手重建（文档 1.5.x，2026-09-05 复核）
- [sources/papers/ume_exo_arxiv_2606_14218.md](../../sources/papers/ume_exo_arxiv_2606_14218.md)、[sources/sites/ume-exo-project.md](../../sources/sites/ume-exo-project.md) — UME-EXO：外骨骼实时力矩反馈 + 全身臂形采集 + ACT 柔顺策略（arXiv:2606.14218）
- [sources/papers/bifrost_umi_arxiv_2605_03452.md](../../sources/papers/bifrost_umi_arxiv_2605_03452.md) — BifrostUMI：无机器人采集 + SKR + G1 全身 visuomotor（arXiv:2605.03452）
- [sources/papers/halomi_arxiv_2606_18772.md](../../sources/papers/halomi_arxiv_2606_18772.md) — HALOMI：egocentric 无机器人示范 + 主动颈 G1 loco-manipulation（arXiv:2606.18772）
- [sources/sites/twist2-project.md](../../sources/sites/twist2-project.md)、[sources/repos/twist2.md](../../sources/repos/twist2.md) — TWIST2 项目页 + 开源仓库（arXiv:2511.02832）
- [sources/papers/clot_arxiv_2602_15060.md](../../sources/papers/clot_arxiv_2602_15060.md)、[sources/sites/clot-project.md](../../sources/sites/clot-project.md) — CLOT 闭环全局遥操作（arXiv:2602.15060；官方页非 clot.github.io）
- [sources/papers/diffusion_and_gen.md](../../sources/papers/diffusion_and_gen.md) — ACT / Diffusion Policy（遥操作数据的下游学习方法）
- [sources/papers/humanoid_touch_dream.md](../../sources/papers/humanoid_touch_dream.md) — HTD 的 VR 全身遥操作、LBC 和触觉同步采集系统
- [sources/repos/isaaclab_decoupled_wbc.md](../../sources/repos/isaaclab_decoupled_wbc.md) — HTD LBC 已开源；遥操作代码仍 on-going
- [sources/repos/robot-io-rio.md](../../sources/repos/robot-io-rio.md) — RIO 的多设备遥操作与实时 Node 管线（arXiv:2605.11564）
- [sources/repos/xpad.md](../../sources/repos/xpad.md) — Linux USB Xbox 手柄内核驱动（paroj/xpad）
- [sources/repos/microsoft-ui-xaml.md](../../sources/repos/microsoft-ui-xaml.md) — WinUI 3：Windows 工控机操作员 HMI 官方 UI 栈（MIT，WinAppSDK 2.4.0）
- [sources/papers/bfm_humanoid_arxiv_2509_13780.md](../../sources/papers/bfm_humanoid_arxiv_2509_13780.md) — BFM：CVAE + 掩码在线蒸馏，让单一策略统一覆盖人形跟踪 / VR 遥操作 / locomotion 多接口
- [sources/papers/pilot_arxiv_2601_17440.md](../../sources/papers/pilot_arxiv_2601_17440.md) — PILOT：VR 遥操作 + 感知 MoE 全身 LLC（arXiv:2601.17440）
- [sources/papers/cwi_arxiv_2606_27676.md](../../sources/papers/cwi_arxiv_2606_27676.md) — CWI：Meta Quest 双手 + 速度/身高全身 loco-manipulation（arXiv:2606.27676）
- [sources/papers/rove_arxiv_2606_17011.md](../../sources/papers/rove_arxiv_2606_17011.md)、[sources/sites/xpeng-robotics-rove.md](../../sources/sites/xpeng-robotics-rove.md) — ROVE：人形 VLA 部署干预采集与次优接管轨迹的 RL 后训练（arXiv:2606.17011）
- [sources/papers/hapmorph_arxiv_2509_05433.md](../../sources/papers/hapmorph_arxiv_2509_05433.md) — HapMorph：AFPA 可穿戴气动框架解耦渲染尺寸与刚度（arXiv:2509.05433）
- [sources/sites/telegate-project.md](../../sources/sites/telegate-project.md)、[sources/papers/telegate_arxiv_2602_09628.md](../../sources/papers/telegate_arxiv_2602_09628.md) — TeleGate：门控专家 + VAE 运动先验全身遥操作（RSS 2026，arXiv:2602.09628）
- [sources/sites/heft-project.md](../../sources/sites/heft-project.md)、[sources/papers/heft_arxiv_2607_02332.md](../../sources/papers/heft_arxiv_2607_02332.md)、[sources/repos/axellwppr_motion_tracking.md](../../sources/repos/axellwppr_motion_tracking.md) — HEFT：PMG + WPC 重载全尺寸人形 VR 遥操作（arXiv:2607.02332）
- [sources/sites/teleopit-project.md](../../sources/sites/teleopit-project.md)、[sources/papers/teleopit_arxiv_2608_01834.md](../../sources/papers/teleopit_arxiv_2608_01834.md)、[sources/repos/teleopit.md](../../sources/repos/teleopit.md) — Teleopit：VR 全身体+连续灵巧手+主动视觉（arXiv:2608.01834）
- [sources/papers/immersive_social_vr_llm_humanoids_arxiv_2607_07430.md](../../sources/papers/immersive_social_vr_llm_humanoids_arxiv_2607_07430.md) — Immersive Social VR+LLM：AVP + 语音高层 locomotion + 双向音频社交（arXiv:2607.07430；未开源）
- [sources/papers/humanoidarena_arxiv_2606_17833.md](../../sources/papers/humanoidarena_arxiv_2606_17833.md) — HumanoidArena：PICO egocentric 采集 + GMR → 双 GMT 分层 benchmark（arXiv:2606.17833）
- [sources/papers/humanoid_surgeon_nature_2026.md](../../sources/papers/humanoid_surgeon_nature_2026.md) — Humanoid Surgeon：人形腹腔镜 in vivo 可行性（Nature 2026；[项目页](https://humanoid-surgeon.github.io/)）

### 外部论文与笔记

- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (RSS 2023) — ALOHA
- He et al., *OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation* (2024)
- [机器人论文阅读笔记：OmniH2O](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/OmniH2O_Universal_Whole-Body_Teleoperation/OmniH2O_Universal_Whole-Body_Teleoperation.html)

## 关联页面

### 先读：上游任务与概念

- [Loco-Manipulation](./loco-manipulation.md) — 遥操作在移动操作中的应用
- [Manipulation](./manipulation.md) — 操作任务整体视角
- [Motion Retargeting](../concepts/motion-retargeting.md) — 人类动作到机器人动作的映射
- [Imitation Learning](../methods/imitation-learning.md) — 遥操作数据的学习方法
- [Diffusion Policy](../methods/diffusion-policy.md) — 遥操作数据训练的扩散策略
- [Humanoid Transformer with Touch Dreaming](../methods/humanoid-transformer-touch-dreaming.md) — 使用触觉遥操作数据训练接触感知策略
- [Query：操作演示数据采集指南](../queries/demo-data-collection-guide.md) — 遥操作采集数据的实操指南
- [具身数据采集四层术语地图](../concepts/embodied-data-collection-four-layers-taxonomy.md) — 视角/设备/教法/产物判读；teleop 在教法层的定位

### 采集接口与硬件（穿戴 / 手持 / 主从）

- [Bioimpedance EIM（Science Robotics 2026）](../entities/paper-bioimpedance-eim-wearable-myography.md) — 双频可穿戴阻抗肌电；功能性运动中编码 **肌束长度 + 激活**（非 kinematic teleop 接口，但指向 **辅助 / 外骨骼闭环** 人体侧传感）
- [HandUMI](../entities/handumi.md) — 开源无机器人示教；一次采集多臂重定向
- [HiFi-UMI / HiFi-UMI-2K（论文实体）](../entities/paper-hifi-umi.md) — 高保真 UMI 2000 h；zero-robot 后训练（arXiv:2607.25895）
- [BifrostUMI（论文实体）](../entities/paper-bifrost-umi.md) — 无机器人示范 → 人形全身扩散策略 + SKR
- [UME-EXO（论文实体）](../entities/paper-ume-exo.md) — 外骨骼力矩反馈 + 全身臂形示教 → ACT 主动柔顺策略
- [DEUX / Glove X（XYZ）](../entities/xyz-deux.md) — 商业手套–三指手 1:1 零样本重定向采数（闭源）
- [TwinDEX（自变量）](../entities/twindex.md) — 三指外骨骼–机械手共设计；纯 robot-free、无软件 retarget（闭源）
- [NestDex（论文实体）](../entities/paper-nestdex.md) — 内层手技能 + clutch copilot 采灵巧示范，部署卸 copilot（arXiv:2608.13362）
- [TeleDexter（论文实体）](../entities/paper-teledexter.md) — hand–object co-tracking 灵巧遥操作与采数引擎
- [SPD（论文实体）](../entities/paper-spd.md) — 仿真 VR 75 h 预训练；真机每任务 1–2 h 微调（CoRL 2026；代码待发布）
- [HapMorph（论文实体）](../entities/paper-hapmorph-pneumatic-haptic-render.md) — 操作员侧可穿戴尺寸+刚度力触觉渲染（arXiv:2509.05433）
- [Ego-OSCAR / Stereo-550（论文实体）](../entities/paper-ego-oscar.md) — 观测-only 开源硬件立体+IMU 头戴（~USD 200；非 teleop/EE 通道）
- [reBot-DevArm（Seeed B601）](../entities/rebot-devarm.md) — 开源桌面臂；Star Arm 102 Leader + LeRobot / ROS2 遥操作采数路径
- [Transformer Transformer（论文实体）](../entities/paper-transformer-transformer.md) — UMI 示范 → 运动条件机体共设计（ALOHA/双臂）

### 全身人形系统与低层执行

- [SPOT 长时人形遥操作](../entities/paper-spot-humanoid-teleoperation.md) — 双目鱼眼 + 广 FOV 立体显示；视点-动作解耦（arXiv:2609.07933；未见代码）
- [TWIST2（论文实体）](../entities/paper-twist2.md) — 便携真机全身遥操作 → visuomotor 自主
- [Teleopit（论文实体）](../entities/paper-teleopit.md) — PICO VR 全身+连续跨手+主动视觉；History/rewind；96 演示 ACT/GR00T（arXiv:2608.01834）
- [HEFT（论文实体）](../entities/paper-heft.md) — 嘈杂 raw VR + WPC 双手负载；全尺寸 L7 **24 kg** 重载遥操作（arXiv:2607.02332）
- [TeleGate（论文实体）](../entities/paper-telegate.md) — 惯性动捕 + 门控冻结专家 + VAE 预判；2.5 h 高动态全身遥操作（RSS 2026，arXiv:2602.09628）
- [CLOT（论文实体）](../entities/paper-amp-survey-16-clot.md) — 闭环全局位姿的长时程全身遥操作
- [PILOT（论文实体）](../entities/paper-pilot-perceptive-loco-manipulation.md) — VR 长程 loco-manipulation 与非结构化地形底层控制
- [CWI（论文实体）](../entities/paper-cwi-composite-humanoid-whole-body-imitation.md) — Quest VR 双手接口 + 复合全身模仿 loco-manipulation（arXiv:2606.27676）
- [ω-0（论文实体）](../entities/paper-omega-0.md) — Pico VR + SONIC 采集 ω-HOME；潜空间 foresight 家务并发 loco-manip（arXiv:2608.06375）
- [HTD 解耦 WBC](../entities/htd-decoupled-wbc.md) — HTD 开源下肢控制器；采数栈仍待发布
- [motion_tracking（代码实体）](../entities/axellwppr-motion-tracking.md) — HEFT 官方 mjlab 训练与 sim2real 检查点
- [HumanoidArena（论文实体）](../entities/paper-humanoidarena.md) — PICO egocentric 采集管线与 TWIST2/SONIC 双 GMT 分层 benchmark（arXiv:2606.17833）
- [REFINE-DP（论文实体）](../entities/paper-loco-manip-161-157-refine-dp.md) — VR 遥操作约 50 条 + 启发式扩数据，再 DPPO 联合微调（arXiv:2603.13707）

### 平台、中间件与操作员工作站

- [XRoboToolkit（论文实体）](../entities/paper-xrobotoolkit.md) — OpenXR 跨平台 XR 遥操作中间层（PICO/Quest；全栈开源）
- [Isaac Teleop](../entities/isaac-teleop.md) — NVIDIA Isaac Lab / Sim / ROS 2 统一 XR 遥操作；Televiz + LeRobot + 无标记手重建
- [Isaac Lab](../entities/isaac-lab.md) — 集成宿主与内置遥操作环境
- [RIO（Robot I/O）](../entities/robot-io-rio.md) — 跨形态 Node 化遥操作与异步策略推理栈
- [xpad](../entities/xpad.md) — Linux USB Xbox 手柄内核驱动与 joystick/evdev 接口
- [WinUI](../entities/winui.md) — Windows 工控机原生操作员控制台（Fluent / XAML）
- [ssik（解析逆运动学）](../entities/ssik.md) — 6R/7R 臂部笛卡尔跟踪：`q_seed` 最近分支与 `seed_tolerance` 跳变检测
- [X2Streaming-TTS（论文实体）](../entities/paper-x2streaming-tts.md) — LLM **令牌级**流式 TTS + 语音状态继承；首音频 15.8 ms（arXiv:2608.18661）
- [LeTools](../entities/letools.md) — 把遥操 rosbag 转成 LeRobot 并部署

### 数据集与下游后训练

- [LET-Base-Dataset](../entities/let-base-dataset.md) — Kuavo VR/全身增量遥操真机小时（OpenLET Base）
- [ROVE（论文实体）](../entities/paper-rove-humanoid-vla-intervention.md) — 人形 VLA 近失败 MoCap 接管与三阶段干预标注（arXiv:2606.17011）
- [CurrentWorld-0](../entities/current-robotics-currentworld.md) — 世界模型内人类接管、回滚与分支后训练（确认未开源）
- [REALab 14 篇技术地图（2026）](../overview/realab-14-papers-technology-map-2026.md) — ModPack 模块化遥操作、HoMMI 无机器人全身示范、UMI-FT 野外力感知采集
- [LEGS（论文实体）](../entities/paper-legs-embodied-gaussian-splatting-vla.md) — 无遥操作合成 loco-manip 演示 vs teleop 数据成本（arXiv:2606.01458）

### 相邻方向与专用领域

- [REK](../entities/rek.md) — VR 格斗体育：竞技向全身 teleop 极端场景
- [Humanoid Surgeon（论文实体）](../entities/paper-humanoid-surgeon-in-vivo-laparoscopy.md) — 通用人形腹腔镜遥操作 in vivo 猪模型验证（Nature 2026）
- [亚厘米级管道检测机器人（论文实体）](../entities/paper-subcentimeter-pipeline-inspection-robot.md) — 清华 DEA 软体蠕虫机器人搭载微型摄像头遥控管内视频检测（Science Robotics 2022）
- [SHELLS（论文实体）](../entities/paper-shells-layered-surface-sampling.md) — 标定多视角前馈稠密语义人头；telepresence / 表情 performance 注册上游（arXiv:2605.31283）
- [UMA（论文实体）](../entities/paper-uma.md) — 多级表面对齐超精细可驱动着装人体；VR telepresence / 变焦数字人资产（arXiv:2506.01802，部分开源）
- [DynHair（论文实体）](../entities/paper-dynhair.md) — 多视角显式发丝动态人头化身；telepresence 头发动力学（ECCV 2026，arXiv:2607.23861，GitHub 占位仓）
- [NPHM（论文实体）](../entities/paper-nphm.md) — 神经参数化 **完整人头** morphable model（SDF 身份 + 形变表情 + 局部场；CVPR 2023，代码+预训练已开源）

## 推荐继续阅读

- [HEFT 项目页](https://heft.axell.top/)
- [HEFT（论文实体）](../entities/paper-heft.md) — PMG / WPC 与 G1+L7 评测归纳
- [TeleGate 项目页](https://anywitresearch.github.io/TeleGate/)
- [TeleGate（论文实体）](../entities/paper-telegate.md) — 门控专家选择与运动先验归纳
- [机器人论文阅读笔记：TeleGate](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/TeleGate__Whole-Body_Humanoid_Teleoperation_via_Gated_Expert_Selection_with_Motion_Prior/TeleGate__Whole-Body_Humanoid_Teleoperation_via_Gated_Expert_Selection_with_Motion_Prior.html)
- [机器人论文阅读笔记：SEW-Mimic](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/SEW-Mimic__Closed-Form_Geometric_Retargeting_Solver_for_Upper_Body_Humanoid_Teleoperation/SEW-Mimic__Closed-Form_Geometric_Retargeting_Solver_for_Upper_Body_Humanoid_Teleoperation.html)
- [机器人论文阅读笔记：Learning Adaptive Neural Teleoperation for Humanoid Robots](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Learning_Adaptive_Neural_Teleoperation_for_Humanoid_Robots/Learning_Adaptive_Neural_Teleoperation_for_Humanoid_Robots.html)
- [机器人论文阅读笔记：HumanPlus](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/HumanPlus_Humanoid_Shadowing_and_Imitation_from_Humans/HumanPlus_Humanoid_Shadowing_and_Imitation_from_Humans.html)
- [机器人论文阅读笔记：ExtremControl](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/ExtremControl__Low-Latency_Humanoid_Teleoperation_with_Direct_Extremity_Control/ExtremControl__Low-Latency_Humanoid_Teleoperation_with_Direct_Extremity_Control.html)
- [机器人论文阅读笔记：HOMIE Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/HOMIE_Humanoid_Loco-Manipulation_with_Isomorphic_Exoskeleton_Cockpit/HOMIE_Humanoid_Loco-Manipulation_with_Isomorphic_Exoskeleton_Cockpit.html)
- [ALOHA 项目主页](https://mobile-aloha.github.io/) — Stanford 双臂遥操作系统
- [OmniH2O 论文](https://arxiv.org/abs/2406.08858) — 人形全身遥操作
- [ACT 论文](https://arxiv.org/abs/2304.13705) — 遥操作数据 + 动作块学习
