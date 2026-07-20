---
type: task
tags: [teleoperation, manipulation, loco-manipulation, data-collection, humanoid]
status: complete
summary: "Teleoperation 让人类通过远程接口直接操作机器人，是数据采集和复杂任务执行的重要桥梁。"
updated: 2026-07-20
sources:
  - ../../sources/blogs/mimicrobotics_m1_u1_full_stack.md
  - ../../sources/sites/rek-com.md
  - ../../sources/sites/hiw-500-dataset.md
  - ../../sources/papers/ume_exo_arxiv_2606_14218.md
  - ../../sources/papers/rove_arxiv_2606_17011.md
  - ../../sources/papers/pilot_arxiv_2601_17440.md
  - ../../sources/papers/halomi_arxiv_2606_18772.md
  - ../../sources/papers/hapmorph_arxiv_2509_05433.md
  - ../../sources/papers/teleoperation.md
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/papers/humanoid_touch_dream.md
  - ../../sources/repos/robot-io-rio.md
  - ../../sources/repos/xpad.md
  - ../../sources/papers/cwi_arxiv_2606_27676.md
  - ../../sources/sites/telegate-project.md
  - ../../sources/papers/telegate_arxiv_2602_09628.md
  - ../../sources/sites/heft-project.md
  - ../../sources/papers/heft_arxiv_2607_02332.md
  - ../../sources/repos/axellwppr_motion_tracking.md
  - ../../sources/papers/humanoid_surgeon_nature_2026.md
---

# Teleoperation（遥操作）

**一句话定义**：操作员通过外部设备实时远程控制机器人完成任务，同时采集高质量示范数据用于后续策略学习。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
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
机械臂 + 夹爪上较顺滑的 leader–follower / 3D 鼠标纠正，在 **全身 + 灵巧手人形** 上常出现 **对齐犹豫、回撤、重定向误差**。[ROVE](../entities/paper-rove-humanoid-vla-intervention.md)（arXiv:2606.17011）将 VLA rollout 近失败时的 MoCap 接管拆为 **rollout → adaptation → recovery**，并证明直接 HG-DAgger 式模仿干预会把 **adaptation 噪声** 学进策略；后训练应优先 **价值引导提取** 而非一律当专家 BC。

### 4. 双臂协调
全身遥操作需要同时控制移动基座 + 双臂：
- 基座运动 vs 手臂运动的优先级
- 碰撞回避（基座移动时避免手臂碰障碍物）
- 双臂时序协调（抓取一件物品时另一臂稳定）

## 主流遥操作系统

| 系统 | 机器人 | 输入设备 | 数据规模 | 特点 |
|------|--------|---------|---------|------|
| ALOHA（Stanford 2023） | 4 臂台式 | Leader Arms | ~50 任务 | 低成本（$20K），精细操作 |
| OmniH2O（CMU/Tsinghua 2024） | Unitree H1/G1 | VR + 手套 | 全身遥操作 | 全身 DOF 控制，含移动基座 |
| **xr_teleoperate（Unitree 官方）** | Unitree G1 / H1 | AVP / PICO / Quest 等 XR | 官方参考实现 | 宇树开源全身遥操作主仓；可与 `unitree_sim_isaaclab` 同 DDS 仿真采数；组织地图见 [Unitree](../entities/unitree.md) |
| HTD（CMU/Bosch 2026） | 人形 + 灵巧手 | VR + 摇杆 + 分布式触觉 | 5 个真实接触丰富任务 | LBC 稳定下肢，VR 采集上身/手部示范，并同步手部力与触觉 |
| UMI（Stanford 2024） | 通用 | GoPro + 夹爪 | 可扩展 | 无需专用机器人，数据可迁移 |
| HandUMI（RoboNet 2026） | PiPER / OpenArm / TRLC-DK1 / YAM 等平行夹爪双臂 | HandUMI 手持接口 + PICO / Quest | 无机器人采集 | **一次采集、多臂重定向**；LeRobot v3 兼容 + 内置校准/QA；见 [实体](../entities/handumi.md) |
| mimic U1 / umimic（mimic 2026） | mimic hand M1 | 固定运动学被动外骨骼 | 中层规模化 | 与 M1 **运动学/触觉/腕相机 1:1**；无 retargeting；见 [实体](../entities/mimic-wearable-u1.md) |
| UME-EXO（Ant / Stanford 2026） | OpenArm 双臂移动平台等 | 上肢外骨骼 + IMU | 26–157 条/任务 | 实时触觉力矩反馈 + 全身臂形/力矩记录 + 子链重定向；ACT 学主动柔顺；见 [论文实体](../entities/paper-ume-exo.md) |
| BifrostUMI（BAAI Aether 2026） | Unitree G1 | Pico 追踪 + 双腕鱼眼夹爪 | 无机器人采集 | UMI 式示范 + 扩散高层 + SKR → 人形全身 WBC；见 [论文实体](../entities/paper-bifrost-umi.md) |
| HALOMI（上交 / 萨塞克斯 / 华理 2026） | Unitree G1 + 主动颈 | Pika Sense 双夹爪 + 头盔 egocentric + Vive | 无机器人采集 | UMI+头手追踪 + π₀.₅ VLA + BFM-Zero 流形 WBC；见 [论文实体](../entities/paper-halomi-humanoid-loco-manipulation.md) |
| TWIST2（Amazon FAR, ICRA 2026） | Unitree G1 | PICO 4 Ultra + 2-DoF 颈 | 真机便携遥操作 | 全身 RL 跟踪 + 扩散 visuomotor 自主；15 min 级百次采集；见 [论文实体](../entities/paper-twist2.md) |
| CLOT（上交 / 上海 AI Lab 2026） | Adam Pro | OptiTrack 全局反馈 | 闭环全局遥操作 | Observation Pre-shift + Transformer + AMP；长时程无漂移；见 [论文实体](../entities/paper-amp-survey-16-clot.md) |
| PILOT（上海交大 2026） | Unitree G1 | VR 头显 + 手柄 | 长程 loco-manip | 感知 **MoE 全身 LLC** 作底层；楼梯/高台等非结构化场景遥操作；见 [论文实体](../entities/paper-pilot-perceptive-loco-manipulation.md) |
| MotionWAM（Mondo / HKUST 2026） | Unitree G1 | **PICO VR 三点追踪** + SMPL→G1 重定向 | 九项全身 loco-manip | Stage 3 **200 episodes/任务** 全身遥操作演示，供 **WAM** 微调；见 [论文实体](../entities/paper-motionwam-humanoid-loco-manipulation-wam.md) |
| Being-M0.7（BeingBeyond 2026） | Unitree G1 | **PICO VR** 头显 + 手柄 + 踝 tracker；XRoboToolkit→SMPL→**SONIC** | 四项全身 loco-manip 后训练 | **>1 万 h** 人数据预训练 **latent video-motion 先验** 后接地；见 [论文实体](../entities/paper-being-m07-humanoid-latent-wam.md) |
| CWI（LimX / HKU 等 2026） | LimX Oli | **Meta Quest VR** + 手柄 | 全身 loco-manip | **双手 9D keypoint + 速度/身高** 蒸馏接口，无需全身 MoCap；见 [论文实体](../entities/paper-cwi-composite-humanoid-whole-body-imitation.md) |
| HIW-500（BitRobot / Unitree / HF 2026） | Unitree G1 | 全身遥操作 | **500+ h / 23K+ 集** | 东南亚 **12** 个真实家庭、**10+** 家务任务；开源最大规模人形遥操作集之一；见 [数据集实体](../entities/hiw-500-dataset.md) |
| TeleGate（USTC / AnyWit 2026） | Unitree G1 | **惯性动捕** 全身关节跟踪 | **2.5 h** 自采六类 | **门控选冻结专家** + VAE 历史→未来先验；避免蒸馏统一策略；跑跳/跌倒恢复；见 [论文实体](../entities/paper-telegate.md) |
| HEFT（清华 / RobotEra 2026） | Unitree G1 + **L7**（175 cm 全尺寸） | **VR 全身参考**（部署吃 raw 流） | PMG 配对 VR + SEED 等 | **PMG** 噪声 VR 跟踪 + **WPC** 窗化双手负载；L7 **24 kg** 重载遥操作 + 高动态跟踪；见 [论文实体](../entities/paper-heft.md) |
| AnyTeleop（UCB 2023） | 多平台 | RGB 相机 | 通用 | 无传感器手套，仅视觉输入 |
| GELLO（Berkeley 2023） | 多 UR/Franka | Leader Arms | 低成本 | 低成本版 ALOHA |
| REK（2025–） | Unitree G1 / H 系 | VR 头显（REK TEK） | 售票现场赛 | **全接触格斗** 竞技向全身映射；非 IL 数据集导向；见 [REK 实体](../entities/rek.md) |
| URKL（2026–） | EngineAI T800 | **队方自控算法**（非 VR pilot） | 全球算法联赛 | **标准化硬件 + 差异化运控**；BO3 自主格斗；与 REK **遥操作** 路线对照；见 [URKL 实体](../entities/urkl.md) |
| Humanoid Surgeon（UCSD 2026） | Surgie 人形 | 手术控制台 + 立体视觉 | in vivo 猪模型 | **腹腔镜胆囊切除**；通用器械 + RCM 遥操作；Nature 活体可行性；见 [论文实体](../entities/paper-humanoid-surgeon-in-vivo-laparoscopy.md) |

## 操作员侧可穿戴力反馈（研究参考）

机器人端触觉传感（F/T、GelSight 等）解决的是 **策略输入**；操作员在 VR/遥操作中还需感知 **虚拟或远程物体的尺寸与刚度**。[HapMorph](../entities/paper-hapmorph-pneumatic-haptic-render.md)（arXiv:2509.05433，SSSA）用 **拮抗式织物气动执行器（AFPA）** 在 **21 g** 可穿戴形态下，经双腔压力解耦渲染 **50–104 mm** 尺寸与 **~0.12–4.7 N/mm** 刚度；10 人感知实验对 9 种尺寸×刚度组合的识别准确率达 **89.4%**。与 [UME-EXO](../entities/paper-ume-exo.md) 的外骨骼力矩反馈、[数据手套 vs 视觉遥操作](../comparisons/data-gloves-vs-vision-teleop.md) 中的 **运动采集** 通道正交——HapMorph 属于 **力触觉显示** 而非示教采集。当前原型仍依赖外部气源，尚非可直接部署的遥操作商品件。

## 跨形态实时 I/O（工程参考）

**[RIO（Robot I/O）](../entities/robot-io-rio.md)**（CMU / Bosch 等，RSS 2026 接收）把 Spacemouse、手柄、键盘、GELLO、手机与 VR 等遥操作入口，与多相机、机械臂/人形控制、数据记录和 **异步 VLA 推理** 收拢到同一套 **Node + 可切换中间件** 抽象里：换设备组合以 **station 配置** 为主，主循环尽量保持通用。适合作为「多接口采集 + 低延迟闭环」的对照阅读，与上表中以单一系统命名的经典遥操作栈互补。

在 **Linux 工作站** 上，**USB 有线 Xbox 手柄** 通常由内核 **[xpad](../entities/xpad.md)** 驱动暴露为 `/dev/input/js*` 与 evdev 节点，再被 pygame、ROS `joy` 或 RIO 手柄 Node 读取；**蓝牙配对** 的 Xbox 手柄则走通用 HID，不经过 xpad。部署前需分清连接方式，避免「模块已加载但无输入」的误判。

## 全身人形：视频 / VR 条件 + 低层 tracking（工程参考）

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
|------|------|
| 任务成功率 | N 次尝试中完成任务的比例 |
| 完成时间 | 完成任务的平均时长（越短越好） |
| 操作员上手时间 | 新操作员达到可接受质量所需的训练时间 |
| 端到端延迟 | 操作员输入到机器人执行的延迟 |
| 示范数据效率 | 用多少条示范可以训练出成功率 ≥ X% 的策略 |

## 参考来源

- **ingest 档案：** [sources/sites/engineai-urkl.md](../../sources/sites/engineai-urkl.md) — URKL：EngineAI 统一 T800 自主算法格斗联赛
- **ingest 档案：** [sources/sites/rek-com.md](../../sources/sites/rek-com.md) — REK：VR 驱动 G1 全接触格斗联赛与租赁
- **ingest 档案：** [sources/sites/hiw-500-dataset.md](../../sources/sites/hiw-500-dataset.md) — HIW-500：东南亚 12 家庭 G1 全身遥操作 **500+ h / 23K+ 集** 开源数据集
- **ingest 档案：** [sources/blogs/mimicrobotics_m1_u1_full_stack.md](../../sources/blogs/mimicrobotics_m1_u1_full_stack.md) — mimic U1 固定运动学外骨骼 + M1 全栈灵巧平台（2026-07）
- **ingest 档案：** [sources/papers/teleoperation.md](../../sources/papers/teleoperation.md) — ALOHA / OmniH2O / UMI / BifrostUMI / AnyTeleop
- **ingest 档案：** [sources/papers/ume_exo_arxiv_2606_14218.md](../../sources/papers/ume_exo_arxiv_2606_14218.md)、[sources/sites/ume-exo-project.md](../../sources/sites/ume-exo-project.md) — UME-EXO：外骨骼实时力矩反馈 + 全身臂形采集 + ACT 柔顺策略（arXiv:2606.14218）
- **ingest 档案：** [sources/papers/bifrost_umi_arxiv_2605_03452.md](../../sources/papers/bifrost_umi_arxiv_2605_03452.md) — BifrostUMI：无机器人采集 + SKR + G1 全身 visuomotor（arXiv:2605.03452）
- **ingest 档案：** [sources/papers/halomi_arxiv_2606_18772.md](../../sources/papers/halomi_arxiv_2606_18772.md) — HALOMI：egocentric 无机器人示范 + 主动颈 G1 loco-manipulation（arXiv:2606.18772）
- **ingest 档案：** [sources/sites/twist2-project.md](../../sources/sites/twist2-project.md)、[sources/repos/twist2.md](../../sources/repos/twist2.md) — TWIST2 项目页 + 开源仓库（arXiv:2511.02832）
- **ingest 档案：** [sources/papers/clot_arxiv_2602_15060.md](../../sources/papers/clot_arxiv_2602_15060.md)、[sources/sites/clot-project.md](../../sources/sites/clot-project.md) — CLOT 闭环全局遥操作（arXiv:2602.15060；官方页非 clot.github.io）
- **ingest 档案：** [sources/papers/diffusion_and_gen.md](../../sources/papers/diffusion_and_gen.md) — ACT / Diffusion Policy（遥操作数据的下游学习方法）
- **ingest 档案：** [sources/papers/humanoid_touch_dream.md](../../sources/papers/humanoid_touch_dream.md) — HTD 的 VR 全身遥操作、LBC 和触觉同步采集系统
- **ingest 档案：** [sources/repos/robot-io-rio.md](../../sources/repos/robot-io-rio.md) — RIO 的多设备遥操作与实时 Node 管线（arXiv:2605.11564）
- **ingest 档案：** [sources/repos/xpad.md](../../sources/repos/xpad.md) — Linux USB Xbox 手柄内核驱动（paroj/xpad）
- **ingest 档案：** [sources/papers/bfm_humanoid_arxiv_2509_13780.md](../../sources/papers/bfm_humanoid_arxiv_2509_13780.md) — BFM：CVAE + 掩码在线蒸馏，让单一策略统一覆盖人形跟踪 / VR 遥操作 / locomotion 多接口
- **ingest 档案：** [sources/papers/pilot_arxiv_2601_17440.md](../../sources/papers/pilot_arxiv_2601_17440.md) — PILOT：VR 遥操作 + 感知 MoE 全身 LLC（arXiv:2601.17440）
- **ingest 档案：** [sources/papers/cwi_arxiv_2606_27676.md](../../sources/papers/cwi_arxiv_2606_27676.md) — CWI：Meta Quest 双手 + 速度/身高全身 loco-manipulation（arXiv:2606.27676）
- **ingest 档案：** [sources/papers/rove_arxiv_2606_17011.md](../../sources/papers/rove_arxiv_2606_17011.md)、[sources/sites/xpeng-robotics-rove.md](../../sources/sites/xpeng-robotics-rove.md) — ROVE：人形 VLA 部署干预采集与次优接管轨迹的 RL 后训练（arXiv:2606.17011）
- **ingest 档案：** [sources/papers/hapmorph_arxiv_2509_05433.md](../../sources/papers/hapmorph_arxiv_2509_05433.md) — HapMorph：AFPA 可穿戴气动框架解耦渲染尺寸与刚度（arXiv:2509.05433）
- **ingest 档案：** [sources/sites/telegate-project.md](../../sources/sites/telegate-project.md)、[sources/papers/telegate_arxiv_2602_09628.md](../../sources/papers/telegate_arxiv_2602_09628.md) — TeleGate：门控专家 + VAE 运动先验全身遥操作（RSS 2026，arXiv:2602.09628）
- **ingest 档案：** [sources/sites/heft-project.md](../../sources/sites/heft-project.md)、[sources/papers/heft_arxiv_2607_02332.md](../../sources/papers/heft_arxiv_2607_02332.md)、[sources/repos/axellwppr_motion_tracking.md](../../sources/repos/axellwppr_motion_tracking.md) — HEFT：PMG + WPC 重载全尺寸人形 VR 遥操作（arXiv:2607.02332）
- **ingest 档案：** [sources/papers/humanoidarena_arxiv_2606_17833.md](../../sources/papers/humanoidarena_arxiv_2606_17833.md) — HumanoidArena：PICO egocentric 采集 + GMR → 双 GMT 分层 benchmark（arXiv:2606.17833）
- **ingest 档案：** [sources/papers/humanoid_surgeon_nature_2026.md](../../sources/papers/humanoid_surgeon_nature_2026.md) — Humanoid Surgeon：人形腹腔镜 in vivo 可行性（Nature 2026；[项目页](https://humanoid-surgeon.github.io/)）
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (RSS 2023) — ALOHA
- He et al., *OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation* (2024)
- [机器人论文阅读笔记：OmniH2O](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/OmniH2O_Universal_Whole-Body_Teleoperation/OmniH2O_Universal_Whole-Body_Teleoperation.html)

## 关联页面

- [Loco-Manipulation](./loco-manipulation.md) — 遥操作在移动操作中的应用
- [Motion Retargeting](../concepts/motion-retargeting.md) — 人类动作到机器人动作的映射
- [Imitation Learning](../methods/imitation-learning.md) — 遥操作数据的学习方法
- [Humanoid Transformer with Touch Dreaming](../methods/humanoid-transformer-touch-dreaming.md) — 使用触觉遥操作数据训练接触感知策略
- [Diffusion Policy](../methods/diffusion-policy.md) — 遥操作数据训练的扩散策略
- [UME-EXO（论文实体）](../entities/paper-ume-exo.md) — 外骨骼力矩反馈 + 全身臂形示教 → ACT 主动柔顺策略
- [BifrostUMI（论文实体）](../entities/paper-bifrost-umi.md) — 无机器人示范 → 人形全身扩散策略 + SKR
- [mimic wearable U1](../entities/mimic-wearable-u1.md) — UMI 式无真机灵巧示范；与 M1 1:1 运动学/传感
- [HALOMI（论文实体）](../entities/paper-halomi-humanoid-loco-manipulation.md) — 无机器人 egocentric 头手示范 → π₀.₅ + 流形约束 WBC
- [TWIST2（论文实体）](../entities/paper-twist2.md) — 便携真机全身遥操作 → visuomotor 自主
- [CLOT（论文实体）](../entities/paper-amp-survey-16-clot.md) — 闭环全局位姿的长时程全身遥操作
- [LEGS（论文实体）](../entities/paper-legs-embodied-gaussian-splatting-vla.md) — 无遥操作合成 loco-manip 演示 vs teleop 数据成本（arXiv:2606.01458）
- [Manipulation](./manipulation.md) — 操作任务整体视角
- [Query：操作演示数据采集指南](../queries/demo-data-collection-guide.md) — 遥操作采集数据的实操指南
- [RIO（Robot I/O）](../entities/robot-io-rio.md) — 跨形态 Node 化遥操作与异步策略推理栈
- [xpad](../entities/xpad.md) — Linux USB Xbox 手柄内核驱动与 joystick/evdev 接口
- [Isaac Teleop](../entities/isaac-teleop.md) — NVIDIA Isaac Lab / Sim XR 遥操作与示范录制
- [Isaac Lab](../entities/isaac-lab.md) — 集成宿主与内置遥操作环境
- [PILOT（论文实体）](../entities/paper-pilot-perceptive-loco-manipulation.md) — VR 长程 loco-manipulation 与非结构化地形底层控制
- [CWI（论文实体）](../entities/paper-cwi-composite-humanoid-whole-body-imitation.md) — Quest VR 双手接口 + 复合全身模仿 loco-manipulation（arXiv:2606.27676）
- [ROVE（论文实体）](../entities/paper-rove-humanoid-vla-intervention.md) — 人形 VLA 近失败 MoCap 接管与三阶段干预标注（arXiv:2606.17011）
- [REK](../entities/rek.md) — VR 格斗体育：竞技向全身 teleop 极端场景
- [HapMorph（论文实体）](../entities/paper-hapmorph-pneumatic-haptic-render.md) — 操作员侧可穿戴尺寸+刚度力触觉渲染（arXiv:2509.05433）
- [TeleGate（论文实体）](../entities/paper-telegate.md) — 惯性动捕 + 门控冻结专家 + VAE 预判；2.5 h 高动态全身遥操作（RSS 2026，arXiv:2602.09628）
- [HEFT（论文实体）](../entities/paper-heft.md) — 嘈杂 raw VR + WPC 双手负载；全尺寸 L7 **24 kg** 重载遥操作（arXiv:2607.02332）
- [HumanoidArena（论文实体）](../entities/paper-humanoidarena.md) — PICO egocentric 采集管线与 TWIST2/SONIC 双 GMT 分层 benchmark（arXiv:2606.17833）
- [Humanoid Surgeon（论文实体）](../entities/paper-humanoid-surgeon-in-vivo-laparoscopy.md) — 通用人形腹腔镜遥操作 in vivo 猪模型验证（Nature 2026）
- [motion_tracking（代码实体）](../entities/axellwppr-motion-tracking.md) — HEFT 官方 mjlab 训练与 sim2real 检查点

## 推荐继续阅读

- [HEFT 项目页](https://heft.axell.top/)
- [HEFT（论文实体）](../entities/paper-heft.md) — PMG / WPC 与 G1+L7 评测归纳
- [TeleGate 项目页](https://anywitresearch.github.io/TeleGate/)
- [TeleGate（论文实体）](../entities/paper-telegate.md) — 门控专家选择与运动先验归纳
- [机器人论文阅读笔记：TeleGate](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/TeleGate__Whole-Body_Humanoid_Teleoperation_via_Gated_Expert_Selection_with_Motion_Prior/TeleGate__Whole-Body_Humanoid_Teleoperation_via_Gated_Expert_Selection_with_Motion_Prior.html)
- [机器人论文阅读笔记：SEW-Mimic](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/SEW-Mimic__Closed-Form_Geometric_Retargeting_Solver_for_Upper_Body_Humanoid_Teleoperation/SEW-Mimic__Closed-Form_Geometric_Retargeting_Solver_for_Upper_Body_Humanoid_Teleoperation.html)
- [机器人论文阅读笔记：Learning Adaptive Neural Teleoperation for Humanoid Robots](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Learning_Adaptive_Neural_Teleoperation_for_Humanoid_Robots/Learning_Adaptive_Neural_Teleoperation_for_Humanoid_Robots.html)
- [机器人论文阅读笔记：HumanPlus](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/HumanPlus_Humanoid_Shadowing_and_Imitation_from_Humans/HumanPlus_Humanoid_Shadowing_and_Imitation_from_Humans.html)
- [机器人论文阅读笔记：ExtremControl](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/ExtremControl__Low-Latency_Humanoid_Teleoperation_with_Direct_Extremity_Control/ExtremControl__Low-Latency_Humanoid_Teleoperation_with_Direct_Extremity_Control.html)
- [机器人论文阅读笔记：HOMIE Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/HOMIE_Humanoid_Loco-Manipulation_with_Isomorphic_Exoskeleton_Cockpit/HOMIE_Humanoid_Loco-Manipulation_with_Isomorphic_Exoskeleton_Cockpit.html)
- [ALOHA 项目主页](https://mobile-aloha.github.io/) — Stanford 双臂遥操作系统
- [OmniH2O 论文](https://arxiv.org/abs/2406.08858) — 人形全身遥操作
- [ACT 论文](https://arxiv.org/abs/2304.13705) — 遥操作数据 + 动作块学习
