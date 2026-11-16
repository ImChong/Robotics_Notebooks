# Interactive World Simulation with Cosmos for Physical AI（NVIDIA 技术报告）

> 一手资料归档；与 [Cosmos 3 基座](../../wiki/entities/cosmos-3.md) 的技术报告不是同一份。以下数值为作者报告，尚未独立复现。

## 元数据

- **题目：** Interactive World Simulation with Cosmos for Physical AI
- **机构：** NVIDIA（报告署名 NVIDIA；完整贡献者名单见附录 A）
- **报告版本日期：** **2026-11-04**（当前官方 PDF 首页明确写有 `Date: 2026-11-4`；原归档 `2026-10-21` 与当前版本不符）
- **技术报告 PDF：** https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/technical-report.pdf
- **arXiv 记录：** https://arxiv.org/abs/2611.06627（同题；本次未逐页比较两个 PDF 的异同）
- **项目页：** https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/
- **官方源码：** https://github.com/NVIDIA/cosmos（文档/入口）；https://github.com/NVIDIA/cosmos-framework（训练与推理实现）
- **权重合集：** https://huggingface.co/collections/nvidia/cosmos3-examples
- **首次入库：** 2026-10-22
- **复核日期：** 2026-11-12
- **核对范围：** 按论文目录通读 arXiv:2611.06627v1 完整可读取 PDF 文本及表格：封面/目录、§1 研究目标、§2 MoT 架构与时序因果 attention、§3 四阶段后训练、§4 KV 缓存与 streaming 推理、§5 双臂建模/数据/训练/效率/评测、§6 空间控制生成与 Isaac Lab 闭环评测、§7 多相机-LiDAR 驱动仿真/实验、§8 相关工作、§9 结论、附录 A 贡献者/B 双臂细节与消融/C Transfer 训练性能与失败案例/D Auto 架构-数据-推理优化及消融、参考文献；对照原表、图注与 NVIDIA 官方项目页/模型卡。**取材为论文 PDF 的完整可读文本与图注，未将截图接口无法取得的高分辨率图片视作逐像素核验，也未逐项复现 GPU 实验。**
- **原文文本入口：** https://arxiv.cnoctave.top/pdf/2611.06627（arXiv v1 的 PDF 文本可读镜像，用于补查正文与附录；论文权威版本仍以 NVIDIA 官方 PDF 与 arXiv 为准）
- **通读：** 已完成

## 研究目标与模型身份

将 Cosmos 3 的视频生成先验变成随动作、场景控制更新的交互式仿真观测，供策略/用户或外部物理仿真器驱动。技术报告按三个任务族组织（Sim-Bimanual、Sim-Transfer、Sim-Auto）；**实际发布四个不同 checkpoint**，Auto 族含离线双向的 **Transfer-Auto** 和因果蒸馏的 **Sim-Auto**。

| checkpoint | 作用与边界 |
|---|---|
| **Cosmos3-Nano-Transfer-Auto** | 双向的离线多视角驾驶数据生成；最多 11 路相机 + LiDAR、WSM 场景控制；适配新传感器的起点 |
| **Cosmos3-Nano-Sim-Auto** | Transfer-Auto 的因果蒸馏流式变体，按新 WSM 控制逐块生成；闭环应用仍需自备仿真状态更新和 WSM 控制渲染 |
| **Cosmos3-Nano-Sim-Transfer** | 外部物理仿真器供给深度/分割/边缘/模糊等空间控制，世界模型只生成 RGB 外观；不替代物理求解器 |
| **Cosmos3-Nano-Sim-Bimanual** | 以相机/机器人动作与历史观测预测下一段视频；**预测视觉后果，不直接输出动作策略** |

四个权重都是 Cosmos3-Nano 后训练示例，模型卡称 16B trainable parameters，许可为 OpenMDW 1.1，使用条件见模型卡/许可正文；“公开权重”不意味着所有原始训练数据公开。

## 共享后训练机制（依据报告 §2–3）

1. **时序因果块：** 当前块的干净动作/视频控制 token 可以与当前噪声视觉 token 联合注意；历史块可见，未来视觉块不可见（Fig. 3）；有显式时间对齐、历史 K/V 缓存。
2. **Stage 0 — Bidirectional Adaptation：** 以全时域可见的基础模型适配控制接口与领域视觉（rectified-flow，Eq. 4）；离线 Transfer-Auto 主要对应此种双向生成路径。
3. **Stage 1 — Causal Teacher Forcing：** 参考历史作为干净条件，以分块因果掩码预测当前目标（Eq. 5）；控制和初始图不计生成目标损失。
4. **Stage 2 — TF-dCM：** 冻结因果教师、以少步一致性蒸馏训练学生；压缩单块去噪预算。
5. **Stage 3 — SF-DMD：** 学生以**自身生成的历史**滚动，冻结的 Stage 0 双向教师与可训练辅助 score 模型形成分布匹配信号；通过截断反传等技巧控制显存（Eq. 10–19）。

报告中首个视觉时间步可单独表示初帧；之后每个视觉 latent 时间步对应 **4 个 RGB 帧**（§3.2）。不要把视频帧率直接等同于每步策略频率。

## 论文原表中的运行性能（非模型卡不同后端的数字）

| 场景与硬件 | 论文表 | 生成配置 | 报告吞吐 / 延迟 |
|---|---|---|---|
| Sim-Bimanual，**1× GB300** | §5.4，Tab. 3 | 2 latent frames = 8 RGB 帧/块；4 去噪步；compiled + CUDA graphs / TAEHV Super；稳态录像动作 | **48.1 FPS**，p50/p95 单块 **166/169 ms**；测量排除 10 个预热块 |
| Sim-Auto，**1× GB200** | §7.4，Tab. 14 | 1 路 RGB，832×480，12 帧/块，4 步 | **32.6 FPS**，p50 **364 ms** |
| Sim-Auto，**16× GB200** | §7.4，Tab. 14 | 7 路 RGB + LiDAR，832×480，12 帧/块，4 步 | **16.7 FPS**，p50 **701 ms** |
| Sim-Auto，**32× GB200** | §7.4，Tab. 14 | 7 路 RGB（无 LiDAR），同尺寸 | **23.2 FPS**，p50 **511 ms** |

**不要把“实时”当成默认全模式 30 FPS。** 以上 7 路/多模态吞吐低于 30 FPS 视频目标，且测试使用已预取的 WSM 控制；接收新控制还有渲染/准备开销（§7.4）。Bimanual 的 48.1 FPS 也只对应表内特定单卡/块长/解码器/优化配置。Sim-Transfer 项目页另列单 GB200、640×640 **15.85 FPS / 240 ms**，但不是上述论文 Tab. 14 的实验，不能混作同一基准。

## 质量与闭环局限（报告 §7.5）

- **7 路 720p Driving-Sim-Eval（91 组）：** Transfer-T2V Frame FID：Transfer-Auto **45.5**、Sim-Auto **48.6**、Cosmos-Transfer2.5 基线 **61.9**；Sim-Auto 的 FVD(I3D) **480**，略优于 Transfer-Auto **488**，不能概括为流式版全部质量下降（Tab. 15）。
- **Traffic-Scenario-Eval（319 组）：** 车辆 F1：Transfer-Auto **0.908**、Sim-Auto **0.853**、Cosmos-Transfer2.5 **0.741**；行人 F1：**0.386 / 0.389 / 0.137**（Tab. 17）。
- **LiDAR 检测 F1@3m：** Transfer-Auto **0.237**、Sim-Auto **0.188**、LidarVerse（116 sweeps）**0.200**（Tab. 16）；流式版不保证所有传感器质量超过前代或非因果教师。
- **1 小时生成展示：** Fig. 27 用 **循环 WSM 控制/文本提示** 和滚动 K/V 缓存在两个场景中生成 1 小时；选定画质与一致性指标无持续退化，但报告也承认物体外观与物理合理性会在长期 rollout 中劣化。**不等于真实策略驾驶 1 小时已验证安全闭环。**
- 真正用于策略训练/评估仍需对接真实控制接口、传感器同步、仿真物理状态，以及在独立闭环任务上量化效用；发布的推理示例与作者自报画质指标并不自动证明实际任务成功率或节省实车测试成本。

## §4–6、§8–9 与附录 B–D 通读补记（2026-11-12）

### §4 Streaming Inference 与 §5 Bimanual

- §4 将跨块 **K/V 缓存、窗口淘汰与 attention sinks** 结合：自回归视频只保留可见历史，避免持续扩增缓存；不同模型的 chunk 步长、解码器、编译、CUDA Graphs 共同决定吞吐和首块响应时间，不能把稳态 FPS 当成端到端交互时延。
- §5 的双臂动作使用统一对齐的动作表示（正文提到 59 维示例，模型卡推理可补零到 64），该模型学习**动作条件视觉前向动力学**而不是机器人控制器；Tab. 3 的 48.1 FPS 是固定硬件、特定编译与分块配置的稳定状态，测试动作来自预先录制序列，并非证明策略闭环精度。
- 附录 B.1–B.5 进一步列出字幕/数据构造、训练设置、快速解码器对照、视频及几何评测与消融；质量与相机位姿指标并非总能同时优于比较模型。

### §6 Sim-Transfer：仿真器闭环的定量证据

- 外部 Isaac Lab 物理仿真器维护状态，用几何/分割/深度控制引导世界模型渲染；生成 RGB 再作为策略感知输入，**模型不负责机器人动力学**。
- 报告 §6.5 与附录 C.4 的 **GR00T N1.7 苹果抓放**实验：在一组经校准的 **32 个初始状态**中，**15k-step 策略**使用 RTX 原生渲染成功 **22/32**，使用 Cosmos 生成式渲染成功 **27/32**；而 **20k-step 策略**在 RTX **24/32**、Cosmos **21/32**，另有红色托盘配置 Cosmos **19/32**、RTX **22/32**。不同策略与条件存在反转，不能用单组 27/32 证明所有闭环任务收益。
- 附录 C 还报告了原始/合格视频样本筛选、运行剖析以及阶段对照；C.5 的典型失败包括**时间涂抹、物体鬼影、外观漂移、几何变形、空间控制不一致**。稳态 15.85 FPS（项目页优化配置）不等于首块或整套策略循环都达到实时。
- 这部分是**报告确有的策略闭环证据**，应与 §7 的 Auto **循环 WSM 控制的一小时离线生成展示**分开：后者不能直接证明车端控制闭环。

### §7 Auto 数据与补充消融

- 相机 RGB 和 LiDAR 的采样率/时序不同：报告根据相机视频与 LiDAR 扫描的时间对齐进行联合编码，并通过相机 WSM 和 HD-map range-map 控制维持多视角场景几何；新装备并非仅替换传感器路径即可无需后训练。
- §7 及附录 D.1–D.6 包括多模态注意力排布、子集摄像头采样、短/长序列训练、分布式推理和缓存窗口消融；上文 Tab. 14 的多视角 FPS 必须带相机数量、LiDAR 有无、GPU 数、分辨率与是否预取 WSM 的限定。
- 天气/光照变化的 prompt 生成只说明条件控制可影响 RGB 和 LiDAR 观测，不能自动推出真实传感器在雨雾环境中的量测误差模型已经经过校准。

### §8–9、附录 A 与整体研究边界

- §8 将该框架放在可控世界模型、因果视频生成/少步蒸馏、流式推理三条工作线中；§9 将三种方向作为**共享后训练框架的示例**，并未声称一个 checkpoint 同时解决全部仿真任务。
- 附录 A 为作者贡献与致谢，不是额外模型架构。附录 B–D 各自的补充数值、训练假设、数据筛选与失败案例必须回到对应实验配置，不可混为通用的“模型成功率”。

## 开源状态与映射

- **开源边界：** NVIDIA/cosmos 提供文档与入口，NVIDIA/cosmos-framework 提供框架和 cookbook/skill；Hugging Face 提供 4 个 checkpoint。**可运行框架与权重已公开**，但完整训练数据、特定装备与端到端闭环仿真需要用户自行准备，实际许可和运行限制以各官方资源为准。
- **模型卡记录：** [四张 checkpoint 官方模型卡归档](../sites/hf_cosmos3_nano_sim_checkpoints_model_cards.md)
- **官方博文逐节记录：** [Hugging Face NVIDIA 博文](../blogs/nvidia_cosmos_interactive_world_simulation_hf_blog.md)
- **项目页复核：** [官方演示/性能/源码导航归档](../sites/nvidia_cosmos_interactive_world_simulation.md)
- **唯一 wiki 实体：** [Cosmos Interactive World Simulation](../../wiki/entities/paper-cosmos-interactive-world-simulation.md)

> **阅读与复现边界：** 已按目录完成论文正文 §1–9 与附录 A–D 的**可读取全文文本/图注/表格**通读，标记 `通读：已完成`。PDF 的部分矢量图或照片**无法以截图单独核对其细节**，未执行训练、推理、真实车辆/机器人的独立复现实验；通读标记不代表实验复现或所有图像的视觉审计。
