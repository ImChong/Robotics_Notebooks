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
- **核对范围：** 在线打开并读取官方报告的封面、目录、§1–3 的架构和时序因果掩码/四阶段配方（Fig. 1、3、4；Eq. 4、5、10、19）、§5.4–5.5 双臂模型速度表（Tab. 3）、§7.4–7.5 驾驶多视角性能和质量（Tab. 14、15、17、18，Fig. 27）及其结论，并对照官方项目页和 NVIDIA 署名的 Hugging Face 博文。**§4、§6、§8–9 以及附录 A–D 尚未逐页完成检查**；网页 PDF 截图接口因文件过大返回错误，未声称目视核查所有图片。
- **通读：** 未完成

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

## 开源状态与映射

- **开源边界：** NVIDIA/cosmos 提供文档与入口，NVIDIA/cosmos-framework 提供框架和 cookbook/skill；Hugging Face 提供 4 个 checkpoint。**可运行框架与权重已公开**，但完整训练数据、特定装备与端到端闭环仿真需要用户自行准备，实际许可和运行限制以各官方资源为准。
- **模型卡记录：** [四张 checkpoint 官方模型卡归档](../sites/hf_cosmos3_nano_sim_checkpoints_model_cards.md)
- **官方博文逐节记录：** [Hugging Face NVIDIA 博文](../blogs/nvidia_cosmos_interactive_world_simulation_hf_blog.md)
- **项目页复核：** [官方演示/性能/源码导航归档](../sites/nvidia_cosmos_interactive_world_simulation.md)
- **唯一 wiki 实体：** [Cosmos Interactive World Simulation](../../wiki/entities/paper-cosmos-interactive-world-simulation.md)

> **阅读边界：** 此次完成的是报告指定方法节与实验表的原文核验，以及项目页逐项核验；**未完成整份报告及附录逐页通读**。本归档明确标为“通读：未完成”，不得因此给 wiki 页补写 `original: 已通读论文`。
