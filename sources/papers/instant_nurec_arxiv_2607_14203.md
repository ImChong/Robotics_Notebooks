# instant_nurec_arxiv_2607_14203

> 来源归档（ingest）

- **标题：** Instant NuRec: Feed-Forward 3D Gaussian Reconstruction for Driving Scene Simulation
- **类型：** paper
- **来源：** arXiv:2607.14203（2026）；项目页 <https://research.nvidia.com/labs/sil/projects/instant-nurec/>
- **代码：** <https://github.com/NVIDIA/instant-nurec> — 归档见 [sources/repos/nvidia-instant-nurec.md](../repos/nvidia-instant-nurec.md)
- **机构：** 英伟达（NVIDIA）Spatial Intelligence Lab（SIL）
- **入库日期：** 2026-09-05
- **一句话说明：** 把标定多相机驾驶片段在 **单次前向** 里变成可仿真的分层 **3DGS** 世界（静态 / 动态 / 天空 + 每相机 ISP）；10–20 s 片段约 **1.5 s** 重建，Waymo PSNR 比最强对照高 **2.01 dB**，并接入 [NuRec](https://docs.nvidia.com/nurec/) 与 AlpaSim 闭环评测。

## 核心论文摘录（MVP）

### 1) 问题：逐场景优化无法吃下车队日志（Abstract / §1）

- **链接：** <https://arxiv.org/abs/2607.14203>
- **核心贡献：** 闭环自动驾驶仿真需要 (i) 照片级渲染、(ii) 静态背景与可编辑动态体分解、(iii) 自由重姿态 / 重定时。OmniRe 等逐场景 3DGS 能忠实复现日志，但每段要 **数十分钟到数小时** 且依赖 LiDAR、拟合位姿、语义与 cuboid。Instant NuRec 把这段代价摊到一次前向：输入 \(V\times T\) 标定多相机图（\(V\in\{1,3,5\}\)，\(T\) 常为 8/12/18 帧、2–4 Hz），输出静态层 \(\mathcal{G}^s\)、带三段折线轨迹的动态层 \(\mathcal{G}^d\)、天空 cubemap 与每相机 \(3\times4\) ISP 仿射；USDZ / NuRec 可消费，AlpaSim 可闭环。
- **对 wiki 的映射：**
  - [Instant NuRec 论文实体](../../wiki/entities/paper-instant-nurec.md)
  - [NVIDIA Omniverse NuRec](../../wiki/entities/nvidia-nurec.md)
  - [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)

### 2) 共享交替注意力编码器 + 多解码头（§3）

- **链接：** <https://arxiv.org/html/2607.14203#S3>
- **核心贡献：**
  - **编码器：** 沿 Depth-Anything-3 / DINOv2，\(14\times14\) patch + 位姿/FoV 正弦编码的 class token；层内交替 **图内自注意** 与 **跨图自注意**。
  - **上下文头：** 三路 DPT 预测深度、法向、四类语义（road / movable / sky / ego-car）。
  - **3DGS 头：** 查询点从深度抬升，**不绑定「每像素一个高斯」**；Dense（每像素）vs Selective（stride + 道路分支，约 **4×** 少高斯、质量接近）。
  - **运动头：** 对同一查询预测前/后位移，拼三段折线；语义 mask 后只有 movable 进动态层。
  - **天空 / ISP：** cubemap 按世界射线 cross-attend；每相机仿射吸收白平衡 / gamma / 暗角。渲染走 **3DGUT**（非针孔、畸变相机）。
  - **长片段：** 重叠 chunk + 静态层 **视锥所有权剪枝**（式 1）；动态层因时间窗短不剪。
- **对 wiki 的映射：**
  - [Instant NuRec 论文实体](../../wiki/entities/paper-instant-nurec.md)
  - [GS-Playground](../../wiki/entities/gs-playground.md) — 对照：仿真侧批量 3DGS 渲染 vs 日志→可仿真世界

### 3) 三阶段训练与内部数据（§4）

- **链接：** <https://arxiv.org/html/2607.14203#S4>
- **核心贡献：** \(\mathcal{L}=\mathcal{L}_{\text{context}}+\mathcal{L}_{\text{motion}}+\mathcal{L}_{\text{render}}\)。Stage 1 复现 Depth-Anything-3 预训练；Stage 2 在约 **4 万** 内部驾驶片段上开天空 / 深度 / 运动头（LiDAR + 自动标注监督）；Stage 3 **冻结编码器**，只训 GS + ISP，用 3DGUT 可微渲染（\(L_1\)+LPIPS）。约 **6 天 / 8 节点**。一阶段联合训更慢且最终更差。
- **对 wiki 的映射：**
  - [Instant NuRec 论文实体](../../wiki/entities/paper-instant-nurec.md)

### 4) Waymo / 内部重建 / 闭环策略排序（§5）

- **链接：** <https://arxiv.org/html/2607.14203#S5>
- **核心贡献：**
  - **Waymo val（STORM 协议，240×160）：** 全图 PSNR **28.26** / SSIM **0.859**，对照 DGGT **26.25 / 0.805**（+**2.01 dB**）；动态区 PSNR **24.93** vs 21.76。深度 AbsRel **0.076**、\(\delta_1\) **0.937**。
  - **内部对照逐场景 NuRec：** Dense PSNR **29.93** vs NuRec **34.38**；检测 precision/recall **0.955/0.940** vs **0.970/0.955**；时间 **~1.5 s vs ~75 min**（三个数量级）。Selective 高斯约 **1/3**，质量略降。
  - **AlpaSim 闭环：** 140 场景 × 20 s × 6 trial；VaVAM / Alpamayo R1 / A-1.5（1/2/4 相机）。Instant NuRec 与逐场景 NuRec **策略排序相同**——选型决策可换便宜重建。
  - **LiDAR 扩展：** 轻量 range-map 分支；内部 Chamfer **0.286** vs NuRec **0.204**；约 **20 s vs 75 min**（~225×）。
- **对 wiki 的映射：**
  - [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

### 5) 开源边界与局限（§6 / 仓库 README）

- **链接：** <https://github.com/NVIDIA/instant-nurec>
- **核心贡献：** 论文写代码在 `nvidia/instant-nurec`。官方仓 **部分开源**：可跑 **静态 3DGS PLY + `.sky.npz` cubemap sidecar + 可选预览渲染**；README 写明动态层等研究模型输出 **不在本静态导出路径**。权重 Hugging Face `nvidia/instant-nurec`；演示 clip 在门控数据集 `nvidia/PhysicalAI-Autonomous-Vehicles-NCore`。PLY 可作 [NuRec](https://docs.nvidia.com/nurec/nurec/reconstruct-av-scene.html) 逐场景精修初始化。局限：高斯预算 vs 细结构；训练相机架外需微调；三段折线抓不住亚秒非刚体；未做流式长日志。
- **对 wiki 的映射：**
  - [nvidia-instant-nurec 仓归档](../repos/nvidia-instant-nurec.md)
  - [NuRec 文档归档](../sites/nvidia-nurec-docs.md)

## 其他公开资料（非 PDF 正文）

- **项目页（画廊、赛道、AlpaSim 闭环、量化条）：** <https://research.nvidia.com/labs/sil/projects/instant-nurec/> — 归档见 [nvidia-research-instant-nurec.md](../sites/nvidia-research-instant-nurec.md)
- **官方仓 + CLI：** <https://github.com/NVIDIA/instant-nurec> — 归档见 [nvidia-instant-nurec.md](../repos/nvidia-instant-nurec.md)
- **产品文档 26.04（Instant NuRec 为推荐初始化）：** <https://docs.nvidia.com/nurec/> — 归档见 [nvidia-nurec-docs.md](../sites/nvidia-nurec-docs.md)
- **权重：** <https://huggingface.co/nvidia/instant-nurec>
- **NCore 演示数据（门控）：** <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore>

## 当前提炼状态

- [x] 论文摘要与核心方法摘录（≥3 条）
- [x] 步骤 2.5 开源边界（部分开源：静态推理可跑）
- [x] wiki 页面映射与 docs / 项目页 / 仓互链
