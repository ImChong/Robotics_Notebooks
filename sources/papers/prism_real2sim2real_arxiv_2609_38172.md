# prism_real2sim2real_arxiv_2609_38172

> 来源归档（ingest）

- **标题：** Counterfactual Video Generation Enables Scalable Humanoid Loco-Manipulation（PRISM）
- **类型：** paper
- **会议：** CoRL 2026
- **arXiv：** <https://arxiv.org/abs/2609.38172>
- **项目页：** <https://prism-real2sim2real.github.io/>
- **代码（项目页）：** <https://github.com/amazon-far/PRISM-Real2Sim2Real>（截至 **2026-09-30** 匿名 GET 返回 **404**，待公开后复核）
- **入库日期：** 2026-09-30
- **一句话说明：** 用 **V2V counterfactual 视频** 把 4 条真人搬箱种子扩成 256 条交互 clip，经 **接触锚定 Real2Sim + 重定向** 训统一 **机载深度 + 摇杆** G1 策略，真机零样本 pick–carry–drop 多类物体。

## 核心论文摘录（MVP）

### 1) 问题与 counterfactual V2V 数据范式（Abstract / §3.1）

- **链接：** <https://arxiv.org/abs/2609.38172>
- **核心贡献：** 互联网/自采 **高质量全身人–物交互视频** 难规模化；PRISM 用 **video-to-video** 在保留背景/光照/机位的前提下替换 manipulated object（box/bin/barrel/ball），生成「本可发生但未发生」的 **counterfactual** 交互，且物体变化会连带 **人的行为策略**（弯腰、手距、搬运方式）——不仅是几何换皮。
- **规模：** 4 条真人 seed × 每类 16 sample × 4 类 → **256** 视频；生成模型 **SeedDance 2.0**。
- **对 wiki 的映射：**
  - [PRISM（Real2Sim2Real loco-manip）](../../wiki/entities/paper-prism-real2sim2real.md)
  - [Loco-Manipulation 任务页](../../wiki/tasks/loco-manipulation.md)

### 2) 接触锚定 Real2Sim 与重定向（§3.2–3.3）

- **链接：** <https://arxiv.org/abs/2609.38172> Fig.2
- **核心贡献：** 后端 **CRISP** 类单目人–场景–相机恢复，扩展 **动态物体**（SAM 2 掩码 + SAM3D 网格）；接触阶段用 **掌–物锚** 正则物体 6D 而非独立 FoundationPose 跟踪（遮挡敏感）。**Contact-anchored retargeting** 在 interaction-preserving IK 上加 **末端–锚点** 项，把 noisy 重建修成可仿真 robot–object 轨迹。
- **对 wiki 的映射：**
  - [CRISP Real2Sim 方法页](../../wiki/methods/crisp-real2sim.md)
  - [Motion Retargeting 概念](../../wiki/concepts/motion-retargeting.md)

### 3) 统一深度策略：co-tracking teacher → DAgger+PPO 学生（§4）

- **链接：** <https://arxiv.org/abs/2609.38172>
- **核心贡献：** **Privileged co-tracking teacher** 同时跟踪人形与物体参考，并加 **接触锚点奖励**（式 2）；**Student** 仅本体 + 摇杆 + **机载深度**（仿真中深度噪声/洞/偏移 + 相机随机化）；蒸馏 **λ·PPO + (1−λ)·DAgger**，末 20K iter λ=0.9。部署：**Unitree G1 29-DoF**，D435i 立体 → **Fast-FoundationStereo** 离板深度，**50 Hz**，无 MOCAP/参考 motion。
- **对 wiki 的映射：**
  - [DAgger 方法](../../wiki/methods/dagger.md)
  - [VideoMimic 实体](../../wiki/entities/videomimic.md)（视频模仿对照）

### 4) 评测与消融（§5）

- **链接：** <https://arxiv.org/abs/2609.38172> Table 1–3
- **核心贡献：** 仅 **OMOMO** 训的策略在 PRISM OOD 上 **12.5%**；PRISM-ID 训的在 OMOMO test **100%**、PRISM OOD **72.92%**。真机 in-domain box **100%**、bin/barrel **93%** 等；OOD 椅子/灯/背包等多类 **60–100%**。消融：baseline（SAM3D+FoundationPose+OmniRetarget）PRISM-ID **22.5%** → 全栈 **96.25% / 72.92%**。
- **对 wiki 的映射：**
  - [Sim2Real 概念](../../wiki/concepts/sim2real.md)
  - [Real2Sim 纵深路线 Stage 4](../../roadmap/depth-real2sim.md)

## BibTeX（项目页 / arXiv）

见 arXiv 页面；标题 *Counterfactual Video Generation Enables Scalable Humanoid Loco-Manipulation*，作者含 Zihan Wang、Zhen Wu 等（Amazon FAR × UC Berkeley × CMU × Stanford）。

## 当前提炼状态

- [x] Abstract 与项目页 Method 区对齐
- [x] 开源状态：项目页 Code 链至 `amazon-far/PRISM-Real2Sim2Real`；仓库 **404 待跟进**
- [x] wiki 实体页 [`paper-prism-real2sim2real.md`](../../wiki/entities/paper-prism-real2sim2real.md)
