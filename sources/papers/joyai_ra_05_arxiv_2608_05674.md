# JoyAI-RA 0.5：Scaling Robot Manipulation Learning via Dual Action Alignment

> 来源归档（ingest）

- **标题：** JoyAI-RA 0.5: Scaling Robot Manipulation Learning via Dual Action Alignment
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.05674>
- **PDF：** <https://arxiv.org/pdf/2608.05674>
- **项目页：** <https://joyai-ra-05.github.io/> — 归档见 [`sources/sites/joyai-ra-05-github-io.md`](../sites/joyai-ra-05-github-io.md)
- **机构：** Joy Future Academy / JoyAI-RA Team（京东 JD）；通讯 Yuzheng Zhuang、Liang Lin
- **骨干：** VLM + Latent-Action-Conditioned World Model（LAC-WM）+ Flow-Matching Action Expert（VLWA）
- **预训练规模：** 人视频 **53K+ h**（含自采 EgoLive **20K+ h**）+ 仿真 **11K+ h** + 真机 **8K+ h**
- **统一动作：** **130-D** 规范状态–动作槽 + 相机系 chunk-relative 末端执行器动作
- **评测平台：** 智元 **AgiBot G1** Real-World AgiBot Benchmark（seen / unseen）
- **入库日期：** 2026-08-07
- **一句话说明：** 用 **隐式 latent-action 对齐**（吃无动作标签人视频）+ **显式 130-D 规范动作对齐**（吃可靠人/机轨迹）把异构数据变成互补监督，再经 **内–外环 RL** 做任务适应与底座改进；在 AgiBot 真机上 seen **92.0** / unseen **75.5**，且人视频缩放未见饱和。

## 开源核查（步骤 2.5）

- **项目页：** <https://joyai-ra-05.github.io/>（核查日 **2026-08-07**）
- **代码：** **未列** GitHub / Hugging Face / 权重下载；页上仅 arXiv 与技术报告叙事
- **结论：** **确认未开源**（截至入库日）；论文亦未给出「code will be released」URL。wiki 实体页 `## 源码运行时序图` 写 **不适用**。

## 核心摘录（面向 wiki 编译）

### 1) 问题：异构数据朴素混训会负迁移

- **链接：** arXiv Abstract；§1 Introduction
- **摘录要点：**
  - 通才操作策略需要人视频 / 仿真 / 真机，但监督形式与具身互不兼容；人视频规模最大却离机器人最远。
  - 现有 **VLA** 有语义、缺动力学，难吃 unlabeled 人视频；**WAM** 有动力学、缺语义与可执行动作标签。
  - 主张：**Vision-Language-World-Action（VLWA）** + **dual action alignment**，把异构源路由到各自可靠的监督通道。
- **对 wiki 的映射：**
  - [JoyAI-RA 0.5](../../wiki/entities/paper-joyai-ra-05.md) — 总实体
  - [VLA](../../wiki/methods/vla.md) — 与 π₀.₅ / InternVLA-A1.5 对照
  - [World Action Models](../../wiki/concepts/world-action-models.md) — VLWA / LAC-WM 定位

### 2) 隐式对齐：多视角 LAM → latent action 条件化 LAC-WM

- **链接：** arXiv §3.3.1；§4.2.1 Stage 1
- **摘录要点：**
  - 头/左腕/右腕三视图 **SpatialConcat** 成统一复合观测；跨人/仿/机共享 **变分 LAM**，由 \(q_\eta(\mathbf{z}\mid\bar{\mathbf{o}}_t,\bar{\mathbf{o}}_{t+1})\) 推断转移 latent。
  - 重建项含 L1 / LPIPS / DINO / flow / depth / VGGT + KL 瓶颈；训练后冻结 encoder，用后验均值 \(\bar{\mathbf{z}}_t\) 作离线标签。
  - Stage 1：以 \(\bar{\mathbf{Z}}_t\) / 语言 / 二者组合条件化 LAC-WM，做未来视频 **flow matching + temporal-difference**；**condition dropout** 使下游可无 latent 推理。
  - 部署时 LAC-WM **冻结**，因果取第一帧隐特征 \(\mathbf{D}_t\) 交给动作专家（不滚像素未来）。
- **对 wiki 的映射：**
  - [JoyAI-RA 0.5](../../wiki/entities/paper-joyai-ra-05.md) — LAC-WM / LAM 机制
  - [World Action Models](../../wiki/concepts/world-action-models.md) — 「训练耦合、部署抽特征」族谱
  - [EgoWAM](../../wiki/entities/paper-egowam-egocentric-human-wam-co-training.md) — 人视频 + 世界目标对照

### 3) 显式对齐：130-D 规范槽 + 相机系 chunk-relative EE

- **链接：** arXiv §3.3.2 Table 1；Eq.(4)–(5)
- **摘录要点：**
  - **130-D** 左右臂/夹爪/指尖/灵巧手 + 底座/头/腰 + 预留槽；具身 adaptor \(\Phi_e\) 映射原生状态/动作，无效维用 mask。
  - 人轨迹用稀疏任务空间（腕位姿 + 五指尖 XYZ），避免稠密 MANO retarget。
  - 末端目标写成 **条件相机系下相对条件 EE 状态的 chunk-relative** 变换，削弱绝对位姿与标定差异。
- **对 wiki 的映射：**
  - [JoyAI-RA 0.5](../../wiki/entities/paper-joyai-ra-05.md) — 规范动作表
  - [Action Chunking](../../wiki/methods/action-chunking.md) — chunk 条件化读法
  - [Green-VLA](../../wiki/entities/paper-greenvla-staged-vla-humanoid.md) — 语义统一动作空间对照（64-D）

### 4) VLWA 架构与四阶段训练 + 内–外环 RL

- **链接：** arXiv §4.1–4.3；Figure 4–5
- **摘录要点：**
  - **Late fusion：** VLM 语义 \(\mathbf{U}_t\) ∥ LAC-WM 动力学 \(\mathbf{D}_t\) → Flow-Matching Action Expert 输出 \(H\times 130\) chunk。
  - Stage 2：冻 LAC-WM，联合训 VLM + Action Expert（\(\mathcal{L}_{VQA}+\mathcal{L}_{FAST}+\mathcal{L}_{FM}\)）。
  - Stage 3：目标机器人（AgiBot G1）高质量演示 post-train。
  - Stage 4：**内环**（边缘）冻底座、学轻量 residual 策略做任务适应；**外环**（中心）异步用成功交互改进 VLWA 并低频同步回边缘。
- **对 wiki 的映射：**
  - [JoyAI-RA 0.5](../../wiki/entities/paper-joyai-ra-05.md) — 训练与 RL 流程图
  - [InternVLA-A1.5](../../wiki/entities/paper-internvla-a15-unified-vla.md) — VLM + foresight/WM + flow 对照
  - [VLA](../../wiki/methods/vla.md) — foundation + 后训练 RL 选型

### 5) 真机 AgiBot Benchmark 与人视频缩放

- **链接：** arXiv §5.1–5.4；Figure 7–11
- **摘录要点：**
  - 六场景（办公/茶室/厨房等）；PnP-Easy / PnP-Hard / Long-Horizon；unseen 含 STG / OCAG / BIG。每任务 20 seen + 10 unseen，子任务完成率 0–100 分。
  - vs \(\pi_{0.5}\)：seen 均分 **92.0 vs 74.0**；unseen 均分最高 **75.5**；去掉双对齐后 unseen 跌至 **29.8**。
  - 隐式对齐主贡献 **外观/光照泛化**；显式对齐主贡献 **seen 精度（PnP-Hard）**。
  - EgoLive 子集 10%→100%：seen **47.8→85.6**、unseen **37.6→60.2**（相同机器人 post-train）；LAC-WM 人视频 10%→100%：seen **83.1→97.5**、unseen **56.9→72.4**（机器人数据固定）。
  - 内–外环 RL（鼠标/耳机位姿 OOD）：两环组合优于单环，达约 **70% / 50%** success。
- **对 wiki 的映射：**
  - [JoyAI-RA 0.5](../../wiki/entities/paper-joyai-ra-05.md) — 结果表
  - [EgoLive](../../wiki/entities/paper-ego-02-egolive.md) — 自采人视频数据源
  - [Manipulation](../../wiki/tasks/manipulation.md) — 真机操作评测语境
  - [具身数据金字塔](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md) — 人视频作主缩放轴

### 6) 局限与工程启示

- **链接：** arXiv §5.2.2；§6 Conclusion
- **摘录要点：**
  - 外环权重同步频率偏低；高频同步会破坏 off-policy 稳定——高同步机制仍是开放问题。
  - 代码/权重截至项目页核查日 **未发布**。
  - 启示：人视频可作 **主缩放轴**（未见饱和），前提是双通道对齐而非朴素混池。
- **对 wiki 的映射：**
  - [JoyAI-RA 0.5](../../wiki/entities/paper-joyai-ra-05.md) — 局限与结论
  - [Ego 分类 02：人→机器人](../../wiki/overview/ego-category-02-human-to-robot.md) — 人视频进策略的对齐叙事

## 当前提炼状态

- [x] arXiv HTML 全文已对齐摘录（2608.05674）
- [x] 项目页开源核查：未列代码（2026-08-07）
- [x] wiki 映射：`wiki/entities/paper-joyai-ra-05.md` 新建
- [ ] 待官方代码/权重公开后补 `sources/repos/` 与源码时序图
