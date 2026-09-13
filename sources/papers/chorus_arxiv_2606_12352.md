# CHORUS: Decentralized Multi-Embodiment Collaboration with One VLA Policy（arXiv:2606.12352）

> 来源归档（ingest）

- **标题：** CHORUS: Decentralized Multi-Embodiment Collaboration with One VLA Policy
- **简称：** CHORUS
- **类型：** paper / vla / multi-robot / multi-embodiment / decentralized / mobile-manipulation
- **arXiv：** <https://arxiv.org/abs/2606.12352>
- **PDF：** <https://arxiv.org/pdf/2606.12352>
- **Hugging Face Papers：** <https://huggingface.co/papers/2606.12352>
- **项目页：** <https://chorus-model.github.io/> — 归档见 [`sources/sites/chorus-model.md`](../sites/chorus-model.md)
- **会议：** CoRL 2026
- **机构：** 斯坦福大学（Ria Doshi, Tian Gao, Annie Chen, Chelsea Finn, Jeannette Bohg）
- **入库日期：** 2026-09-13
- **一句话说明：** 在预训练 VLA（π₀.₅）上微调单一共享权重，推理时每台机器人独立运行、仅依赖本地观测与机器人身份 prompt，无需机间通信；真机多本体协作任务上显著优于从零扩散策略与集中式 VLA 基线。

## 开源状态（步骤 2.5，2026-09-13）

- **结论：** **截至入库日未开源** — 项目页有摘要、方法图、真机视频与 BibTeX，**无 GitHub / 权重 / 数据链接**；arXiv 页面亦无关联代码仓库。

## 核心摘录（面向 wiki 编译）

### 摘录 1：问题与去中心化设定

- 移动多机协作难在：集中式策略需拼接全队观测，随团队规模 **上下文与算力膨胀**；传统去中心化常需 **每机一策** 或推理时对齐/通信弥补部分可观测。
- CHORUS 主张：预训练 VLA 的 **visuomotor 先验** 足以让每台机器人仅凭 **本地相机 + 机器人身份 prompt** 做出 **反应式协作**，推理时 **零共享相机、零本体状态、零机间通信**。

**对 wiki 的映射：** [paper-chorus](../../wiki/entities/paper-chorus.md)

### 摘录 2：训练与部署机制

- 在 **π₀.₅** 骨干上微调 **单一共享策略**；robot sampler 从多机示范中抽取单机器人 `(observation, action)` 元组。
- 每步 prepend **机器人身份 prompt**；输出 **32 维 padded action**，统一不同本体动作空间与控制频率。
- 部署：各机加载 **同一权重副本**，异步执行；团队扩大时 **参数与上下文窗口不变**。

**对 wiki 的映射：** [paper-chorus](../../wiki/entities/paper-chorus.md)、[π0 Policy](../../wiki/methods/π0-policy.md)

### 摘录 3：真机任务与量化结果

- **平台：** Kinova、ARX、YAM 移动操作臂；二机与三机编队。
- **任务：** 洗衣篮对侧抬升、移动卷尺测量、图书馆图书交接、三机协同搬运等；协调完全依赖 **视觉感知队友**。
- **结果（项目页）：** 相对从零去中心化扩散策略平均成功率 **+64 pp**；队友扰动下反应性 **+40 pp**（相对无权重共享微调）；均值成功率 **优于集中式 VLA**；三机任务 **90%** 成功率且 **无架构改动**。

**对 wiki 的映射：** [paper-chorus](../../wiki/entities/paper-chorus.md)、[人形多机协调](../../wiki/concepts/humanoid-multi-robot-coordination.md)

## 当前提炼状态

- [x] 项目页核查（2026-09-13）
- [x] wiki 实体页已建
