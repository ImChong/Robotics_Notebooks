# HAIC：Humanoid Agile Object Interaction Control via Dynamics-Aware World Model（论文通读归档）

> 来源归档（ingest · 原文通读）

- **标题：** HAIC: Humanoid Agile Object Interaction Control via Dynamics-Aware World Model
- **作者与机构：** Dongting Li, Xingyu Chen, Qianyang Wu, Bo Chen, Sikai Wu, Hanyu Wu, Guoyao Zhang, Liang Li, Mingliang Zhou, Diyun Xiang, Jianzhu Ma, Qiang Zhang, Renjing Xu（清华大学、香港科技大学（广州）、ETH Zurich、小米机器人实验室）
- **类型：** paper
- **发表：** RSS 2026 / arXiv:2602.11758v2（2026-05-08）
- **原始链接：**
  - <https://arxiv.org/abs/2602.11758>
  - 项目页：<https://haic-humanoid.github.io/>
- **入库日期：** 2026-11-14
- **核对范围：** 通读 `arXiv:2602.11758v2` PDF 全文抽取文本（第 1–3597 行）：覆盖 Abstract、I 引言、II 相关工作、III 方法论（POMDP 建模、Object Adapter、显式几何投影、Privilege Adapter、Teacher-Student 与非对称微调）、IV 真机与消融实验（Table I–VII、§IV.D 泛化与家具扩展）、References 与 Appendix A–E（Table VIII–XV 资产规格、PD 参数、奖励项与推理时延）。未复算原始图表像素曲线；项目页补充视频里的 `0–20 kg` 拉车与 `0–70 kg` 推车外推范围见旧版项目页核查归档，与 arXiv 正文实验区间区分标注。
- **通读：** 已完成
- **一句话说明：** 针对滑板、手推车、多对象串联与跨坡梯搬运等**欠驱动（underactuated）且存在频繁视觉遮挡**的人形-物体交互（HOI）任务，提出无需外部状态估计器的动力学感知世界模型（Dynamics-Aware World Model）与非对称学生微调框架，实现仅依赖本体感知历史的真机敏捷交互。

## 核心论文摘录

### 1) 问题建模：欠驱动 HOI 的 POMDP 与 Teacher-Student 蒸馏
- **位置：** §III.A Problem Formulation
- **要点：** 与末端刚性夹持（fully actuated）物体不同，滑板、推/拉车等欠驱动物体拥有独立动力学与非完整约束，且交互中易遮挡机载视觉。策略在部署时仅观测量：本体历史 \(H_t = \{s_{t-n}^{prop},\dots,s_t^{prop}\}\)、未来参考动作 \(F_t = \{s_t^{ref},\dots,s_{t+m}^{ref}\}\)、前一动作 \(a_{t-1}\) 以及名义静态点云模板 \(\mathcal{P} \in \mathbb{R}^{N\times 3}\)。

### 2) Dynamics-Aware World Model：Object Adapter + 显式几何投影 + Privilege Adapter
- **位置：** §III.B
- **要点：**
  - **Object Adapter：** 基于历史本体与参考动作自回归预测物体相对机器人的高阶动力学状态 \(\hat{s}_t^{obj} = (\hat{p}_t, \hat{R}_t, \hat{v}_t^{lin}, \hat{v}_t^{ang}, \hat{a}_t^{lin}, \hat{a}_t^{ang})\)。
  - **Explicit Geometric Projection：** 将名义静态点云 \(\mathcal{P}\) 通过预测位姿变换到机器人根坐标系下 \(\mathcal{P}_t^{dyn} = \hat{R}_t \cdot \mathcal{P} + \hat{p}_t\)，提供空间接地的动态占据与碰撞边界。
  - **Privilege Adapter：** 结合 \(\hat{s}_t^{obj}\) 与 \(\mathcal{P}_t^{dyn}\) 重建特权编码 \(\hat{z}_t^{priv}\)，对齐 Teacher Privilege Encoder 的潜在特征 \(z_t^{priv}\)。

### 3) 非对称微调与真机实验结果
- **位置：** §III.C, §IV.B–D, Table I–VI, Appendix B–E
- **要点：**
  - 针对纯蒸馏下 Student 遭遇未见状态导致的分布偏移，第二阶段冻结/不再依赖 Teacher 动作直接监督，改为 **Asymmetric Fine-Tuning**：Critic 仍保留仿真特权信息，Student Actor 与 World Model（含 EMA 稳定化）在策略探索分布上联合微调。
  - **真机硬件（Unitree G1）：** 控制 23 个关节（不含 6 个腕部自由度），策略频率 `50 Hz`（真机 PD `500 Hz`，仿真 PD `200 Hz`），机载总推理时延 `0.17 ms`；单张 RTX 4090 上 4096 并行环境训练约 8 小时。
  - **对比基准 `HDMI*`（Table I–IV）：**
    - 滑板（Skateboarding）：Glide 成功率 `100%` vs `20%`，Complete 成功率 `60%` vs `0%`。
    - 推/拉车（Cart）：Pull 成功率 `100%` vs `40%`；Push 成功率 `100%` vs `0%`。
    - 多对象串联（Sequential）：`Pull Cart w/ Box` `40%` vs `0%`；`Push Cart w/ Box` `100%` vs `0%`。
    - 复合地形搬箱（Multi-terrain）：平地、楼梯、斜坡均 `100%`，`w/ Stair + Slope` 组合地形 `HAIC 100%` vs `HDMI* 0%`。
  - **泛化与消融（§IV.D, Table V–VII）：** 训练箱体范围 `(0.1–0.2 m, 1–2 kg)`，测试集在固定标准模板下零样本泛化至 `0.25 m, 3 kg`；日常家具（推桌、拉杆箱、带轴偏航转椅）`100%` SR；消融显示仅靠 `Proprio` 或只预测位姿 `Vec-Pose` 在滑板下板与推/拉车轨迹跟踪中明显退化，同时预测高阶动力学并作几何投影的完整 HAIC 综合表现最优，且对比 RMA 在 `Pull Cart` 上成功率更高（`100%` vs `70%`）。

## 对 wiki 的映射

- [paper-haic](../../wiki/entities/paper-haic.md)
- [haic](../../wiki/methods/haic.md)
