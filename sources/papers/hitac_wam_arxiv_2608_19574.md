# HiTac-WAM（arXiv:2608.19574）

> 来源归档（ingest）

- **标题：** HiTac-WAM: A Hierarchical Tactile World Action Model for Contact-Rich Robot Manipulation
- **类型：** paper / world-action-model / tactile / contact-rich
- **arXiv：** <https://arxiv.org/abs/2608.19574>
- **机构：** 中国科学院自动化研究所；ImprintX Robotics；北京智源人工智能研究院（BAAI，作者隶属）
- **入库日期：** 2026-08-22
- **一句话说明：** 分层触觉 WAM：对每个候选 action chunk 预测 **接触 → 3D 形变 → 滑移风险** 有向层次未来；用触觉预报排序候选并在执行期做 **预报–观测偏差** 触发重规划。

## 开源状态（步骤 2.5，2026-08-22）

| 资源 | 状态 |
|------|------|
| arXiv HTML/PDF | **已发布** |
| 项目页 / GitHub | **截至入库日未列公开链接** — 确认未开源 |

## 核心论文摘录

### 1) 问题与层次预报（Abstract / §I）

- **核心贡献：** 视觉 rollout 在遮挡下难以区分 **接触后果**；同视觉历史下多候选 chunk 可能 **外观相似、触觉后果不同**。HiTac-WAM 在预训练 WAM 上增 tactile 分支，将每个候选的触觉未来分解为 **contact → deformation → slip**，下游 stage 以 **stop-gradient** 条件于上游。
- **对 wiki 的映射：**
  - [HiTac-WAM 论文实体](../../wiki/entities/paper-hitac-wam.md)
  - [VT-WAM](../../wiki/entities/paper-vt-wam-visuotactile-contact-rich.md)（联合视触觉 WAM 对照）
  - [World Action Models](../../wiki/concepts/world-action-models.md)

### 2) 架构（§III）

- **Directed attention mask：** tactile query 可读 video–action 上下文；video/action query **不可读** tactile keys。
- **层次头（Eq. 3）：** \(\widehat{C}=\sigma(f_C)\)；\(\Delta\widehat{D}=f_D(z,\mathrm{sg}(z^C),a)\)；\(\widehat{p}^{\mathrm{slip}}=\sigma(f_R(\cdot,\mathrm{sg}(\widehat{C}),\mathrm{sg}(\|D\|)))\)；输出经 contact gate。
- **对 wiki 的映射：**
  - [Humanoid Transformer Touch Dreaming](../../wiki/methods/humanoid-transformer-touch-dreaming.md)（触觉预测族）

### 3) 选择与在线验证（§III-C–D）

- **选择：** 对 \(K\) 个随机候选批量预报，按任务进度 \(\rho_{\mathrm{task}}\) 与 \(J_C,J_D,J_R\) 成本排序，取 \(k^*\)。
- **执行：** 保留 \(\widehat{\mathcal{T}}^{(k^*)}\) 为参考；连续偏移超 KDE 阈值 \(\gamma_{\mathrm{task}}\) 则中止前缀、回安全态并重采样。
- **对 wiki 的映射：**
  - [Action Chunking](../../wiki/methods/action-chunking.md)

### 4) 实验（§IV）

- **平台：** IMETA-Y1 + 双侧 **DM-Tac W2** + 三 RGB；芯片抓取、黑板擦除、USB 插入。
- **预测：** contact F1 **0.921**；层次相对 deformation-only **−17.6%** 3D L2；slip AUPRC 相对 slip-only **+60.4%**。
- **真机：** 仅选候选 **31.1%→61.1%**；完整系统 **72.2%** 平均成功率。
- **对 wiki 的映射：**
  - [Bimanual Manipulation](../../wiki/tasks/bimanual-manipulation.md)

### 5) 局限

- 依赖特定 DM-Tac 硬件与 FG-CLTP 编码器；未开源；候选数与重规划延迟成本未充分报告。
