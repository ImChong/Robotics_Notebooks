# transformer-transformer.github.io（Transformer Transformer 项目页）

- **标题：** Transformer Transformer — A Unified Model for Motion-Conditioned Robot Co-design
- **类型：** site / project-page
- **URL：** <https://transformer-transformer.github.io/>
- **配套论文：** [arXiv:2607.25798](https://arxiv.org/abs/2607.25798) — 归档见 [`sources/papers/transformer_transformer_arxiv_2607_25798.md`](../papers/transformer_transformer_arxiv_2607_25798.md)
- **代码：** <https://github.com/real-stanford/transformer-transformer> — 归档见 [`sources/repos/transformer-transformer.md`](../repos/transformer-transformer.md)
- **PDF：** <https://transformer-transformer.github.io/static/paper.pdf>
- **入库日期：** 2026-07-30

## 一句话摘要

Stanford / Columbia 官方站点：给定操作示范（末端运动）与用户奖励，用统一 DiT 生成完整机器人（连杆/关节/电机/惯量）并用同一网络跨具身控制验证；展示 RoboTokens、Dynamics Self-Guidance、三设计空间 vs CMA-ES，以及 ALOHA 抛布真机结果。

## 公开信息要点（截至入库日）

- **机构 / 作者：** Huy Ha、C. Karen Liu、Shuran Song（Stanford + Columbia）。
- **页首卖点：** 抛布 ALOHA 设计跟踪误差 −73%、峰值关节速度 −30%；同一架构覆盖 wheeled bimanual / quadruped / humanoid 等形态与「生成 + 控制」用例。
- **导航：** Paper PDF · arXiv · **Code（GitHub）** · 技术总结视频。
- **主干板块：**
  - Demonstrate → Generate → Validate 三步共设计
  - **RoboTokens**（embodiment 蓝 / dynamics 橙；相对 MJCF 紧凑）
  - **Transformer Transformer**（DiT；掩码切换 generator / controller）
  - **Dynamics Self-Guidance**（自有动力学 → 奖励梯度引导）
  - 多轨迹 diffusion composition（UMI 洗碗 hold-out）
  - vs Random / CMA-ES 的 test-time scaling
  - ALOHA2 cloth flinging 真机对比
  - Limitations（primitive 几何、控制器相关 r=0.53、test-time 约 1 分钟平台、RL 数据贵）
- **BibTeX / 引用：** 页内 References 节；完整列表见论文。

## 开源核查（步骤 2.5，2026-07-30）

| 资产 | 状态 |
|------|------|
| 项目页 Code 链 | **有** → `github.com/real-stanford/transformer-transformer` |
| 训练/评测代码 | **已开源**（见 repos 归档） |
| 预训练权重 + 评测轨迹 | **已发布**（`real.stanford.edu/transformer-transformer`） |
| 大规模训练 Zarr | **已发布**（HF `hqhuy/transformer-transformer`） |
| 宣称「即将开源」残留 | 无；页上直接链 Code |

## 为何值得保留

- **非 PDF 证据：** 真机 ALOHA 前后对比、扩散生成多样形态、奖励/轨迹条件切换设计分布。
- **复现入口三角：** 项目页 ↔ GitHub ↔ arXiv 与 lab 托管资产一致。
- **共设计选型：** 与 VGDS / CMA-ES / 纯 RL 共设计对照时的一手叙事页。

## 关联资料

- 论文归档：[`sources/papers/transformer_transformer_arxiv_2607_25798.md`](../papers/transformer_transformer_arxiv_2607_25798.md)
- 代码仓库：[`sources/repos/transformer-transformer.md`](../repos/transformer-transformer.md)
- Wiki：[`wiki/entities/paper-transformer-transformer.md`](../../wiki/entities/paper-transformer-transformer.md)
