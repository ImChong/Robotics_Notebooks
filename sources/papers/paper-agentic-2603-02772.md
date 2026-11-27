# Self-Evolutionary Replanning for Failure-Aware Motion Planning

- **作者：** Guoliang Li, Ruihua Han, Chengyang Li, He Li, Shuai Wang, Wenchao Ding, Hong Zhang, Chengzhong Xu
- **机构：** 澳门大学（University of Macau）、香港大学（University of Hong Kong）、中国科学院深圳先进技术研究院（SIAT, CAS）、复旦大学（Fudan University）、南方科技大学（SUSTech）
- **版本 / 载体：** arXiv:2603.02772v2 [cs.RO]（2026 年 3 月 3 日提交 v1，2026 年 9 月 26 日更新 v2；8 页，9 图，4 表，已投 IEEE 待审）
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2603.02772) · [arXiv HTML 全文](https://arxiv.org/html/2603.02772)
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv v2 原文、摘要页及 HTML 全文均未列出独立项目主页或官方源码仓库链接）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§I Introduction、§II Related Work、§III System Overview、§IV-A ILAD、§IV-B MARS、§V-A IR-SIM Simulation、§V-B Habitat Simulation、§V-C Real-World Experiments、§VI Conclusion，以及 Table I–IV 与全部公式 Eq. (1)–(16)）
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [Self-Evolutionary Replanning for Failure-Aware Motion Planning (SERP)](../../wiki/entities/paper-agentic-2603-02772.md)
- **问题定义：** 面向闭环执行失败的运动规划器自演化重规划框架 SERP，同时自适应规划器参数 $\theta = \{q_s, p_u, \eta, d_{\min}, d_{\max}\}$ 与自适应损失权重 $\mathcal{W} = (\alpha, \beta, \gamma, \omega)$。
- **核心机制一（ILAD，§IV-A）：**
  - 局部数值优化（AD）通过可微规划器逆向传播执行损失 $\mathcal{L}_{\text{ILAD}}(\theta; \mathcal{W})$ 更新 $\theta$（Eq. 8），保持 $\mathcal{W}$ 不变。
  - 当近期适应历史 $\mathcal{B}_k$ 持续失败且相对损失下降低于阈值 $\delta_L$ 时触发语境学习（IL，Eq. 9，$z_k = 1$），由 VLM（IR-SIM 中使用 `Qwen3-VL-Plus`）联合改写 $\theta$ 与 $\mathcal{W}$ 并重置 $\mathcal{B}_k$。
- **核心机制二（MARS，§IV-B）：**
  - 维护成功/失败恢复记忆 $\mathcal{M} = \{\mathcal{M}^+, \mathcal{M}^-\}$，以第一视角 RGB 与两张局部几何示意图（全局轴对齐、局部参考路径对齐）索引失败状态。
  - 使用 `Qwen3-VL-Embedding-2B` 召回并使用 `Qwen3-VL-Reranker-2B` 结合局部几何特征重排，得到 $\mathcal{R}_k$。
  - VLM 基于 $\mathcal{R}_k$ 生成 $J$ 个候选配置 $\hat{\mathcal{Q}}_k$（Habitat 实验使用 `Qwen3.7-Plus` 生成 $J=4$ 个候选），在低保真重放环境（IR-SIM）中并行执行最多 $N=3$ 轮带 AD 的闭环推演（Eq. 13），由 VLM 选出最佳配置 $(j^\star, n^\star)$ 应用至目标环境，并将真实执行恢复轨迹沉淀回 $\mathcal{M}$。
- **实验核对摘要（Table I–IV）：**
  - IR-SIM（65 m × 65 m，30/40/50 障碍物各 100 场景）：SERP 成功率分别为 95.0%、91.0%、87.0%（DiffTune 为 79.0%、76.0%、60.0%），碰撞率分别为 1.0%、4.0%、5.0%（DiffTune 为 17.0%、21.0%、35.0%）。
  - IR-SIM 消融（50 障碍物 100 场景，Table II）：完整 SERP 94.0% 成功率；去 IL 降至 74.0%（碰撞率升至 20.0%）；去 AD 降至 84.0%。
  - Habitat HM3D（6 个场景各 50 测试 = 300 场景，Table III）：在线空记忆起步下 SERP 成功率 87.7%（碰撞率 1.7%），高于 DiffTune（77.0%，碰撞率 4.7%）与去 MARS 变体（81.3%，碰撞率 2.7%）；但 TotTime 增至 74.72 s（对比 DiffTune 49.46 s、SERP w/o MARS 55.90 s）。
  - 真机验证（带 3D 雷达、RGB-D 相机与 Intel NUC 的类车机器人，Table IV）：门全开场景下有/无 SERP 均通过（平均耗时 11.06 s vs 11.43 s）；门接近关闭窄通道（2/4/6 障碍物）下无 SERP 全部失败，启用 SERP 后 3 组全部成功通过（平均 18.59 s；Case 2 中遇到卡滞后 1 个适应轮次、耗时 $< 3\,\mathrm{s}$ 完成重规划）。
- **开源核查记录：** `context.json` 中 `candidate_repos=[]`、`github_urls=[]`，arXiv HTML 全文中也未提供 GitHub 仓库或独立项目页面链接，因此不创建 `sources/repos/paper-agentic-2603-02772.md`。
