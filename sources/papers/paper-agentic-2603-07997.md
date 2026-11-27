# CMMR-VLN: Vision-and-Language Navigation via Continual Multimodal Memory Retrieval

- **作者：** Haozhou Li*、Xiangyu Dong*、Huiyan Jiang、Yaoming Zhou、Xiaoguang Ma†（通讯作者：`maxg@mail.neu.edu.cn`）
- **机构：** 东北大学（Northeastern University：佛山研究生创新学院、软件学院、机器人科学与工程学院）；北京航空航天大学（Beihang University，航空科学与工程学院）
- **版本 / 载体：** arXiv:2603.07997v1 [cs.AI]，2026-03-09 提交（CC BY 4.0，arXiv HTML 全文）
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2603.07997) / [arXiv HTML 全文](https://arxiv.org/html/2603.07997)
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，论文全文、arXiv 页面及公开索引均未提供独立项目主页或官方 GitHub 代码仓库链接）
- **核对范围：** 通读论文全部章节（Abstract、§I Introduction、§II Related Work、§III-A MEM、§III-B RAGP、§III-C Reflection & Memory Update、§IV-A~E Experiments & Real Robot、§V Conclusions）及 Table I–III 全部定量指标与消融实验；未见官方代码仓库，未做静态源码与本地运行核对。
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [CMMR-VLN 实体页](../../wiki/entities/paper-agentic-2603-07997.md)
- **研究动机（§I–§II）：** 既有训练自由（Training-Free）LLM 导航代理（如 NavGPT、MapGPT、DiscussNav）缺少对历史多模态导航经验的选择性回忆与再利用，在长程路线与视觉相似岔路口易随机选错并引发级联偏离。
- **三模块架构（§III）：**
  1. **Multimodal Experience Memory (MEM, §III-A)：** 以 Matterport3D 视点（或真机连续环境中基于显著地标划分的区域）为记忆单元，存储全景 SkyBox 图像、视点 ID、微调 Detic 检测的显著地标文本及历史经验；利用 CLIP 编码图像与地标文本得到混合嵌入 $\{m_j\}$，并用 FAISS 建立可双向反查的索引。
  2. **Retrieval-Augmented Generation Pipeline (RAGP, §III-B)：** 在每步将指令 $I$、候选视角 RGB 观测 $O=\{o_1,\dots,o_K\}$、历史轨迹与决策理由 $H_t=\{(\mathrm{id}_1,r_1),\dots,(\mathrm{id}_{t-1},r_{t-1})\}$ 及增量构建的语义拓扑图 $M=(V,E)$ 送入提示管理器。通过指令感知注意力 $\alpha_k = \operatorname{softmax}(u^\top W v_k)$ 融合多视角 CLIP 特征 $v_{\mathrm{obs}} = \sum_{k=1}^K \alpha_k v_k$，再用余弦相似度在 FAISS 中检索最相关历史经验 $E^*$，转化为显式高优先级约束规则 $R$（$\mathcal{PM}=\{I,O,H,M;R\}$），驱动 GPT-4o 输出 Analysis / Planning ($P_t$) / Action ($\mathrm{id}_t$)。
  3. **Reflection & Memory Update (§III-C)：** 回合结束后判定成功或失败（R2R 以终点距目标 $\le 3\text{ m}$ 为成功）。成功轨迹将指令与完整路径写入沿途各视点记忆单元，并通过效率过滤器仅保留更短/更优路径；失败轨迹定位**首个错误决策步**并归类为 MRD（Mid-route deviation）、FGR（False goal recognition）或 PGC（Post-goal continuation），仅将出错视点、决策理由、错误类型与全景图像写入记忆单元（重复错误不重复写入）。
- **实验核对（§IV, Table I–III）：**
  - **R2R val-unseen（11 场景、783 条轨迹，Table I）：** CMMR-VLN（GPT-4o）达到 NE = 5.10、OSR = 63、SR = 52、SPL = 51，优于 NavGPT（NE 6.46 / OSR 42 / SR 34 / SPL 29）、MapGPT（5.63 / 57 / 43 / 34）与 DiscussNav（5.32 / 61 / 43 / 40）。
  - **消融实验（R2R 子集 72 环境、216 条轨迹，Table II）：** 完整 CMMR-VLN（NE 5.08 / OSR 65 / SR 56 / SPL 50.91）对比去掉显式导航规则 `w/o NR`（5.59 / 56.5 / 48.2 / 40）与替换为固定场景描述 `w/ SD`（10.20 / 40 / 40 / 37.63），验证显式规则约束 $R$ 与反思筛选更新的必要性。
  - **真机实验（TurtleBot 4 Lite + 距底座 50 cm RGB 相机，20 条指令含 REVERIE 风格目标指令，Table III）：** 以地标区域划分构建记忆单元，采用离散运动原语（如前移 50 cm、左转 45°），CMMR-VLN 取得 SR = 30，高于 NavGPT（10）、MapGPT（20）与 DiscussNav（20）。
- **局限（§IV-D, §IV-E, §V）：** 性能依赖经验库覆盖度与质量；真机连续环境成功率仍仅 30%；需要 GPT-4o 推理开销与额外投影矩阵 $W$ 及微调 Detic 检测器，且官方未发布源码与权重。
