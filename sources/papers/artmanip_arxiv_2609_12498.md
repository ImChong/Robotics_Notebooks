# ArtManip: Category-Level Articulated In-Hand Manipulation（arXiv:2609.12498）

> 来源归档

- **arXiv：** <https://arxiv.org/abs/2609.12498>
- **PDF：** <https://arxiv.org/pdf/2609.12498>
- **项目页：** <https://artmanip.github.io/>
- **代码：** <https://github.com/youngcv/artgym>（ArtGym，官方实现）
- **机构：** 浙江大学、北京通用人工智能研究院（BIGAI）、清华大学、北京大学
- **开源状态：** **已开源**（仿真训练管线；步骤 2.5 复核 2026-09-15）
- **入库日期：** 2026-09-14（索引）；2026-09-15（深读 + 代码复核）
- **索引来源：** [wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md](../blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md)

## 一句话说明

类别级 **铰接物体手内操作**：程序化物体与 **功能抓取** 合成扩训练分布，**Teacher–Student** 两阶段 RL（特权状态 + 铰接物理随机化 + 奖励课程 + 潜表示蒸馏）在 **抓稳与驱动内部关节** 的耦合动力学下实现跨实例、跨初始抓取泛化，并 **零样本 sim2real** 到 12 个真机物体。

## 核心论文摘录

1. **任务定义：** Category-level in-hand manipulation of **articulated objects** — 在灵巧手上操作带内部关节的物体（如刀、订书机、夹子），要求泛化到 **同类未见实例** 与 **多样初始抓取**。
2. **瓶颈一（抓稳 + 推动）：** 控制物体 **内部 DoF** 与在 **自由漂浮基座** 上维持 **抓取稳定** 强耦合；手指既要施加驱动力矩/力推动关节，又不能导致物体在掌内滑脱或整体失稳。
3. **瓶颈二（数据规模）：** 获取多样铰接物体模型与 **任务导向功能抓取** 人工成本高，而策略对 **初始构型** 极敏感。
4. **初始构型管线：** 程序化生成 **两连杆 + 单 prismatic/revolute 关节** 的 box 原语资产；用 **类别级接触区域模板** 自动合成多样功能抓取。
5. **策略学习：** **两阶段** — Teacher 用 **特权状态**、**铰接物理随机化**、**奖励课程** 学鲁棒接触-关节动力学；Student 从 **可观测历史** 蒸馏 **潜表示**，用于部署与零样本真机。
6. **实验：** 四类物体类别仿真泛化；**12 个真实物体**（多样形状与关节机构）零样本迁移。

**对 wiki 的映射：** [paper-artmanip](../../wiki/entities/paper-artmanip.md)
