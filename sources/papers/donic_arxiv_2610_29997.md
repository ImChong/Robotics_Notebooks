# DONIC: Generalist Dexterous Bimanual Tracking Control from Human Demonstrations（arXiv:2610.29997）

> 来源归档（ingest）

- **标题：** DONIC: Generalist Dexterous Bimanual Tracking Control from Human Demonstrations
- **简称：** DONIC
- **类型：** paper / dexterous-manipulation / bimanual / imitation-learning / reinforcement-learning / tracking
- **arXiv：** <https://arxiv.org/abs/2610.29997>（v1，2026-10-21）
- **作者：** Mingze Sun、Ziang Cao、Yuan Fei、Qiankuan Tao、Zihan Wang、Tianqi Liu、Junhao Chen、Haiwen Diao、Fangzhou Hong、Zhaoxi Chen、Ziwei Liu
- **机构：** Ropedia
- **项目页：** <https://ropedia.github.io/DONIC/>
- **官方代码仓库：** <https://github.com/Ropedia/DONIC>
- **数据集：** BiHOI；论文介绍 1,575 段、229 个物体、14.77 小时的多视角双手交互序列
- **入库日期：** 2026-10-23
- **一句话说明：** DONIC 将多类操作和多个人类动作来源联合训练到一个双手跟踪策略，以物体运动与各手接触角色为训练信号，泛化到未见轨迹和物体。

## 摘要级要点

- **问题：** 灵巧手参考跟踪研究多集中在单手；双手接触时存在协作角色不同、接触时序异步和高质量数据稀缺等难点。
- **数据：** BiHOI 使用多视角数据重建双手和物体，再重定向为机器人参考轨迹；与公开数据源共同用于训练。
- **策略：** 单一策略同时控制两只 Wuji 灵巧手，训练结合 PPO 与行为克隆，并把经过验证的成功执行回收到示范中。
- **奖励与优势：** 物体中心、逐手奖励持续跟踪物体运动，并按参考接触角色监督每只手；Clip-Centered Advantages（CCA）减去每段轨迹的平均优势偏置，同时保留段内差异。
- **评测：** 不微调时，在训练物体上的未见轨迹成功率为 86.79%，在留出物体上为 57.32%；相对双手化 DexTrack、Dexplore 基线分别高 77.36 和 45.73 个百分点。该设置测试轨迹与物体泛化，不等于未见操作类别或任意目标泛化。
- **开源状态：** 官方仓库已提供 BiHOI 处理 pipeline；DONIC 训练/评测代码、策略权重和数据集在仓库中标为待发布（核查日期：2026-10-23）。

## 核心论文摘录（MVP）

### 1) 面向双手接触的通用跟踪
- **链接：** [arXiv 摘要](https://arxiv.org/abs/2610.29997)
- **摘录要点：** 一个共享策略跨操作类别和人类动作来源训练；物体运动反馈与左右手各自的接触参考共同约束执行。

### 2) BiHOI 数据集
- **链接：** [DONIC 项目页](https://ropedia.github.io/DONIC/)
- **摘录要点：** 项目页展示 1,575 段双手交互、229 个物体和 32 视角采集；重建手与物体，再重定向到机器人参考轨迹。

### 3) 代码与数据处理
- **链接：** [官方 DONIC 仓库](https://github.com/Ropedia/DONIC)
- **摘录要点：** 仓库说明 BiHOI pipeline 可用；训练及评测实现、权重和 BiHOI 数据集仍列在待发布清单中。仓库明确列出 pipeline 依赖项目的非商业许可条款。

## 对 wiki 的映射

- 实体页：[paper-donic.md](../../wiki/entities/paper-donic.md)
- 相关主题：灵巧手-物体跟踪、双臂/双手操作、模仿学习、强化学习

## 本次补充核查（2026-11-01，未完成）

- 已定位 [论文 PDF v1](https://arxiv.org/pdf/2610.29997)、[项目页](https://ropedia.github.io/DONIC/)与[官方仓库](https://github.com/Ropedia/DONIC)；本次尚未完成全文与 BiHOI pipeline 运行入口源码通读，须完成后补充项目流程图和源码运行时序图。
- 官方仓库目前列出 BiHOI 处理 pipeline，DONIC 训练与评测代码、权重及 BiHOI 数据集仍列为待发布；处理代码及第三方依赖的许可条件需分别核对。
