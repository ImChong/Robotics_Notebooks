# OTRetarget: Joint Robot and Object Motion Retargeting via Optimal Transport（arXiv:2609.36602）

> 来源归档（ingest）

- **标题：** OTRetarget: Joint Robot and Object Motion Retargeting via Optimal Transport
- **类型：** paper / humanoid / human-object-interaction / motion-retargeting / loco-manipulation
- **arXiv abs：** <https://arxiv.org/abs/2609.36602>
- **PDF：** <https://simple-robotics.github.io/publications/otretarget/static/paper/otretarget.pdf>
- **Hugging Face 论文索引：** <https://huggingface.co/papers/2609.36602>
- **项目页：** <https://simple-robotics.github.io/publications/otretarget/> — 归档见 [OTRetarget 项目页](../sites/otretarget-project.md)
- **代码：** 截至 2026-10-02 项目页的 “Code” 文本没有可点击仓库 URL；按**待发布 / 未核实可运行代码**记录。
- **数据集：** OMOMO（[官方项目页](https://lijiaman.github.io/projects/omomo/)；[官方代码与下载说明](https://github.com/lijiaman/omomo_release)）。另有 [Hugging Face 社区镜像](https://huggingface.co/datasets/snorfyang/omomo)，页面未提供 dataset card 或明确许可，不视为官方发布。
- **作者 / 机构：** Guillaume Besset、Erwann Carn、Timothée Carecchio、Valentin Tordjman-Levavasseur、Fabian Schramm、Yann de Mont-Marin、Justin Carpentier（INRIA Willow）；Ajay Suresha Sathya（INRIA Willow / Stanford）。
- **发表时间：** 2026-09-29（arXiv 首次发布）
- **入库日期：** 2026-10-02
- **一句话说明：** 用表面几何关系与熵正则最优传输搬运交互目标，再用受约束 IK 联合求解机器人和多个物体的姿态，使物体轨迹能适配目标机器人，而非照搬人体演示的世界坐标轨迹。

## 核心摘录（面向 wiki 编译）

### 1) 研究问题

骨架姿态本身难以描述手掌与物体、手脚与地面的接触。若固定原始物体轨迹，身材不同的机器人可能够不到物体；若简单缩放整段场景，又可能破坏桌面和地面接触。

### 2) 方法

1. 在人体、机器人和场景物体的表面采样点，以有符号距离、最近表面点和相对方向构成 proximity triple，显式描述接近、接触及接触位置。
2. 在标准姿态上使用熵正则最优传输（Sinkhorn）建立人体部位与机器人连杆之间的表面对应。
3. 每帧用带有关节/速度限制和碰撞约束的 IK，同时优化机器人关节姿态与所有交互物体位姿；目标兼顾交互关系和原动作风格。
4. 可把演示物体替换为不同几何形状，并重新求解物体与机器人轨迹。

### 3) 评测与真机演示

- 在 OMOMO 上，论文报告机器人–物体交互 Jaccard 为 **87%**、深度误差 **8.7 mm**；论文报告的 OmniRetarget 对照分别为 **28%** 与 **29.3 mm**。
- 展示 Unitree G1 搬箱并放到桌面；下游全身策略在仿真训练后迁移到实体 G1。
- 几何接触指标衡量重定向质量，不等同于真机长时成功率或接触力安全性。

## 对 wiki 的映射

- 详情页：[OTRetarget](../../wiki/entities/paper-otretarget.md)
- 关联数据集：[OMOMO](../../wiki/entities/omomo-dataset.md)
- 主题方法：[动作重定向](../../wiki/concepts/motion-retargeting.md)、[OmniRetarget](../../wiki/entities/paper-hrl-stack-03-omniretarget.md)、[HOI-Retarget](../../wiki/entities/paper-hoi-retarget.md)

## 开源状态核查（2026-10-02）

- **论文与演示：** arXiv PDF 和项目页可访问。
- **代码：** 项目页显示 “Code” 字样，但未提供可点击仓库链接；暂列待发布 / 未核实，不将其标注为开源。
- **评测数据：** 论文使用 OMOMO。官方项目页/仓库提供其下载说明；Hugging Face 上可见一个社区镜像，但当前页面缺少 dataset card 与明确许可说明，不能替代官方许可信息。

## 当前提炼状态

- [x] 核对论文、项目页与 PDF
- [x] 记录代码链接状态与 OMOMO 数据入口边界
- [x] 建立 wiki 详情与数据集交叉引用
