# RoM-Nav 项目页

- **URL：** <https://wdc3iii.github.io/rom-nav/>
- **论文 PDF（页内）：** <https://wdc3iii.github.io/rom-nav/paper/rom-nav.pdf>
- **arXiv：** <https://arxiv.org/abs/2609.19272>
- **机构：** 加州理工学院（Caltech）计算与数学科学系；亚马逊 Safe Autonomy Frontiers（SAF）实验室
- **作者：** William D. Compton、Zachary Olkin、Ryan Bena、Aaron D. Ames
- **代码：** 截至 **2026-10-01** **待发布** — 页眉仅有 Paper / arXiv（占位）/ Video，**无 GitHub 或 Hugging Face 链接**；BibTeX 中 `eprint` 仍为 `TODO`
- **关联论文：** [rom_nav_arxiv_2609_19272.md](../papers/rom_nav_arxiv_2609_19272.md)
- **关联 wiki：** [paper-rom-nav](../../wiki/entities/paper-rom-nav.md)

## 步骤 2.5 核查摘要（2026-10-01）

- 项目页方法节含架构图、仿真对比表、Poisson CBF 硬件实验与 G1 真机四组部署视频；**未提供可克隆仓库**。
- 真机栈：**Unitree G1** + Mid-360 LiDAR + 下视 ZED Mini；导航 **5 Hz** 输出平面速度至 **50 Hz 冻结 locomotion 策略**（限幅：前向 1 m/s、横向 0.25 m/s、角速度 1 rad/s）。
- 训练算力：RoM 阶段单 H100 **~12 h**；RoM-Nav kickstart **~32 h**；合计 **<45 h** 单卡。
