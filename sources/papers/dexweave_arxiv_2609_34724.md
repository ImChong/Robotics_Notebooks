# DexWeave（arXiv:2609.34724）

> 来源归档（ingest）

- **标题：** DexWeave: Learning Dexterous Humanoid Loco-Manipulation from Human Demonstrations
- **arXiv：** <https://arxiv.org/abs/2609.34724> · <https://arxiv.org/pdf/2609.34724>
- **项目页：** <https://dexweave.github.io/>
- **作者：** Naichuan Sun、Haotian Shen、Yizhang Zhang、Luying Feng、Haoze Wang、Yuanbo Xiangli、Yaochu Jin、Peidong Liu
- **入库日期：** 2026-10-05
- **摘要：** 交互一致重定向、解剖结构化策略与 G1/灵巧手部署。
- **详情：** [DexWeave](../../wiki/entities/paper-dexweave-humanoid-loco-manipulation.md)

## 核心摘录

> 2026-10-05 核对 arXiv PDF v1（26 pages）。

- 两阶段重定向：身体/手部专用求解器初始化后，沿上身交互链（手臂、手腕、手指）联合细化，保持下肢支撑。
- 解剖感知 Transformer：区域 token + 定向遮罩注意力，物体信息只条件化上身通路；PPO 单阶段训练，无跟踪器预训练、蒸馏或残差。
- 论文报告：灵巧移动操作成功率 97.5%（Object MLP 85.0%，InterMimic 57.74%），收敛约快 2×；GRAB/HUMOTO 主指尖误差 4.342 / 7.198 mm。
- IsaacLab 训练、MuJoCo sim-to-sim、Unitree G1 + Inspire 手真机定性部署。
