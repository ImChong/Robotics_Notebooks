# RAIBO-Wagon 官方项目主页

- **名称：** RAIBO-Wagon: Real-Time Navigation System for a Legged Robot Towing a Wagon
- **项目页：** https://insungyang.github.io/raibo-wagon/
- **论文：** https://arxiv.org/abs/2610.39267v1
- **机构：** 韩国科学技术院（KAIST）Robotics and Artificial Intelligence Lab
- **核对日期：** 2026-11-23
- **核对范围：** 官方项目页 TL;DR、Method 视频入口、Results、Baseline Study、BibTeX、顶部与页脚导航中的资源链接；对照论文 §VI 和附录
- **通读：** 已完成

## 开源核查

截至本次核查，项目页提供视频、方法说明、实验分组结果及引用信息；**未列出可访问的官方 GitHub 训练/推理源码、数据集或模型权重下载链接**。因此列为「**代码与数据未发现公开入口**」，不等同于断言作者永不开放，亦不能把非官方仓库当成可复现依据。

## 官方结果核对

- 实际平台为 RAIBO2 + 定制拖车，连接器编码器测量机器人与拖车相对 yaw；实验仅使用板载感知/计算。
- 15 m 室外无先验地图路线：8 类障碍（A–H），每类 5 次，**34/40** 成功；Terrain H 有两个人阻挡通路，成功 **3/5**。
- Terrain F 的基线：本方法 **5/5**，Hybrid A* **0/5**，RL 局部导航 **0/5**。
- 自重约 25 kg 的拖车额外装载 5/10/15 kg，在 Terrain F 分别 **5/5**；40 kg 为**含拖车本体的总质量**，不是额外 40 kg 载荷。
- 论文 Fig.5 显示 256 条 CFM 候选与经 MPPI 精修的输出，窄道中二者可能明显偏离 A* 直连路径。

## 相关归档

- [原始论文全文核对与局限](../papers/raibo-wagon-legged-towing-navigation_arxiv_2610_39267.md)
- [RAIBO-Wagon 独立论文节点](../../wiki/entities/paper-raibo-wagon-legged-towing-navigation.md)
