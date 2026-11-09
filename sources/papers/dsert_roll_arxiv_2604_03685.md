# DSERT-RoLL（CVPR 2026；arXiv:2604.03685）论文原始资料归档

- **英文标题：** DSERT-RoLL: Robust Multi-Modal Perception for Diverse Driving Conditions with Stereo Event-RGB-Thermal Cameras, 4D Radar, and Dual-LiDAR
- **中文标题：** 面向多样化驾驶条件的鲁棒多模态感知——双目事件/RGB/热成像、4D 雷达与双 LiDAR
- **作者：** Hoonhee Cho、Jae-Young Kang、Yuhwan Jeong、Yunseo Yang、Wonyoung Lee、Youngho Kim、Kuk-Jin Yoon；前三位同等贡献
- **机构：** 韩国科学技术院（KAIST）Visual Intelligence Lab
- **会议：** CVPR 2026，pp. 33022–33035
- **arXiv：** https://arxiv.org/abs/2604.03685
- **CVF Open Access：** https://openaccess.thecvf.com/content/CVPR2026/html/Cho_DSERT-RoLL_Robust_Multi-Modal_Perception_for_Diverse_Driving_Conditions_with_Stereo_CVPR_2026_paper.html
- **论文全文（项目官方镜像）：** https://jeongyh98.github.io/dsert-roll/static/pdfs/paper.pdf
- **补充材料：** https://openaccess.thecvf.com/content/CVPR2026/supplemental/Cho_DSERT-RoLL_Robust_Multi-Modal_CVPR_2026_supplemental.pdf
- **官方项目：** https://jeongyh98.github.io/dsert-roll/
- **官方 3D 检测代码：** https://github.com/Chohoonhee/DSERT-RoLL-3DOD
- **官方双目基线代码：** https://github.com/mickeykang16/LIGA-DSERT-RoLL
- **数据集：** https://github.com/jeongyh98/DSERT-RoLL-Dataset
- **核对日期：** 2026-11-09
- **核对范围：** 论文正文第 1–7 节（背景、传感器/数据标注、方法、表 1–5、消融、3D/2D 评测、结论），主要图 1–4；补充材料的许可证条款和章节目录；官方项目页与三套 GitHub README、3D 检测评测入口及事件预处理源码。
- **通读：** 已完成
- **映射 wiki：** [DSERT-RoLL 主实体](../../wiki/entities/paper-dsert-roll.md)

## 核心事实（来自原文）

1. **数据集：** 约 22K 同步多模态帧；双目 RGB、双目事件、双目热成像、4D Radar、长/短距两台 LiDAR，以及 GPS/IMU。六种天气（晴、雾、轻/大雨、轻/大雪）、四种光照（正常、低光、过曝、HDR）；类别含 Vehicle、Pedestrian、Bike；有 2D/3D 框、跟踪 ID 与自车里程计。
2. **数据划分：** 论文第 3.2 节报告约 7:3 训练/测试比例；数据仓库公布 132 train / 58 val 共 190 个 sequence。两者计数粒度不同，不应把 sequence 数量当成 frame 数量。
3. **3D 融合策略：** LiDAR + 4D Radar 分别体素化，BEV 特征生成初始 3D box；稀疏体素索引求并集并融合；将体素中心经相机内外参投影至 RGB、thermal、event 特征图；通过可变形采样/跨注意力聚合、模态置信门控，最后对候选框执行 3D ROI pooling 与 box refinement。
4. **主模型使用的模态：** 论文第 5.1 节明确使用**长距 LiDAR**，相机模态仅用**左目**；不能把硬件的双 LiDAR / 双目全部误报为该 baseline 的输入。事件数据转为 5 通道 voxel representation。
5. **评测口径：** Waymo-style 3D AP、IoU 0.5，主表聚焦车辆，按天气与光照分别给数值。表 3、4 全融合（R+E+T+4R+L）在晴天为 90.30、雾为 71.42、重雪为 72.94，数值是各条件 AP，**不是无条件整体 AP**。仅 LiDAR 重雪为 54.14；对比 SAMFusion 同条件分别是 87.03、65.13、70.59。
6. **训练/设置：** 论文 4 张 NVIDIA Quadro RTX 8000 GPU；采样点 4、refinement S=6；训练损失是 RPN、confidence、regression 三项之和（两系数均设 1）。
7. **风险：** 模态多使硬件、标定、同步和算力成本增加；恶劣天气不等于全部场景完美鲁棒，表 3 表明重雪 AP 仍显著低于晴天。应单独核验对雷达/LiDAR 异步、外参误差、实车部署的适配。
8. **许可：** 补充材料 §1 明确数据集及配套代码供研究使用，按 CC BY-NC 4.0 非商业许可分发；任何商用需另行确认授权与各仓库的具体条款。

## 资料关系

- [官方项目页归档](../sites/dsert-roll-project.md)
- [数据集与可视化仓库归档](../repos/dsert-roll-dataset.md)
- [多模态 3D 检测仓库归档](../repos/dsert-roll-3dod.md)
- [双目 LIGA 代码归档](../repos/dsert-roll-liga-stereo.md)
