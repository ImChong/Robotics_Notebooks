# DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation

> 来源归档（ingest · 2026-09-17）

- **标题：** DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation
- **类型：** paper
- **作者：** Chen Wang, Haochen Shi, Weizhuo Wang, Ruohan Zhang, Li Fei-Fei, C. Karen Liu
- **机构：** 斯坦福大学（Stanford）— The Movement Lab / SVL
- **会议：** RSS 2024
- **arXiv：** <https://arxiv.org/abs/2403.07788>
- **项目页：** <https://dex-cap.github.io/>
- **代码：** <https://github.com/j96w/DexCap>
- **数据集：** <https://huggingface.co/datasets/chenwangj/DexCap-Data>
- **分类：** 06_Manipulation
- **入库日期：** 2026-06-11（占位）；2026-09-17 深读升格
- **一句话说明：** 可穿戴 SLAM+EMF 手套 mocap 与胸挂 RGB-D 点云，经 DexIL（指尖 IK + 点云模仿学习）把人类野外示范迁移到 LEAP Hand 双臂，可选人机闭环修正。

## 核心摘录（策展，非全文）

1. **问题：** 现有 hand mocap 便携性差，且 mocap→控制策略的迁移困难；纯视觉 VR 手追踪在遮挡交互中易失败。
2. **DexCap 系统：** 胸挂 RGB-D LiDAR + 3 SLAM 相机追踪腕/掌；Rokoko EMF 手套测相对掌系指尖 3D；背包容 NUC 约 40 分钟续航；采集吞吐约为 teleoperation 的 3×。
3. **观测对齐：** 相机架快拆（<20 s）可在人与机器人间切换，使策略使用与人类采集相同的 chest 相机视角。
4. **DexIL：** RGB-D 建 3D 点云 → 机器人操作空间；指尖 IK 把 EMF 手套数据重定向到 LEAP Hand 16 维关节；可见人手时用 FK 生成机器人手点云 mesh 补 visual gap；Diffusion Policy 以点云为输入预测 20 步 46 维双臂+双手动作。
5. **人机闭环：** rollout 时脚踏切换 residual 腕部修正与全手 teleop IK；修正轨迹与原数据均匀采样 fine-tune（如 1 h mocap + 30 次修正完成泡茶/剪刀任务）。
6. **评测：** 六项灵巧任务；30 分钟 mocap 无 teleop 自主 rollout；in-the-wild 数据可泛化未见物体。
7. **开源（2026-09-17 核查）：** MIT 代码 + Hugging Face 原始/处理后数据均已发布。

## 对 wiki 的映射

- [paper-notebook-dexcap-scalable-and-portable-mocap-data-collecti](../../wiki/entities/paper-notebook-dexcap-scalable-and-portable-mocap-data-collecti.md)
- 项目页：[dexcap.md](../sites/dexcap.md)
- 代码：[dexcap.md](../repos/dexcap.md)
- 分类父节点：[paper-notebook-category-06-manipulation](../../wiki/overview/paper-notebook-category-06-manipulation.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2403.07788>
- 项目页：<https://dex-cap.github.io/>
- 代码：<https://github.com/j96w/DexCap>
- 数据集：<https://huggingface.co/datasets/chenwangj/DexCap-Data>
