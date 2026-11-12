# DexSeed: Expanding Human Hand-Object Interactions into Robot Demonstrations for Dexterous Policy Learning / 将人手–物体交互扩展为灵巧操作示范

> 原始论文来源归档；这里只归纳论文的公开方法、数据与评测边界，不替代全文。

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.14092
- **全文：** https://arxiv.org/html/2611.14092v1
- **版本：** arXiv:2611.14092v1（2026-11）
- **核对范围：** HTML 全文：Introduction、Related Work、Method III-A–III-C、Experiments IV-A–IV-F、Conclusion；表 I–VII 及末尾参考文献；该 v1 HTML 未附单独附录。
- **通读：** 已完成
- **收录日期：** 2026-11-12
- **项目页：** https://robbyant-research.github.io/DexSeed/
- **官方代码：** https://github.com/robbyant-research/DexSeed

## 论文概要

DexSeed 是把少量人手–物体交互映射、扩增并物理验证为机器人灵巧抓取示范的数据合成框架。

## 机制要点

- 从少量人手交互种子出发，在模拟接触中生成机器人可执行示范。
- 核心步骤含抓取关系保持重定向、IK/轨迹扩增、RL 手指控制及 VLA 微调。
- 官方仓库仅发布介绍与素材，源码待发布；真机结果属于论文报告，尚未独立复现。

## 评测边界

论文报告由 19 条种子人手交互生成约 25,270 条机器人示范；仿真已见对象组合成功率 97.5%、未见组合配置约 91.25%；真实机器人所列八种抓取方式中，基线到方法的表现跨度约 20%→80%。实验协议与汇总方式需对照原表，不能误写成所有任务统一 80%。

## 通读原文：方法、实验与附录复核

### Grasp Bank 与生成链

从 GRAB 的 **19** 条人手交互筛出 **8** 类抓取，建立含手部几何、抓取和对象关系的 170 维特征；InteractionMesh 保持手物邻域结构，IK 扩增机械臂轨迹，Isaac Lab 中 **RL 手指策略**在机械臂跟踪误差下生成物理可执行示范，再借助 3D Gaussian Splatting 渲染头部和腕部相机画面训练 LingBot-VLA。

### 数据与训练设置

表 I：19 条种子→**25,270** 条已物理验证轨迹；RL 训练用 **4,096** 并行环境、1500 iter、30Hz policy/120Hz simulation。VLA 输出 7 维 Franka + 20 维灵巧手，8 张 H20 训练 55K steps；部署约 5Hz、50-step chunk，每次执行 30 步（RTC）。

### 数字必须保留语境

表 IV：仿真 Seen **97.50%**，未见布局/几何/外观的组合分别 **95.00% / 91.25% / 91.25%**。表 VII：真机 8 类抓取（G1–G8）逐项 **60、40、70、70、20、50、50、80%**，即不同抓取模式在 10 次试验中的 **20%–80%** 范围，绝非相对基线的 20%→80% 改进。局限是只评估抓取/放置，未覆盖 in-hand 与双手操作；无触觉输入。官方仓库当前仅 README/assets，训练实现待发布。

## 仍待独立验证

- 论文成功率与延迟属于作者报告结果，尚未在本地复现实验；需检查代码和数据版本是否可获取。

## 对 wiki 的映射

- [DexSeed](../../wiki/entities/paper-dexseed.md)
