# Embodied-Navigator / TAMP-Nav（arXiv:2608.17512）

> 来源归档（ingest）— 根据 arXiv v2 论文全文及作者官方项目页整理，不是原文镜像。

- **论文标题：** Embodied-Navigator: Point, Think, Memorize, and Align for Efficient Navigation
- **方法名：** TAMP-Nav
- **类型：** embodied navigation / VLN / VLM-based navigation
- **作者：** Hongyan Feng、Sunlai Chen、Xuanyu Liu、Miao Pan、Yangfan Xie、Yuxiang Cui、Zhongxiang Zhou、Rong Xiong、Wenqi Zhang、Jianwei Yin、Yueting Zhuang、Xuhong Zhang
- **机构：** 浙江大学 School of Software Technology；浙江人形机器人创新中心；论文列示 OmniAI Group of ZJU ACES Lab
- **论文：** [arXiv 摘要](https://arxiv.org/abs/2608.17512) · [HTML v2 全文](https://arxiv.org/html/2608.17512v2) · [PDF](https://arxiv.org/pdf/2608.17512)
- **项目页：** <https://zju-omniai.github.io/Embodied-Navigator/>
- **代码：** <https://github.com/ZJU-OmniAI/Embodied-Omni>（原仓库已迁移至此）
- **版本：** v1 提交于 2026-08-18；v2 更新于 2026-08-27
- **一句话说明：** TAMP-Nav 让 VLM 在图像中选二维像素作为导航目标，再结合深度投影给 SLAM 控制器；用关键节点触发 CoT、锚点轨迹记忆与 Space-Time Indicators（STI）压缩历史，并以双层 GRPO 对齐导航结果和过程反馈。

## 论文要点

1. **Point：像素到三维动作。** VLM 从 RGB 图像中选择二维像素，系统利用深度把目标投影为三维 waypoint，再交给低层 SLAM 控制器，减少让 VLM 直接回归连续三维坐标的负担。
2. **Think：选择性推理。** 模型根据导航状态决定何时生成 Chain-of-Thought（CoT），把推理集中在路口、门口等关键决策节点。
3. **Memorize：锚点轨迹记忆。** Anchor-Trajectory Memory 保留关键节点的高保真历史，并以轻量 Space-Time Indicators 压缩冗余路径信息，面向长程导航控制上下文开销。
4. **Align：双层 GRPO。** 将全局轨迹结果奖励与局部过程奖励结合，用更密集的反馈优化规划与物理环境执行之间的一致性。

## 论文报告结果

- R2R-CE Val-Unseen：成功率（SR）**66.2%**，SPL **58.8%**。
- RxR-CE Val-Unseen：SR **65.7%**，SPL **56.9%**。
- 训练使用 **90k 条轨迹、约 700k 次交互**。论文报告平均每条轨迹约 9 次交互；单张 A800 上每任务平均推理时间 **16.58 秒**。
- 在长程子集（专家路径长度 >12.5 m）上，AT-Mem 变体报告 SR **49.8%**。

这些数字来自论文报告的对应验证集与实验协议；不要跨基准或不同成功阈值直接比较。推理耗时也依赖模型、硬件与评测设置。

## 复现与代码

- 论文实验从 Qwen2.5-VL-7B 初始化，并使用固定的非学习 SLAM 控制器。
- 作者官方项目页提供论文介绍与演示；代码仓库现由 `ZJU-OmniAI/Embodied-Omni` 承载，旧的 `Embodied-Navigator` 仓库说明项目已迁移。
- 阅读复现入口时，应区分导航策略训练、像素深度到三维 waypoint 的转换，以及低层 SLAM 执行器。

## 对 wiki 的映射

- [Embodied-Navigator / TAMP-Nav 详情页](../../wiki/entities/paper-embodied-navigator-tamp-nav.md)
- [视觉–语言导航任务页](../../wiki/tasks/vision-language-navigation.md)
- [导航纵深路线](../../roadmap/depth-navigation.md)
