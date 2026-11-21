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
- **通读：** 已完成
- **一句话说明：** TAMP-Nav（基座 Qwen2.5-VL-7B）让 VLM 从 4 视角全景 RGB 中选二维像素作为导航目标并由深度反投影给 SLAM 控制器；结合关键节点选择性 CoT（26.3% 触发率）、显式锚点 + RoPE 空间时间指示器（STI）记忆与两层 GRPO（G=8 轨迹 × M=4 候选动作），R2R-CE SR 66.2%，Unitree Go2 100 次真机零样本试验 SR 60.0%。

## 原文通读要点（arXiv:2608.17512v2 & 代码库）

1. **Point（像素到三维动作）：** 4 路全景 RGB 经 VLM 选择视角及 2D 像素，借助对齐深度和相机内参反投影到局部 3D 点再由固定 SLAM 控制器执行（SFT 对比直接输出度量 waypoint 将 R2R-CE SR 从 30.9% 提至 55.7%）。
2. **Think（选择性推理 + MultiNav-CoT）：** 基于时空关键节点挖掘（Spatiotemporal Key Node Mining）与 Gemini 2.5 Flash 构建 90k MultiNav-CoT 轨迹数据，训练模型仅在关键节点展开 CoT（R2R-CE 自动触发率 **26.3%**、均值 **9 次**交互/轨迹 vs StreamVLN/DualVLN 约 30 次）。
3. **Memorize（锚点轨迹记忆 AT-Mem）：** 触发 CoT 的节点保留显式锚点 `(STI, M_vis, M_state)`，锚点之间压缩为 RoPE 兼容时空指示器 `STI = (t, x, y, sin yaw, cos yaw)`；去掉 STI 后长程子集 SR 下降 4.2 个百分点。
4. **Align（两层 GRPO）：** 每条指令并行展开 `G=8` 条轨迹、每步 `M=4` 个候选动作与退火引导采样，将局部步奖励（接近目标、避碰、停止、推理价值、格式合法）与全局轨迹奖励（成功、SPL、推理密度）各自标准化叠加；将 R2R-CE SR 从 SFT 55.7% 进一步推高至 **66.2%**（RxR-CE **65.7%**）。
5. **真机部署与权重（`ZJU-OmniAI/Embodied-Omni/embodied_navigator`）：** 在 Unitree Go2（4 路 RGB-D + Hesai XT16，FastLIO + FAR Planner）开展 **100 次**零样本真机评测取得 **60.0% SR**（StreamVLN 49.0%，DualVLN 53.0%）；代码提供 `scripts/sft_dthink_7b.sh`、`scripts/grpo_dthink_7b.sh`、`scripts/evaluate.sh` 与 `src/server/` FastAPI + ROS2 客户端，权重点 `UnderTides/Embodied-Navigator-7B-GRPO` 已公开。

## 复现与代码

- 论文实验从 Qwen2.5-VL-7B 初始化，并使用固定的非学习 SLAM 控制器。
- 作者官方项目页提供论文介绍与演示；代码仓库现由 `ZJU-OmniAI/Embodied-Omni` 承载，旧的 `Embodied-Navigator` 仓库说明项目已迁移。
- 阅读复现入口时，应区分导航策略训练、像素深度到三维 waypoint 的转换，以及低层 SLAM 执行器。

## 对 wiki 的映射

- [Embodied-Navigator / TAMP-Nav 详情页](../../wiki/entities/paper-embodied-navigator-tamp-nav.md)
- [视觉–语言导航任务页](../../wiki/tasks/vision-language-navigation.md)
- [导航纵深路线](../../roadmap/depth-navigation.md)
