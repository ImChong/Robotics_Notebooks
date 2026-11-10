# ToolDex — Ego-Exo Creative Physical Tool Use（论文来源归档）

- **英文标题：** ToolDex: An Ego-Exo Human Demonstration Dataset for Creative Physical Tool Use
- **中文标题：** 创造性物理工具使用的第一/第三人称演示数据集
- **作者：** Mohamed Elmoghany 等（Stanford、CMU）
- **论文：** https://arxiv.org/abs/2611.09036
- **HTML 原文：** https://arxiv.org/html/2611.09036v1
- **项目页：** https://tooluse-dataset.github.io/
- **版本：** arXiv v1，2026-11-06
- **核对范围：** 阅读作者 HTML 原文的任务定义、数据统计、相机/标定/标注、3D/4D 重建、VLM 理解、机器人策略基准与附录 A.1–A.6；交叉查看项目页的公开展示和资源入口；未下载完整数据集或复现实验。
- **通读：** 已完成

## 论文证据及对 wiki 的映射

1. **手—工具—目标对象：** 将“人操纵工具、工具改变目标状态、环境支撑物施加约束”建模为有目标条件的操作过程。工具选择与接触几何不能简化为普通抓取。
   - 对 wiki 的映射：[ToolDex 实体](../../wiki/entities/paper-tooldex.md)「核心原理」。
2. **数据体量：** 10,853 次演示、58.3M 帧、60 种高层任务组合、168 个变体；45 种工具、23 种支撑物、16 类可变形对象、16 类动作。论文表 1 的 10.8K/58.3M 是舍入表示。
   - 对 wiki 的映射：实体「核心信息」。
3. **采集与标注：** 8 台 RealSense D435 外部 RGB-D 相机（1280×720@30Hz，软件同步），Vision Pro 上的 ZED Mini 提供头部第一视角立体 RGB；输出工具 6-DoF 位姿、手部 MANO、目标/工具分割、视频和语言指令。
   - 对 wiki 的映射：实体「流程总览」。
4. **三类 benchmark：** 物理工具选择/空间定位/行为规划、稀疏视角 3D/4D 重建、机器人模仿及强化学习策略。6 项策略迁移任务中平均成功率：直接运动学迁移 14.0%、BC 4.6%、ManipTrans (RL) 40.5%、Dex-X (RL) 55.1%；结果只针对论文的评测条件。
   - 对 wiki 的映射：实体「实验与评测」。
5. **局限：** 主要是室内桌面操作，材料及真实接触分布不完整；遮挡会导致标注噪声；部分 VLM 工具选择评价允许多个合理答案，不可等同真实部署成功率。
   - 对 wiki 的映射：实体「局限与风险」。

## 开源边界

- 项目页可访问可视化器、任务 CSV 与 Summary JSON；**未核实完整 58.3M 帧原始视频可批量下载**，也未发现官方完整训练实现 GitHub 仓。
- 论文注明研究数据非商业授权；须以实际数据发布页面条款为准。
- [网站来源归档](../sites/tooldex-project.md) 与 [ToolDex 实体](../../wiki/entities/paper-tooldex.md)。
