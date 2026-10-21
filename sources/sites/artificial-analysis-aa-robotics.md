# AA-Robotics v0.1 — 官方文章归档

- **类型：** 官方博客 / 基准发布
- **发布：** 2026-10-14
- **项目：** [Artificial Analysis](https://artificialanalysis.ai/articles/aa-robotics)
- **代码：** 截至 2026-10-21，文章及其官方页面未列公开源码仓库；复现实验所需的 Harness/API 细节未完全公开。
- **说明：** 研究预览，报道首批 8 项桌面真机任务。

## 内容摘录

Artificial Analysis 将 AA-Robotics v0.1 定位为具身智能研究预览：让前沿语言模型在 coding-agent Harness 内通过简单 HTTP API 控制 6 轴机械臂（5 个关节与夹爪），考察理解、规划及物理执行。每个模型在隔离容器中得到简短说明，可查看桌面周围 3 台相机及腕部 1 台相机画面，并在容器中编写、运行控制代码。任务均为 zero-shot；任务结束后根据终态与通过标准评分，完整完成得 100，部分完成按进度计分。

首批任务覆盖按按钮、将玩具放入区域、倒饮料、整理积木、形状匹配、写字、搭塔、测量盒子。文章报告的平均分依次为 GPT-6 Astra 69.4、GPT-6.1 Sol 64.6、Gemini 4 Argon 55.3、Claude Opus 5.5 48.8、Claude Fable 5.1 46.3；正文将 Astra 分数四舍五入为 69。模型平均每项任务约 30 分钟，人工基线约 20 秒（约 100 倍差距）；Astra 平均每项耗时 30 分钟、成本 $16.36。各模型分别在 Codex、Claude Code、Antigravity 中以 high effort 运行。

文章将操纵灵巧度、空间推理与应对意外列为关注能力。错误示例包括空抓、抓取不稳、物体滑落、碰撞改变场景、未预判接触引发的连锁影响。该结果说明当前瓶颈并非单一“抓取”：精确控制、视觉/空间判断与策略调整相互耦合。

## 来源

- Artificial Analysis, [Welcome to the era of embodied AI: Introducing AA-Robotics v0.1](https://artificialanalysis.ai/articles/aa-robotics), 2026-10-14.
