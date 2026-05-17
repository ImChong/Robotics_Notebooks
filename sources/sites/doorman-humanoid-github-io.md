# doorman-humanoid.github.io（DoorMan 项目页）

> 来源归档（ingest）

- **标题：** DoorMan — NVIDIA GEAR Lab
- **类型：** site / project-page
- **官方入口：** <https://doorman-humanoid.github.io/>
- **入库日期：** 2026-05-17
- **一句话说明：** DoorMan 论文配套站点：强调 **Isaac Lab 大规模物理+视觉随机化**、三阶段 **Teacher（PPO）→ Student（DAgger）→ GRPO 微调** 的叙述与算力标注，汇总真机泛化视频叙事、与人类遥操作对比及常见失败模式，并给出 BibTeX 与论文/HTML 版链接。

## 页面公开信息（检索自 2026-05-17）

| 资源 | URL |
|------|-----|
| 项目首页 | <https://doorman-humanoid.github.io/> |
| 论文 abs | <https://arxiv.org/abs/2512.01061> |
| 代码仓库 | <https://github.com/NVlabs/GR00T-VisualSim2Real> |

## 管线叙述摘录（与论文一致，便于 wiki 溯源）

1. **Mass Scale Simulation Randomization**：程序化铰接物体，随机化质量、把手类型、铰链阻尼/刚度、纹理、背景等。
2. **Teacher-Student-Bootstrap**：在经典特权蒸馏之上，**GRPO 微调** 用于部分可观测下的学生自举；站点给出粗略算力标注（Teacher **1×L40S、约 6 h**；DAgger **32×L40S、约 24 h**；GRPO **64×L40S、约 12 h**——以页面为准，复现以仓库配置为准）。
3. **Real-World Generalization**：多样把手几何、外观、地点与门型。
4. **相对人类遥操作**：宣称平均完成时间最多快约 **7.15 s**（与论文 **31.7%** 叙事一致，以论文表格为准）。
5. **Failure Cases**：未观测扰动、距离估计误差、未建模铰接状态；示例：卡在门框、距离判断失误。

## 站点披露的渲染工作流（附录级工程信息）

页面说明宣传视频基于 **Isaac Sim**：按 [IsaacLab 动画录制指南](https://isaac-sim.github.io/IsaacLab/main/source/how-to/record_animation.html) 导出 USD 动画 → Isaac Sim 交互回放 → [Animation Curve](https://docs.omniverse.nvidia.com/extensions/latest/ext_animation-motionpath.html) 飞镜 → [Movie Capture](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_movie-capture.html) 出片。

## 对 wiki 的映射

- [`wiki/entities/paper-doorman-opening-sim2real-door.md`](../../wiki/entities/paper-doorman-opening-sim2real-door.md) — 论文方法栈与实验归纳页。
- [`wiki/entities/gr00t-visual-sim2real.md`](../../wiki/entities/gr00t-visual-sim2real.md) — 与 VIRAL 并列的开源框架总览。

## BibTeX（站点页提供，便于引用）

站点给出的条目使用 `eprint={2512.01061}`、`year={2025}` 等字段；正式引用以 arXiv / 录用版本为准。
