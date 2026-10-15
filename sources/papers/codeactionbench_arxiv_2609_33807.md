# CodeActionBench（arXiv:2609.33807）

> 来源归档（paper）

- **标题：** CodeActionBench: Evaluating Agentic Code-as-Policy for Embodied Manipulation
- **类型：** paper
- **作者：** Yiheng Lyu、Xueying Jiang、Wenhao Li、Shijian Lu、Gongjie Zhang
- **机构：** 南洋理工大学（NTU）；独立研究者
- **arXiv：** <https://arxiv.org/abs/2609.33807>
- **HTML：** <https://arxiv.org/html/2609.33807v1>
- **PDF：** <https://arxiv.org/pdf/2609.33807>
- **项目页：** <https://codeactionbench.org/>
- **代码：** <https://github.com/lyhkk/CodeActionBench>
- **发表日期：** 2026-09-27（arXiv v1）
- **入库日期：** 2026-10-15
- **再核日期：** 2026-10-15（项目页 + GitHub + arXiv）
- **一句话说明：** 用 25 个固定仿真操作任务评测通用多模态 agent 在没有特权场景状态、专用感知/抓取输出与预设策略时，能否通过可执行代码从视觉观察构造并修订机器人操作策略。

## 开源状态

- **部分开放**（2026-10-15 核查）：官方仓库已公开，包含 benchmark task、reference harness、robot tools、复现脚本和文档，代码采用 MIT 许可证；项目页提供演示与结果入口。README 标注完整轨迹、观察图像与视频数据集 **coming soon**，因此目前不能据此复现全部论文轨迹分析。

## 核心摘录

1. **问题定义：** 既有 Code-as-Policy 系统可能由外部模块提供物体位置、抓取候选或技能；本基准将任务相关视觉证据选择、3D 估计、操作目标构造、执行恢复和停止判断留给被测 agent。
2. **任务与环境：** 25 个 SAPIEN 仿真操作任务，建立在 RoboTwin 2.0 的 ALOHA-AgileX 双臂配置上，覆盖单臂/双臂抓取、搬运、交接、工具使用、排列堆叠、瞬时接触和铰接物体操作；每项使用固定场景实例。
3. **接口边界：** 提供多视角 RGB、相机标定几何计算、机器人/接触反馈与受限运动 API；不向 agent 暴露深度、精确物体姿态、任务目标、专用感知/抓取结果或 verifier 状态。agent 可直接调用工具，也可通过 Python 程序组合调用并根据反馈迭代。
4. **控制变量：** 固定任务实例、机器人接口、资源预算与隐藏物理结果 verifier；verifier 在 episode 结束后独立核验物理状态，且不把判定返回 agent。
5. **评测：** 9 种 agent 配置 × 25 项任务 × 3 次尝试，共 675 次；成功率为 2.7%–73.3%。GPT-6 Astra + Codex CLI 的 verifier 成功率为 73.3%，至少成功完成 22/25 项任务一次。
6. **可执行性分析：** 轨迹分析把空间检查点进度、最终物理成功与 agent 的完成判断分开；位置看似到位并不保证物体关系成立，失败集中暴露空间对齐、物体保持和完成判断问题。

**对 wiki 的映射**

- [CodeActionBench 论文实体](../../wiki/entities/paper-codeactionbench.md)
- [CodeActionBench 项目页](../sites/codeactionbench-org.md)
- [CodeActionBench 官方仓库](../repos/codeactionbench.md)
