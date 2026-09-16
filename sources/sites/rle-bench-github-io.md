# rle-bench.github.io（RLE-Bench 项目页与 Leaderboard）

- **标题：** RLE-Bench: A Qualifying Exam for Coding Agents as Robot Learning Engineers
- **类型：** site / project-page / leaderboard
- **URL：** <https://rle-bench.github.io/>
- **博客：** <https://rle-bench.github.io/blog/> — [Introducing RLE-Bench](https://rle-bench.github.io/blog/)（2026-09-09 更新）
- **代码：** <https://github.com/RLE-Bench/RLE-Bench> — [`sources/repos/rle-bench.md`](../repos/rle-bench.md)
- **评测框架：** [Harbor](https://github.com/laude-institute/harbor)（Laude Institute）
- **机构：** 哈佛大学（Harvard University）、佐治亚理工学院（Georgia Tech）联合主导
- **入库日期：** 2026-09-16

## 一句话摘要

面向 **通用 coding agent** 的 **全栈机器人学习工程** 资格考：9 个任务族 × 4 条能力工作流（交互控制 / 策略开发 / 感知估计 / 机械设计），在 **物理仿真闭环** 中测 agent 能否观察—实验—改代码—再提交；**RLE Index** 四族等权 0–100 汇总。

## 开源核查（步骤 2.5，2026-09-16）

| 入口 | 结果 |
|------|------|
| 项目页 / Leaderboard | **已公开**：任务概览、RLE Index、分族分数、成本/耗时/上下文长度对比（需 JS 加载完整榜） |
| GitHub `RLE-Bench/RLE-Bench` | **已开源**（MIT）：9 个 `tasks/task0x`、Harbor 评测 CLI（`rlebench prepare/run/summarize/view`）、Docker 仿真栈 |
| Hugging Face `RLE-Bench/task05` | **部分数据**：T05 NanoVLA 相关数据集已发布；其余 task 以仓内生成资产为主 |
| arXiv / PDF | **截至入库日项目页未挂论文 PDF**；README 提供 `@misc{rlebench2026}` bibtex |

**判定：已开源（评测系统 + 任务规范可本地复现）；论文预印本待发布。**

## 公开要点（编译自首页 + 博客，2026-09-16）

### 四工作流 × 九任务

| 工作流 | 任务 | 一句话 |
|--------|------|--------|
| **Interactive Control** | T01 Agentic Control | RoboCasa 厨房五任务，agent-in-the-loop；L1/L2/L3 三档 harness |
| | T02 Harness Engineering | agent 自建感知/控制 harness，交给 **新 agent** 零样本解 held-out 任务 |
| | T03 Embodied Reasoning | 须 **交互采样** 才能回答的具身推理（非静态 VQA） |
| **Policy Development** | T04 Whole-Body Motion Tracking | 人形 motion tracking → 导出 ONNX；MuJoCo-Warp → MuJoCo-C hidden 扰动 |
| | T05 NanoVLA Recipe | LIBERO + RoboTwin 六轨 VLA **训练配方**；按 replay 预算交付 recipe |
| **Perception & Estimation** | T06 Pose Estimation | 非对称物体平面位姿；四传感/方法变体；CPU <10 Hz 罚则 |
| | T07 Bin Clearing | 视觉+力反馈闭环清 bin；hidden 八堆 clutter |
| **Mechanical Design** | T08 Mobile Base Design | Panda/UR5e/xArm7 共用移动底座 MJCF + 控制器 |
| | T09 Gravity Compensation for Gello | GELLO 主臂被动重力补偿 + 自适应前馈共设计 |

### Agent 开发环

`Build → Act → Observe → Revise`：开发在 **公开仿真** 中进行；提交后在 **hidden 物理条件**（场景/seed/embodiment/扰动）下独立打分。各 task 有 **时间 / 交互步数 / CPU·GPU** 预算；Harbor 环境默认 **断网**，模型 API 走 allowlist。

### RLE Index

- 单 task 归一化 0–100 → **工作流内平均** → **四工作流等权平均** = RLE Index。
- 博客初榜叙事（具体分数需 JS 榜或 results JSON）：
  - **感知/交互**：GPT-6 Astra 与 Claude Opus 5 差距最大（视觉 grounding 分化）。
  - **策略学习**：Astra / Opus 5 / GPT-5.6 Sol 更接近。
  - **机械设计**：最难；T08 案例可达 shelf/payload 满分仍 **静态/动态稳定性归零**。

### 团队（项目页 Footer）

- Core：Haitong Ma（Harvard）、Chenxiao Gao、Rushi Qiang（Georgia Tech）
- Advisors：Na Li（Harvard）、Bo Dai（Georgia Tech）

## 关联资料

- 官方博客：[Introducing RLE-Bench](../blogs/rle_bench_introducing_blog_2026-09-14.md)
- 代码归档：[rle-bench.md](../repos/rle-bench.md)
- 外部评论：[Walter Zhu X 长文](../blogs/walterzhu8_gpt6_astra_embodied_ai_2026-09-16.md)（Astra 与具身 AI 路线；与 RLE-Bench 上 Astra 演示互参）
- wiki：[RLE-Bench](../../wiki/entities/rle-bench.md)

## 对 wiki 的映射

- 实体页：[RLE-Bench](../../wiki/entities/rle-bench.md) — 四工作流、九任务、Harbor 复现、初榜读法
- 交叉：[ASPIRE](../../wiki/methods/aspire.md)、[ENPIRE](../../wiki/methods/enpire.md)（coding agent 机器人闭环）、[RoboCasa](../../wiki/entities/robocasa.md)（T01/T02）、[LIBERO benchmark](../../wiki/entities/libero-benchmark.md)（T05）、[具身评测基准选型闭环](../../wiki/overview/hub-embodied-eval-benchmark.md)
