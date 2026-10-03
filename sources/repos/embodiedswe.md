# EmbodiedSWE / CoSiGen

- **标题：** EmbodiedSWE — Coding Agents for Long-Horizon Dexterous Robotics
- **类型：** repo
- **来源：** EmbodiedSWE organization；项目团队包含 ByteDance Seed、Yale、Princeton、Carnegie Mellon University、Stanford、UCLA、University of Washington 等机构成员
- **链接：** <https://github.com/EmbodiedSWE/EmbodiedSWE>
- **项目页：** <https://embodiedswe.github.io/>
- **论文：** <https://arxiv.org/abs/2609.27308>
- **资产数据集：** <https://huggingface.co/datasets/EmbodiedSWE/robobench-assets>
- **入库日期：** 2026-10-03
- **一句话说明：** 基于 Isaac Lab 的长时程灵巧机器人任务 benchmark 与 coding-agent harness，并提供将通过验证的 agent 解扩增为机器人学习示范的 EmbodiedSWE-Gen。
- **沉淀到 wiki：** 是 → [论文实体页](../../wiki/entities/paper-embodiedswe.md)

## 开源状态核查（2026-10-03）

| 项 | 值 |
|----|-----|
| **开放程度** | **已开源** — 官方 GitHub 公共仓库含 benchmark 环境、运行与评测代码；二进制仿真资产单独托管于 Hugging Face |
| **代码许可证** | Apache-2.0（仓库 `LICENSE`） |
| **资产许可证** | 数据集卡整体声明 Apache-2.0；作者制作资产适用该许可，第三方资产沿用上游许可，包含部分 CC BY-NC / CC BY-NC-SA 条目 |
| 默认分支 | `main` |
| 仓库说明 | 仍在积极开发，README 提醒目录、接口及设置可能变化 |
| 仿真环境 | Isaac Sim 5.1、Isaac Lab 2.3.2；Linux、NVIDIA GPU、CUDA 12.x 驱动、uv |
| 论文任务规模 | 28 项长时程任务、6 类任务套件；论文正文描述 5 种 embodiment |
| 当前项目页规模 | 项目页当前列出 17 种 embodiment 配置（包含机器人/控制变体，数字口径与论文摘要的机器人种类不同） |

## 仓库结构与运行入口

| 路径 / 入口 | 用途 |
|-------------|------|
| `robobench/` | 环境 API、机器人/场景/控制器组合、任务套件与资产 manifest |
| `eval/` | coding agent 隔离运行、任务提交、离线 grading |
| `scripts/bootstrap_isaaclab_5_1.sh` | 安装仿真依赖并准备 robobench 环境 |
| `python -m robobench.scripts.smoke --list` | 列出已注册任务 |
| `python -m robobench.scripts.fetch_assets` | 获取按目录打包的仿真资产；`--check` 校验本地文件 |
| Hugging Face `robobench-assets` | USD / mesh / PBR 纹理 / locomotion policy / 演示视频等二进制资产，按 manifest 中 SHA-256 校验 |

## 对机器人研究与工程的价值

- **Agent-native benchmark：** agent 读任务说明，利用通用仿真状态、渲染/检查工具和 IK 或 operational-space controller，自行写出 `solve(env)` 程序；任务完成由独立 grader 在新环境离线评估，降低直接改状态或钻评分规则的空间。
- **从代码解到示范：** coding agent 找到一个可运行解后，数据管线对场景、策略、阶段、动力学和视觉变化进行分层扩增，获得更大规模的轨迹用于 VLA 微调。
- **Agent 改进闭环：** 仓库还包含从已有任务变体生成新任务并依据验证结果改进 coding agent 的方向。
- **实机证据：** 论文报告仅用 agent 生成的仿真示范微调 VLA 后，完成四阶段灯具拆解任务；这证明了一个任务级 sim-to-real 结果，不代表广泛的实机泛化。

## 复现注意事项

1. 初始化需安装 Isaac Sim / Isaac Lab，并接受 Omniverse EULA；依赖与硬件要求较高。
2. 仓库把较大的二进制场景资产放在 Hugging Face；需通过官方脚本下载及校验。数据集卡为 7.21 GB 的资产集合。
3. 数据集含多种上游模型与扫描素材；其中标注为 NonCommercial 的条目不可用于商业用途。引用或再分发时应逐项核实 NOTICE 与上游许可。
4. 仓库处于活跃开发状态，应按当前 README 和 commit 固定复现环境，不要依赖旧路径。

## 对 wiki 的映射

- 论文实体页：[EmbodiedSWE](../../wiki/entities/paper-embodiedswe.md)
- 相关 coding-agent 论文：[Agentic Coding Agent](../../wiki/entities/paper-agentic-coding-manipulation.md)
- 操作任务：[Manipulation](../../wiki/tasks/manipulation.md)
- 项目页：[EmbodiedSWE project page](../sites/embodiedswe-github-io.md)

## 参考链接

- GitHub：<https://github.com/EmbodiedSWE/EmbodiedSWE>
- 项目页：<https://embodiedswe.github.io/>
- 论文：<https://arxiv.org/abs/2609.27308>
- 仿真资产：<https://huggingface.co/datasets/EmbodiedSWE/robobench-assets>
- 数据集许可证与来源表：<https://huggingface.co/datasets/EmbodiedSWE/robobench-assets>
