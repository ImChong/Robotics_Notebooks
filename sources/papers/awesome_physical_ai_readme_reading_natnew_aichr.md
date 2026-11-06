# awesome-physical-ai 双清单 README 通读

> 来源归档（原文通读 · 支撑 `wiki/overview/awesome-physical-ai-technology-map.md`）

- **标题：** awesome-physical-ai（natnew 与 aichr 两个同名仓库的 README）
- **类型：** repo / curated-list
- **原文：** <https://github.com/natnew/awesome-physical-ai>（raw README.md，735 行）· <https://github.com/aichr/awesome-physical-ai>（raw README.md，255 行）；2026-11 通过 raw.githubusercontent.com 获取全文
- **一句话说明：** 两份面向 Physical AI 的 Awesome 清单。natnew 偏工程与评测治理（14 个 canonical 类别 + 附录），aichr 偏模型/感知/硬件的按栈分组（VLA、3D 视觉、边缘推理、硬件等 12 组），技术地图页即以二者合并去重为基础。
- **核对范围：** 两份 README 全文均已读完（natnew：Get Started、14 类 + 附录各节、Contributing；aichr：全部分节与 Legend）。仅核对 README 自身的结构、分类与条目描述；未逐条打开清单内约 500 个外链，也未读 natnew 的 CONTRIBUTING.md 与 docs 站。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **natnew 结构：** 顶部声明 "Last updated 2026-11-1 · 257 entries across 14 canonical categories"（本站快照时为约 229 条，上游已增长）。Get Started 给出 5 步入门路径（Gymnasium CartPole → MuJoCo → LeRobot → OpenVLA → 深入模仿学习/Sim2Real/硬件/评测）和五条路线（Foundations、Robot Learning、Foundation Models、Evaluation & Safety、Hardware），并提示"无需硬件，先从仿真开始"。
- **natnew 14 类：** Simulators、Datasets、Benchmarks、Evaluation Methodology、Robotics Foundation Models、World Models、Manipulation、Locomotion、Sim-to-Real、Safety & Robustness、Governance & Policy、Production Patterns / Reference Architectures、Courses、Companies；每类约 15–25 条，条目格式为 `[名称](链接) — 一句描述`。
- **natnew 附录：** Books、Tutorials & Guides、Key Papers（23）、Survey Papers（10）、Hardware Platforms（30）、Conferences、Community、Newsletters & Blogs、People to Follow、Related Awesome Lists。Governance 类含 EU AI Act、UK AI Safety Institute 等法规/机构条目。
- **aichr 结构：** 12 个分组：Foundation Models (VLA)、3D Computer Vision、Datasets、Benchmarks、Simulators、Edge AI & Inference、Hardware & Actuation、Frameworks & Libraries、Robot Platforms、Research Labs、Courses & Tutorials、Community。
- **aichr 图例：** 📄 论文、🎬 演示视频、💻 开源代码、🤖 硬件验证（Hardware-Verified）。VLA 组分 Industry Models（GR00T、π0、π0.5、Gemini Robotics）与 Academic Papers（RT-1/RT-2、PaLM-E、Octo 等）。
- **两清单交集的主线：** 仿真器（Isaac Lab、MuJoCo、ManiSkill）、数据集（Open X-Embodiment）、基准（LIBERO）、LeRobot、VLA 模型均同时出现；aichr 独有 3D 视觉与边缘推理（Jetson、TensorRT、llama.cpp），natnew 独有评测方法学、治理政策、生产参考架构。
- **格式性质：** 两份 README 都只给"标题 + 外链 + 一行描述"，不含方法讲解或复现细节；aichr 采用 CC0 许可。

## 对 wiki 的映射

- [`wiki/overview/awesome-physical-ai-technology-map.md`](../../wiki/overview/awesome-physical-ai-technology-map.md)
- [`wiki/entities/awesome-physical-ai-natnew.md`](../../wiki/entities/awesome-physical-ai-natnew.md) · [`wiki/entities/awesome-physical-ai-aichr.md`](../../wiki/entities/awesome-physical-ai-aichr.md)
