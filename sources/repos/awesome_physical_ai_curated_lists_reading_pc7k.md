# awesome-physical-ai（natnew / aichr）两份 README 通读

> 来源归档（原文通读 · 支撑 `wiki/comparisons/awesome-physical-ai-curated-lists.md`）

- **标题：** awesome-physical-ai（两个同名独立仓库）
- **类型：** repo / curated-list
- **原文：** <https://github.com/natnew/awesome-physical-ai>（README.md，main，735 行）；<https://github.com/aichr/awesome-physical-ai>（README.md，master，255 行）
- **一句话说明：** natnew 版是 14 个 canonical 类别 + 附录的工程导向资源地图；aichr 版是 ~12 节的扁平 CC0 索引，偏 VLA / 3D 视觉 / 边缘硬件。
- **核对范围：** 2026-11 通过 raw.githubusercontent.com 取得两份 README 全文并逐节通读：aichr 全部章节（Foundation Models 至 Community / Legend / CC0 页脚）；natnew 的 Get Started、Contents、14 个 canonical 类别全部条目、全部附录（Books 至 People to Follow）、Related Awesome Lists 与 Contributing。条目行较长时只读到条目名与前半句描述，行末长描述与 `<!-- tags -->` 注释未逐字核对；未打开各条目的外链目标，也未读 CONTRIBUTING.md、docs 站与 CI workflow。
- **通读：** 已完成

## 核心摘录（均出自 README）

- **natnew 版结构：** 顶部徽章（stars、license、link-check 工作流）与 Get Started 折叠区；Quick start 五步为 Gymnasium CartPole、MuJoCo、LeRobot、OpenVLA、再深入（模仿学习 / sim-to-real / 硬件 / 评测）；另给 Foundations / Robot Learning / Foundation Models / Evaluation & Safety / Hardware 五条路线，并提示"无需硬件，先从仿真开始"。
- **14 个 canonical 类别（Contents）：** Simulators、Datasets、Benchmarks、Evaluation Methodology、Robotics Foundation Models、World Models、Manipulation、Locomotion、Sim-to-Real、Safety & Robustness、Governance & Policy、Production Patterns / Reference Architectures、Courses、Companies；附录 Books、Tutorials & Guides、Key Papers、Survey Papers、Hardware Platforms、Conferences、Community、Newsletters & Blogs、People to Follow、Related Awesome Lists、Contributing。
- **natnew 条目风格：** 每条为"链接 — 一句描述"，多数条目下带 `<!-- tags: paper / tool / open-source / benchmark … -->` 注释；Safety 节含 Safety Gym、CPO、Control Barrier Functions、SafeVLA、BadVLA 等；Governance 节含 NIST AI RMF、EU AI Act、ISO 10218 / 13482、UL 4600 等标准；Production 节以 ROS 2 / DDS / MCAP / Zenoh 等中间件为主。
- **natnew 的 Related Awesome Lists：** 列出 Awesome LLM Robotics、Awesome Robotics、Awesome World Models、Awesome Imitation Learning 等相邻清单，说明它自视为入口而非终点。
- **aichr 版结构：** Contents 含 Foundation Models (VLA)（Industry Models：GR00T、π0、π0.5、Gemini Robotics；Academic：RT-2、RT-1、PaLM-E、Octo 等）、3D Computer Vision（3DGS、Depth Anything、SAM 系）、Datasets、Benchmarks、Simulators、Edge AI & Inference（TensorRT-LLM、Jetson、vLLM、llama.cpp）、Hardware & Actuation（触觉、EtherCAT、ROS 2 Control、计算平台）、Frameworks & Libraries、Robot Platforms、Research Labs、Courses & Tutorials、Community。
- **aichr 条目风格：** 以 Legend 标注 📄 Paper / 🎬 Demo / 💻 Code / 🤖 Hardware-Verified；每节条目数较少，无 staged 学习路径，无独立文档站；页脚为 CC0 徽章，末尾指向 CONTRIBUTING.md。
- **两份重叠面：** 都收 MuJoCo、Isaac Lab、Genesis、PyBullet、Habitat、SAPIEN、Webots、LeRobot、Open X-Embodiment、BridgeData V2、CALVIN、ManiSkill、LIBERO、π0 系、RT-1/RT-2、Octo、PaLM-E、Mobile ALOHA、DayDreamer 等；natnew 独有 Safety / Governance / Evaluation / World Models / Locomotion 等类别，aichr 独有 3D 视觉、边缘推理与量化、触觉 / 执行器、研究实验室、社区论坛。

## 对 wiki 的映射

- [`wiki/comparisons/awesome-physical-ai-curated-lists.md`](../../wiki/comparisons/awesome-physical-ai-curated-lists.md)
- [`wiki/entities/awesome-physical-ai-natnew.md`](../../wiki/entities/awesome-physical-ai-natnew.md)
- [`wiki/entities/awesome-physical-ai-aichr.md`](../../wiki/entities/awesome-physical-ai-aichr.md)

## 参考来源

- natnew：<https://github.com/natnew/awesome-physical-ai>
- aichr：<https://github.com/aichr/awesome-physical-ai>
