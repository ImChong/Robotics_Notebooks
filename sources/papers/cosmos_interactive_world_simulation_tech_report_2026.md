# Interactive World Simulation with Cosmos for Physical AI（技术报告）

## 元数据

- **题目：** Interactive World Simulation with Cosmos for Physical AI
- **机构：** NVIDIA（完整作者与致谢列于报告附录 A，本次未逐一核对）
- **日期：** 2026-10-21（报告首页）
- **报告 PDF：** https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/technical-report.pdf（约 30 MB；经 curl 下载并用 pdftotext 读取前若干页）
- **arXiv：** 报告未给出 arXiv 编号（未核实是否另有 arXiv 版）
- **项目页：** https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/
- **官方代码：** https://github.com/nvidia/cosmos；https://github.com/nvidia/cosmos-framework（报告列出）
- **模型合集：** https://huggingface.co/collections/nvidia/cosmos3-examples
- **入库日期：** 2026-10-22

## 报告要解决的问题（据摘要）

把 Cosmos 世界基础模型改造成可实时交互的世界仿真：视觉观测随动作与场景控制流式生成，使用户与具身智能体能与演化中的虚拟环境交互。既可预测动作的视觉后果，也可由外部仿真器维护的物理状态生成观测。

## 覆盖的场景（据摘要）

- 相机导航与机器人操作（交互式视频预测）
- 可控视频迁移（sim-to-real 视觉迁移）
- 多视角自动驾驶（AV）仿真

## 开源模型（报告列出）

| 模型 | 用途（据项目页与模型卡） |
|------|------|
| Cosmos3-Nano-Sim-Bimanual | 相机导航与机器人动作条件的视觉动力学预测（双臂，AgiBot） |
| Cosmos3-Nano-Sim-Transfer | 深度、分割、边缘、模糊图等空间控制 → RGB 视频，文本控制外观 |
| Cosmos3-Nano-Sim-Auto | 多视角驾驶视频生成（项目页与报告使用此名） |
| Cosmos3-Nano-Transfer-Auto | 同一驾驶方向的 Hugging Face 模型卡名称（报告也列出此名） |

## 模型卡信息（据 Hugging Face 模型卡，WebFetch 摘要）

- 许可：OpenMDW 1.1（模型卡称可商用与非商用）。
- 架构：Mixture-of-Transformers，由 Cosmos3-Nano 经后训练与因果蒸馏得到；模型卡列 16B 可训练参数，HF 元数据约 15B（BF16）。
- Bimanual：约 13 亿数据点（393 个数据集条目）；仅列 AgiBot 作为 embodiment；需要其他本体另行后训练。
- Transfer：约 24.5 fps（GB200，480p）；控制 fidelity 与 Cosmos3-Nano 相比有提升，但 DOVER 与 prompt following 略低。
- Transfer-Auto：支持 1–11 路相机与 128 线 LiDAR（Hyperion 8.1 相机组）；模型卡示例关闭了 guardrails，部署前需自行加安全措施。

## Robotics_Notebooks 映射

- 实体页：[Cosmos Interactive World Simulation](../../wiki/entities/paper-cosmos-interactive-world-simulation.md)
- 基座：[Cosmos 3](../../wiki/entities/cosmos-3.md)（arXiv 2606.02800，另一份技术报告）

## 一手来源

- 技术报告 PDF（前 2 页与目录；完整报告未逐页阅读）
- 项目页与 HF 模型卡（经 WebFetch 摘要）
