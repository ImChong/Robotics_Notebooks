# NVIDIA Cosmos Lab：Interactive World Simulation 官方项目页

> 一手项目页归档；2026-11-12 查阅，避免把宣传中的“实时”理解为所有配置均达到目标视频帧率。

- **类型：** 官方项目 / 技术展示
- **英文标题：** Interactive World Simulation with Cosmos for Physical AI
- **中文标题：** 面向物理 AI 的 Cosmos 交互式世界仿真
- **发布主体：** NVIDIA Cosmos Lab
- **官方链接：** https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/
- **技术报告（2026-11-04）：** https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/technical-report.pdf
- **代码入口：** https://github.com/NVIDIA/cosmos
- **实现框架：** https://github.com/NVIDIA/cosmos-framework
- **模型集合：** https://huggingface.co/collections/nvidia/cosmos3-examples
- **核查日期：** 2026-11-12
- **核对范围：** 已浏览项目页全部公开可读取文本：标题、简介、共同后训练配方、三个演示模型区、示例/可控输入、FPS 与硬件说明、技术报告及四个权重的链接；网页中的动态视频/交互点云未逐帧验证，指标以作者自述标注。
- **通读：** 已完成
- **沉淀到 wiki：** [唯一实体页面](../../wiki/entities/paper-cosmos-interactive-world-simulation.md)

## 项目结构与链接

项目页以**三个交互仿真方向**组织演示：Sim-Bimanual（动作 → 未来视觉）、Sim-Transfer（空间控制 → RGB）、Sim-Auto（多视角驾驶 → 同步观测）。技术报告开放权重表与 NVIDIA 在 Hugging Face 的博文还列出第四个 **Transfer-Auto**（多相机离线双向生成），不能将“三个演示区”误读为“三个 checkpoint”。

| 官方模型卡 | 面向输入 | 典型输出 |
|---|---|---|
| [Sim-Bimanual](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual) | 机器人动作 / 相机导航，已有视觉历史 | 未来视觉观测，非动作决策 |
| [Sim-Transfer](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer) | 深度、语义分割、边缘或模糊视觉控制 + 文本 | 按几何控制生成 RGB 外观 |
| [Sim-Auto](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto) | 驾驶 WSM 场景控制与相机/ LiDAR 条件 | 逐块同步多视角驾驶视觉/点云 |
| [Transfer-Auto](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto) | 每相机的 WSM + 文本，支持多视角布局 | 离线双向生成最多 11 路相机 + LiDAR |

## 项目页的性能卡片（作者自测）

| 模型 | 页面展示的硬件/场景 | 页面列出的性能 |
|---|---|---|
| Sim-Bimanual | 1 张 GB300，480p，特定加速预设 | **48.1 FPS**、**166 ms** chunk latency |
| Sim-Transfer | 1 张 GB200，640×640 | **15.85 FPS**、**240 ms** latency |
| Sim-Auto | 16 张 GB200，7 路 480p RGB + LiDAR | **16.7 FPS** |

三种场景**不具可直接横向排名的可比性**，因为相机数、分辨率、加速配置、传感器模态和硬件不同。报告 §7.4 明确 Sim-Auto 的 7 路模式仍低于 30 FPS 视频目标，并测试了预取控制的运行路径。

## 开源核验与工程含义

- **已开放：** 官方技术报告、Cosmos 框架代码、四个 Hugging Face 检查点、模型卡和推理/后训练 cookbook/skill。
- **自建部分：** 驾驶团队仍需仿真状态更新、World Scenario Map（WSM）渲染与同步；机器人团队需实际本体动作编码/归一化、策略回路和任务级度量。
- **边界：** 视觉模型生成的是**可用于策略观察的仿真观测**，物理状态是否由外部引擎维护取决于模型类型；Demo 质量与特定计算配置不等同于已验证的现实车辆/机器人闭环安全性。

## 相关归档

- [技术报告及实测表格](../papers/cosmos_interactive_world_simulation_tech_report_2026.md)
- [NVIDIA 官方 Hugging Face 博文](../blogs/nvidia_cosmos_interactive_world_simulation_hf_blog.md)
- [四个模型卡归档](./hf_cosmos3_nano_sim_checkpoints_model_cards.md)
- [Cosmos Framework 源码归档](../repos/nvidia_cosmos_framework.md)
