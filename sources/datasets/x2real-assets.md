# X2Real Simulation Assets（Hugging Face 数据集）

- **类型：** dataset / simulation assets
- **数据集：** <https://huggingface.co/datasets/x-square-robot/x2real-assets>
- **发布方：** X Square Robot（自变量机器人，Hugging Face 组织页）
- **关联论文：** <https://arxiv.org/abs/2609.27449>
- **一句话说明：** X2Real 仿真用静态资产快照，包含机器人、物体、室内场景、材质、纹理、抓取标签与任务采样池；它不是机器人动作或轨迹数据集。

## 页面核实到的内容

- README 列出 **74,423 个文件**，按 **107 个 TAR 分片**发布，总体积约 **44.2 GB**；下载并恢复时还要预留额外空间，文档建议至少 100 GB 可用磁盘。
- 资产目录包括导航场景、benchmark 对象与背景、机器人 USD、URDF / STL 模型、抓取标签、任务对象配置和采样池。
- 页面列出 5 个评测板 / 设置、73 个 ID case：ArtiXon Arm-6A 桌面与挑战任务、Quanta X1 移动操作、ARX R5 桌面、Franka Panda-Hand 桌面。
- 页面明确说明此仓库只有静态仿真资产，不含 LeRobot episodes、观测、动作、视频或轨迹；仿真代码、任务定义、控制逻辑和策略权重另行提供。
- README 提供下载 / 校验脚本。元数据列出的 GitHub 项目仓库在本次 GitHub API 查询中返回 404；数据集卡中的代码许可仍标为 TODO，故不要推断代码已公开可复现或资产具备特定再分发许可。

## 和论文摘要中的“近 300 小时轨迹”区分

论文摘要提到近 300 小时标注仿真轨迹；这里的 x2real-assets 则是仿真场景及模型资产包。不要把 74,423 个静态资产文件误认为上述轨迹数据，也不要将资产数据集等同于完整基准实现。

## 沉淀到 wiki

- [X2Real 项目详情](../../wiki/entities/x2real-project.md)
- [X2Real 论文详情](../../wiki/entities/paper-x2real.md)
