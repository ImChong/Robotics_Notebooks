# NVIDIA SimReady Assets 数据集

> 来源归档

- **标题：** NVIDIA SimReady Assets
- **类型：** dataset（SimReady OpenUSD 3D 资产）
- **发布者：** NVIDIA
- **数据集：** https://huggingface.co/datasets/nvidia/simready-assets
- **所属发现体系：** [NVIDIA SimReady Central](https://huggingface.co/spaces/nvidia/simready-central)
- **规范与校验：** https://github.com/NVIDIA/simready-foundation
- **入库日期：** 2026-10-25
- **一句话说明：** NVIDIA 在 Hugging Face 发布的 SimReady 3D 资产数据集，可通过 SimReady Central 发现，用于 OpenUSD 场景中的机器人仿真与数字孪生工作流。

## 页面核实

- Hugging Face 数据集页展示 **2,575 个 3D Assets**；页面同时列出 **222 行** Dataset Viewer 数据和 **472 GB**总文件大小（核查日期：2026-10-25）。页面展示的资产数、Viewer 行数与总文件体积是不同统计项，不应混为同一口径。
- 页面预览包含工业设备、包装箱、推车、食品、厨房用具、防护用品和工具等资产。
- Hugging Face 当前页面显示 **No dataset card yet**。因此此归档仅记录数据集页面可见信息；质量、字段完整性、物理属性及来源须按单个资产和对应 SimReady profile 核验。
- SimReady 资产以 OpenUSD 为基础，目标是为相应仿真用例提供结构、仿真元数据和验证证据。碰撞几何、质量、摩擦、关节、语义标签、材质或传感器属性是否存在，取决于具体资产和 profile；不能据此宣称每个模型都包含全部物理属性。
- 数据集页未提供可读的数据集卡片来确认统一许可。下载或使用前应检查数据集文件、资产包元数据和各源资产的许可条款；NVIDIA SimReady Central 代码仓库的 Apache-2.0 许可不自动覆盖数据资产。

## Robotics_Notebooks 映射

- 生态与使用方式：[NVIDIA SimReady Central](../../wiki/entities/nvidia-simready-central.md)
- 官方资产提交流程：[NVIDIA-Omniverse/simready-central](../repos/nvidia-simready-central.md)

## 一手资料

- [Hugging Face 数据集页面](https://huggingface.co/datasets/nvidia/simready-assets)
- [NVIDIA/Hugging Face：SimReady Central 公告](https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories)
- [NVIDIA SimReady Central 文档](https://docs.omniverse.nvidia.com/simready/latest/simready-central.html)
- [NVIDIA SimReady Foundation 规范与校验工具](https://github.com/NVIDIA/simready-foundation)
