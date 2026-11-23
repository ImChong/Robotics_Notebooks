# NVIDIA SimReady Assets 数据集

> 来源归档

- **标题：** NVIDIA SimReady Assets
- **类型：** dataset（SimReady OpenUSD 3D 资产）
- **发布者：** NVIDIA
- **数据集：** https://huggingface.co/datasets/nvidia/simready-assets
- **所属发现体系：** [NVIDIA SimReady Central](https://huggingface.co/spaces/nvidia/simready-central)
- **规范与校验：** https://github.com/NVIDIA/simready-foundation
- **入库日期：** 2026-10-25
- **通读：** 已完成
- **本次复核：** 2026-11-21
- **一句话说明：** NVIDIA 在 Hugging Face 发布的 SimReady 3D 资产数据集，可通过 SimReady Central 发现，用于 OpenUSD 场景中的机器人仿真与数字孪生工作流。

## 页面核实

- 2026-10-25 历史快照：**2,575 个 3D Assets**、222 行 Dataset Viewer 数据、472 GB。**2026-11-21 新快照**：**2,919 个 3D Assets（USD）**、**518 GB**，Dataset Viewer 已禁用。两组为不同日期与口径，不能将 Viewer 行数视为资产数。
- 页面预览包含工业设备、包装箱、推车、食品、厨房用具、防护用品和工具等资产。
- Hugging Face 2026-11-21 页面已有简短 dataset card，标注 `cc-by-4.0`。该卡片仍不能证明每件资产的质量、关节、碰撞、物理行为均准确，应按单个资产与 SimReady profile 检查。
- SimReady 资产以 OpenUSD 为基础，目标是为相应仿真用例提供结构、仿真元数据和验证证据。碰撞几何、质量、摩擦、关节、语义标签、材质或传感器属性是否存在，取决于具体资产和 profile；不能据此宣称每个模型都包含全部物理属性。
- 数据集页面标注 **CC BY 4.0**，但单个包仍有独立许可证字段及资产来源记录，使用前核对一致性；NVIDIA SimReady Central 代码仓库的 Apache-2.0 不自动覆盖数据资产或其他 OEM 数据集。

## 下载量与传播口径（2026-11-21）

- Hugging Face [数据集页面](https://huggingface.co/datasets/nvidia/simready-assets)显示 **Downloads last month: 1,731,953**。这是平台定义的“近一个月下载次数”，非累计下载量、独立用户数或资产个数。
- 本次未取得公开历史趋势榜的原始排名快照，故不确认“发布两天登顶第一”。资产质量传闻没有指向可复核的 USD 版本和 issue，暂不列为已验证缺陷。

## Robotics_Notebooks 映射

- 生态与使用方式：[NVIDIA SimReady Central](../../wiki/entities/nvidia-simready-central.md)
- 官方资产提交流程：[NVIDIA-Omniverse/simready-central](../repos/nvidia-simready-central.md)

## 一手资料

- [Hugging Face 数据集页面](https://huggingface.co/datasets/nvidia/simready-assets)
- [NVIDIA/Hugging Face：SimReady Central 公告](https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories)
- [NVIDIA SimReady Central 文档](https://docs.omniverse.nvidia.com/simready/latest/simready-central.html)
- [NVIDIA SimReady Foundation 规范与校验工具](https://github.com/NVIDIA/simready-foundation)
