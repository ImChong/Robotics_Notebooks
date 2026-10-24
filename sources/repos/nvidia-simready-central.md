# NVIDIA-Omniverse/simready-central

> 仓库归档

- **仓库：** https://github.com/NVIDIA-Omniverse/simready-central
- **类型：** repo（OEM SimReady 资产提交、审核与本地索引工具）
- **维护者：** NVIDIA Omniverse
- **许可证：** Apache-2.0（NVIDIA 编写代码；数据资产保留各自来源许可证）
- **入库日期：** 2026-10-24
- **一句话说明：** NVIDIA 用于接收并审核 OEM SimReady 资产提交的公开 GitHub 仓库；资产包实际托管于 Hugging Face 数据集。
- **相关项目：** [NVIDIA SimReady Central 资产库](../../wiki/entities/nvidia-simready-central.md)

## 仓库定位

该仓库不是 2,000 多个 3D 模型的代码仓或资产镜像。它提供 OEM 提交流程、资产包清单格式、校验/打包命令，以及把公开提交拉到本地并生成可筛选资产目录的示例。每条提交记录指向 Hugging Face 上具体数据集与固定版本的资产包。

## 使用与开放边界

- 仓库源码公开，Apache-2.0。
- OEM 提交前须完成 NVIDIA onboarding；仓库按 Hugging Face 数据集命名空间登记包清单，提交由 NVIDIA 审核。
- 浏览已发布资产可以按仓库文档把包拉到本地查看其来源、版本、许可、验证和 benchmark 信息。
- 仓库代码许可不覆盖其指向的 3D 资产；使用资产时逐包核对来源数据集的许可与限制。

## 对 wiki 的映射

- [NVIDIA SimReady Central](../../wiki/entities/nvidia-simready-central.md) — 资产入口、Agent 搜索/组场景流程与适用边界。
- [NVIDIA Physical AI 数据集](../../wiki/entities/nvidia-physical-ai-datasets.md) — NVIDIA 的更广泛 Hugging Face 数据集集合。

## 官方材料

- [README](https://github.com/NVIDIA-Omniverse/simready-central)
- [OEM 资产提交指南](https://github.com/NVIDIA-Omniverse/simready-central/blob/main/docs/oem-asset-submission-guide.md)
- [NVIDIA/Hugging Face SimReady Central 公告](https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories)
