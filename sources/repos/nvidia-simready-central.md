# NVIDIA-Omniverse/simready-central

> 仓库归档

- **仓库：** https://github.com/NVIDIA-Omniverse/simready-central
- **类型：** repo（OEM SimReady 资产提交、审核与本地索引工具）
- **维护者：** NVIDIA Omniverse
- **许可证：** Apache-2.0（NVIDIA 编写代码；数据资产保留各自来源许可证）
- **入库日期：** 2026-10-24
- **通读：** 已完成
- **本次复核：** 2026-11-21
- **一句话说明：** NVIDIA 用于接收并审核 OEM SimReady 资产提交的公开 GitHub 仓库；资产包实际托管于 Hugging Face 数据集。
- **相关项目：** [NVIDIA SimReady Central 资产库](../../wiki/entities/nvidia-simready-central.md)

## 仓库定位

该仓库不是 2,000 多个 3D 模型的代码仓或资产镜像。它提供 OEM 提交流程、资产包清单格式、校验/打包命令，以及把公开提交拉到本地并生成可筛选资产目录的示例。每条提交记录指向 Hugging Face 上具体数据集与固定版本的资产包。

## 使用与开放边界

- 仓库源码公开，Apache-2.0。
- OEM 提交前须完成 NVIDIA onboarding；仓库按 Hugging Face 数据集命名空间登记包清单，提交由 NVIDIA 审核。
- 浏览已发布资产可以按仓库文档把包拉到本地查看其来源、版本、许可、验证和 benchmark 信息。
- 仓库代码许可不覆盖其指向的 3D 资产；使用资产时逐包核对来源数据集的许可与限制。

## 官方源码入口及运行边界

- **提交/评审：** `simready-manage submit` 按 profile 执行验证、打包、上传，再提交固定 HF commit 的 `submissions/<namespace>/<dataset>` 清单 PR；仓库 [README](https://github.com/NVIDIA-Omniverse/simready-central) 明确说明资产最终由 NVIDIA 审核。
- **本地浏览：** [index-to-library-guide.md](https://github.com/NVIDIA-Omniverse/simready-central/blob/main/docs/index-to-library-guide.md) 按 `simready-manage pull --no-convert` → `docs/examples/index_to_library.py` → `docs/examples/library_page.py` 生成 `manifest.csv`、`library.json`、`library.html`，对每件资产展示许可、validation 与 benchmark 结果。
- **校验依据：** [acceptance-criteria.md](https://github.com/NVIDIA-Omniverse/simready-central/blob/main/docs/acceptance-criteria.md) 分 `SUB.*`（提交格式）、`SCAN.*`（包扫描）和 `REVIEW.*`（人工资产审查）。例如 `REVIEW.002` 要求目标 profile 必选功能通过校验，`REVIEW.003` 要求附上 benchmark 结果，`REVIEW.006` 核对 Isaac Sim/PhysX 运行表现。
- **边界：** 仓库提供的是提交流程和目录示例，不包含 SimReady Central 后端检索服务的完整实现、全库 USD 文件或通用机器人训练代码。

## 对 wiki 的映射

- [NVIDIA SimReady Central](../../wiki/entities/nvidia-simready-central.md) — 资产入口、Agent 搜索/组场景流程与适用边界。
- [NVIDIA Physical AI 数据集](../../wiki/entities/nvidia-physical-ai-datasets.md) — NVIDIA 的更广泛 Hugging Face 数据集集合。

## 官方材料

- [README](https://github.com/NVIDIA-Omniverse/simready-central)
- [OEM 资产提交指南](https://github.com/NVIDIA-Omniverse/simready-central/blob/main/docs/oem-asset-submission-guide.md)
- [NVIDIA/Hugging Face SimReady Central 公告](https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories)
