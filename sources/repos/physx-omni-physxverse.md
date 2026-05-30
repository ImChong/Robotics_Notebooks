# PhysX-Omni / PhysXVerse（Hugging Face 数据集）

> 来源归档

- **标题：** PhysXVerse — general physics-grounded simulation-ready 3D dataset
- **类型：** repo / dataset（Hugging Face）
- **链接：** <https://huggingface.co/datasets/PhysX-Omni/PhysXVerse>
- **维护者：** PhysX-Omni 团队
- **入库日期：** 2026-05-30
- **一句话说明：** 首个 **通用 sim-ready 物理 3D** 公开集之一：在 **五维基础物理标注**（绝对尺度、材料、affordance、运动学、功能描述）上系统标注；整体结构与 **PhysXNet** 对齐；基于 **PartVerse** 并致谢其贡献者。
- **沉淀到 wiki：** 是 → [`wiki/entities/physx-omni.md`](../../wiki/entities/physx-omni.md)

## 数据集卡片要点（HF，2026-05-30）

- **规模（论文）：** **8.7K+** 高质量 sim-ready 3D 资产，**2.9K+** 语义类别（室内家具、无人机、机器人、车辆、大型场景组件等）；部件数 **1–65**。
- **标注流水线：** PartVerse 分割 → 多视角渲染 → **GPT 类 VLM 初标** → **人工校验**（尺度、affordance、材料、功能描述、运动学等，与 PhysXGen 叙事一致）。
- **存储：** 公开卡片显示总大小约 **113 GB**（以下载时 HF 元数据为准）。
- **训练配套：** 仓库 `dataset/` 下体素化、**64³ RLE 微调表征** 与条件图（每物体 **25 视角**）预处理脚本；格式模板见 PhysX-Anything 仓库 `training_data_template.json`。

## 相关数据集（训练常一并使用）

| 名称 | URL | 角色 |
|------|-----|------|
| PhysXNet | <https://huggingface.co/datasets/Caoza/PhysX-3D> | 前序 sim-ready 3D 语料 |
| PhysX-Mobility | <https://huggingface.co/datasets/Caoza/PhysX-Mobility> | 关节/可动部件专项 |
| PhysXVerse | <https://huggingface.co/datasets/PhysX-Omni/PhysXVerse> | 本文新构建、类别最广 |

## 对 wiki 的映射

- [`wiki/entities/physx-omni.md`](../../wiki/entities/physx-omni.md)
- [`sources/papers/physx_omni_arxiv_2605_21572.md`](../papers/physx_omni_arxiv_2605_21572.md)
- [`sources/repos/physx-omni.md`](physx-omni.md)
