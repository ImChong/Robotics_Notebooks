# Jiaxin-Lu/humoto 官方代码仓库核对归档

> 来源归档（GitHub 源码实体核对）

- **仓库：** <https://github.com/Jiaxin-Lu/humoto>
- **类型：** repo
- **许可证：** 仓库根目录未包含独立 `LICENSE` 文件；数据与公开子集/完整版使用授权遵循官方数据页 <https://adobe-research.github.io/humoto/> 规定
- **核对范围：** 核对 `README.md`、`requirements.txt`、`render_humoto_pytorch3d.py`、`human_model/bone_names.py`、`human_model/human_model.py`、`utils/load_humoto.py` 与 `scripts/extract_pk_data.py`。
- **通读：** 已完成
- **入库日期：** 2026-11-14

## 源码架构与可运行管线核对

1. **人体与骨骼模型（`human_model/`）**：`bone_names.py` 定义 65 个 Mixamo 标准骨骼名称；`human_model.py` 提供 `HumanModelDifferentiable` PyTorch 可微正运动学模型，用于从关节旋转、局部/全局位姿还原三维网格顶点与骨骼关节点。
2. **数据加载接口（`utils/load_humoto.py`）**：`load_one_humoto_sequence` 加载单个序列化序列的人体姿态、物体 6-DoF 轨迹、网格资产及配套多级文本标注。
3. **数据导出与 Blender 前处理管线（`scripts/extract_pk_data.py` 与 `README.md`）**：依赖 `bpy==4.0.0` 从清理后的 `.blend` 工程依次执行尺度归一化、骨架迁移与 pickle 打包导出（对应 `clear_human_scale.py` -> `transfer_human_model.py` -> `extract_pk_data.py` -> `copy_text.py` 工作流）。
4. **端到端 PyTorch3D 可视化入口（`render_humoto_pytorch3d.py`）**：读取序列字典，运行 `HumanModelDifferentiable` 还原人-物同步 4D 网格序列并借助 PyTorch3D 渲染验证。仓库同时说明公开基础子集托管于 `adobe-research/humoto`，完整版需经官方途径联系作者获取。

## 对 wiki 的映射

- [paper-notebook-humoto-a-4d-dataset-of-mocap-human-object-intera](../../wiki/entities/paper-notebook-humoto-a-4d-dataset-of-mocap-human-object-intera.md)
