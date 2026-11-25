# UniRig 官方代码与发布物 | Source archive

- **类型：** repo
- **代码：** https://github.com/VAST-AI-Research/UniRig
- **固定提交：** [6793c66](https://github.com/VAST-AI-Research/UniRig/tree/6793c6640ff01c8fb389f3993434124bb43d2933)
- **代码许可：** MIT；训练数据和第三方素材仍须分别看条款。
- **模型：** [VAST-AI/UniRig model card](https://huggingface.co/VAST-AI/UniRig)
- **核对范围：** README 发布状态与推理步骤、官方模型卡及权重目录、推理 shell、任务配置、两套 Lightning 系统及 Writer、Blender 合并入口。静态核对，未运行 GPU 权重。
- **通读：** 已完成
- **入库日期：** 2026-11-25
- **对应节点：** [UniRig](../../wiki/entities/paper-unirig-2504-12451.md)

| 步骤 | 固定实现入口 | 说明 |
|---|---|---|
| 提取 | `launch/inference/extract.sh` → `src.data.extract` | Blender 导入网格或带骨架 FBX，提取推理 NPZ |
| 骨架 | `generate_skeleton.sh` → `run.py` → `ARSystem` / `ARWriter` | 自回归并反序列化骨架，输出骨架 FBX |
| 蒙皮 | `generate_skin.sh` → `run.py` → `SkinSystem` / `SkinWriter` | 读取带骨架输入，预测顶点权重并输出已蒙皮 FBX |
| 回写 | `merge.sh` → `src.inference.merge.transfer` | 需要时把源资产的骨架/权重转到原始目标网格；骨架-only 源不能补出蒙皮 |

维护者说明公开骨架与蒙皮权重来自 [Articulation-XL2.0](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0)，官方摘要列出 skeleton 与 skin 两个 checkpoint 目录。代码与 Rig-XL/VRoid 数据发布不等于对应论文主实验权重发布；README 仍将后者及物理辅助骨属性列为待发布。
