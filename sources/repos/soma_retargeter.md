# SOMA Retargeter

> 来源归档

- **标题：** SOMA Retargeter
- **类型：** repo
- **链接：** https://github.com/NVIDIA/soma-retargeter
- **许可：** Apache-2.0
- **入库日期：** 2026-06-08
- **一句话说明：** NVIDIA SOMA 生态的 BVH→人形关节 CSV 重定向库：Newton + Warp GPU IK、人体比例缩放、足部稳定与关节限位；当前主打 Unitree G1（29 DoF）。
- **沉淀到 wiki：** 是 → [`wiki/entities/soma-retargeter.md`](../../wiki/entities/soma-retargeter.md)

## 摘录要点

- 输入：**SOMA 统一比例骨架** BVH；大规模数据见 HuggingFace [SEED](https://huggingface.co/datasets/bones-studio/seed)。
- 输出：机器人可播放 **CSV 关节轨迹**；提供交互 viewer 与 headless 批处理。
- 关联：[SOMA-X](https://github.com/NVlabs/SOMA-X)（wiki [`soma-x.md`](../../wiki/entities/soma-x.md)）、[GEM-X](https://github.com/NVlabs/GEM-X)、[Kimodo](https://github.com/nv-tlabs/kimodo)、[ProtoMotions](https://github.com/NVlabs/ProtoMotions)、[GMR](https://github.com/YanjieZe/GMR)。
- 状态：活跃开发中，API 可能变更。
