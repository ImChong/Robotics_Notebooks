# fairmotion

> 来源归档（上游动捕数据工具；非机器人重定向器）

- **标题：** fairmotion
- **类型：** repo
- **链接：** https://github.com/facebookresearch/fairmotion
- **维护者：** Meta Research（原 Facebook AI Research）
- **License：** BSD-3-Clause
- **状态：** 2023-05 已归档（read-only）
- **入库日期：** 2026-06-08
- **一句话说明：** 通用动捕数据处理库（BVH / ASF-AMC / AMASS / AMASS-DIP 读写、3D 变换、FK、可视化与 motion_prediction 等示例任务）；常作人形重定向/运动生成管线的上游数据与骨架基础设施，本身不提供机器人重定向。
- **沉淀到 wiki：** 是 → [`wiki/entities/fairmotion.md`](../../wiki/entities/fairmotion.md)

## 生态位置

- 上游数据层：与 AMASS 同属人体运动数据/表示层，统一异构动捕格式。
- 下游：再接 [PHC](phc.md) / GMR 等做真正的人→机器人重定向；本仓不含机器人侧映射。
