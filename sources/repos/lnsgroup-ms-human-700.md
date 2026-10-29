# LNSGroup/MS-Human-700

> 来源归档

- **标题：** MS-Human-700: Whole-body Human Musculoskeletal Model
- **类型：** repo
- **链接：** <https://github.com/LNSGroup/MS-Human-700>
- **项目页：** <https://lnsgroup.cc/research/MS-Human-700>
- **论文：** [arXiv:2312.05473](../papers/ms_human_700_arxiv_2312_05473.md)
- **许可证：** Apache-2.0
- **入库日期：** 2026-10-27
- **一句话说明：** LNS Group 发布的 MuJoCo 肌骨人体模型 XML 与资产，包含全身、下肢运动和右手操作三个模型变体。
- **开源状态：** 已开源（模型 XML 与资产公开；以仓库 README 和许可证为准）
- **沉淀到 wiki：** 是 → [MS-Human-700](../../wiki/entities/ms-human-700.md)

## 模型入口

- MS-Human-700.xml：全身模型，90 个身体节段、206 个关节（约束后 85）、700 个肌肉执行器。
- MS-Human-700-Locomotion.xml：下肢运动模型，80 个身体节段、36 个关节、100 个肌肉执行器。
- MS-Human-700-Manipulation.xml：右臂与手操作模型，127 个身体节段、42 个关节、81 个肌肉执行器。

README 建议将 XML 文件拖入 MuJoCo simulate 查看器检查模型。
