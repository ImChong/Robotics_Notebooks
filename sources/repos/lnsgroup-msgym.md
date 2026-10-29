# LNSGroup/msgym

> 来源归档

- **标题：** msgym: Gymnasium Environments for MusculoSkeletal Models
- **类型：** repo
- **链接：** <https://github.com/LNSGroup/msgym>
- **模型仓库：** <https://github.com/LNSGroup/MS-Human-700>
- **项目页：** <https://lnsgroup.cc/research/MS-Human-700>
- **论文：** [arXiv:2312.05473](../papers/ms_human_700_arxiv_2312_05473.md)
- **许可证：** Apache-2.0
- **入库日期：** 2026-10-27
- **一句话说明：** 为 MS-Human-700 提供 MuJoCo/Gymnasium 运动与操作环境，以及 DynSyn-SAC 训练入口。
- **开源状态：** 已开源（环境与训练脚本公开；README 指向 GitHub Releases 中的训练 checkpoint）
- **沉淀到 wiki：** 是 → [MS-Human-700](../../wiki/entities/ms-human-700.md)

## 环境入口

- msgym/LocomotionFullEnv-v1：全身运动模仿。
- msgym/LocomotionLegsEnv-v1：仅腿部运动模仿。
- msgym/ManipulationEnv-v1：右臂抓取与抬升。
- README 提供 DynSyn-SAC/SB3-Scripts/train.py 与 eval.py 作为训练、评估入口。MuJoCo 模型通过 git submodule 引入，克隆时需带 --recursive。
