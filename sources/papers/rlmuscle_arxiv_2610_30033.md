# RLMuscle: Muscle-Driven Character Animation via Reinforcement Learning

- **类型：** paper
- **链接：** <https://arxiv.org/abs/2610.30033>
- **入库日期：** 2026-10-28
- **机构：** 北京航空航天大学、西湖大学、伦敦大学学院

## 资料要点

- 强化学习控制器在一维 Hill 肌肉模型中学习参考动作跟踪激活，再用相同本构规律驱动三维肌肉变形。
- 论文报告 194 块肌肉的全身模型在 RTX 4090 上仿真 18 ms/帧；仓库注明该数字不含 USD 导出，不代表完整动画管线延迟。
- 官方仓库提供 Warp/XPBD 仿真示例与单独下载的资产；README 未明确提供论文完整 RL 训练复现入口，不能据此认定完整训练已发布。

## 对 wiki 的映射

- [论文实体](../../wiki/entities/paper-rlmuscle.md)：方法、实验及工程局限。
