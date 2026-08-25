# andireposit/Stand-Up-Motion-on-Compliant-Surface-for-Humanoid

> 来源归档

- **标题：** G1 软地面起身评测与权重
- **类型：** repo
- **代码：** <https://github.com/andireposit/Stand-Up-Motion-on-Compliant-Surface-for-Humanoid>
- **论文：** <https://arxiv.org/abs/2608.20852>
- **权重：** `model/g1_fallen_stand_soft_ground.zip` + `g1_fallen_stand_soft_ground_vecnormalize.pkl`
- **入库日期：** 2026-08-25
- **一句话说明：** IIT Kanpur G1 软地面起身：MuJoCo 评测、参考 CSV、已训 PPO 策略与网格资源；训练脚本未完整发布。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [G1 Compliant-Surface Stand-Up](../../wiki/entities/paper-g1-compliant-surface-standup.md) | 实体归纳页 |
| [HoST](../../wiki/entities/paper-host-humanoid-standingup.md) | 无参考多地形起身 |
| [Unitree G1](../../wiki/entities/unitree-g1.md) | 硬件平台 |
| [Balance Recovery](../../wiki/tasks/balance-recovery.md) | 任务语境 |

## 布局（README）

- `eval.py` — 加载策略并在 MuJoCo viewer 复现起身
- `g1.xml` — G1 模型与软接触参数
- `stand_up_lying_R_002__A475_new.csv` — 参考轨迹
- `model/*.zip` — 已发布 checkpoint
- `media/` — 论文图与演示视频

## 开源状态

| 组件 | 状态 |
|------|------|
| 评测 + MuJoCo 资产 + 参考动作 | **已发布** |
| 软地训练权重 | **已发布** |
| 完整 PPO 训练管线 | **未在仓库发布**（截至 2026-08-25） |
