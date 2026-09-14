# SteadyTray（UCSD 人形托盘平衡残差 RL）

> 来源归档

- **标题：** SteadyTray / ReST-RL
- **类型：** repo
- **来源：** UC San Diego（Michael Yip 组）
- **链接：** <https://github.com/AllenHuangGit/steadytray>
- **入库日期：** 2026-09-14
- **一句话说明：** SteadyTray 官方仓库：基于 IsaacLab fork 的四阶段 PPO 训练（行走预训练 → 端盘微调 → 残差教师 → 学生蒸馏），含预训练 checkpoint 与 MuJoCo sim2sim 部署。
- **沉淀到 wiki：** [`wiki/entities/paper-notebook-steadytray.md`](../../wiki/entities/paper-notebook-steadytray.md)

---

## 核心定位

**SteadyTray** 是 [arXiv:2603.10306](https://arxiv.org/abs/2603.10306) 的官方代码入口。方法 **ReST-RL** 在稳健 **base locomotion policy** 上挂 **残差模块**（Residual Action Adapter / Residual FiLM Adapter），专门抵消步态引起的托盘末端抖动；经四阶段课程在 **Unitree G1** 上实现 **96.9%** 变速跟踪成功率与 **74.5%** 外力扰动鲁棒性，并 **零样本 sim-to-real**。

---

## 仓库结构要点（README，2026-09-14）

| 路径 | 作用 |
|------|------|
| `source/steadytray/` | IsaacLab 扩展包（`pip install -e`） |
| `scripts/rsl_rl/train.py` | 四阶段训练入口 |
| `scripts/rsl_rl/play.py` | Isaac Sim 推理可视化 |
| `model/model_9999.pt` | 预训练学生策略 |
| `deploy/deploy_mujoco/` | MuJoCo sim2sim（`deploy_mujoco.py`） |
| `exported/` | 导出策略权重 |

**依赖：** 需先克隆并 Docker 挂载 [`IsaacLab_SteadyTray`](./isaaclab-steadytray.md) fork。

---

## 四阶段训练任务名

| Stage | Task | 说明 |
|-------|------|------|
| 1 | `G1-Steady-Tray-Pre-Locomotion` | 上身冻结的 base 行走 |
| 2 | `G1-Steady-Tray` | 端盘奖励微调 |
| 3 | `G1-Steady-Object` | 残差教师（特权观测） |
| 4 | `G1-Steady-Object-Distillation` | 蒸馏可部署学生 |

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [ResMimic](./resmimic.md) | 同为 **冻结/预训练基座 + 残差** 的人形 loco-manipulation 范式 |
| [unitree_rl_lab](../entities/unitree-rl-lab.md) | README 致谢的训练框架基座 |
| [Unitree G1](../../wiki/entities/unitree-g1.md) | 论文与仿真目标平台 |
