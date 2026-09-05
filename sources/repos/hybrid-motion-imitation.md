# Hybrid-Motion-Imitation（G1 非官方 HIL / GfR 扩展）

> 来源归档

- **标题：** Hybrid Motion Imitation
- **类型：** repo
- **链接：** <https://github.com/jiashunwang/Hybrid-Motion-Imitation>
- **作者：** Jiashun Wang（HIL / GfR 一作）
- **许可：** Apache-2.0
- **语言：** Python
- **入库日期：** 2026-09-05
- **一句话说明：** 基于 [Holosoma](https://github.com/amazon-far/holosoma) 的 **非官方** 实现与扩展，把 HIL / GfR 的「actor 不见逐步参考、track + gen 并行」配方接到 29-DoF Unitree G1 的 **箱攀** 与 **搬箱**。
- **沉淀到 wiki：** [`wiki/entities/paper-hil-hybrid-imitation-learning.md`](../../wiki/entities/paper-hil-hybrid-imitation-learning.md)

---

## 开源边界（务必读）

README 原文：*This is an unofficial implementation and extension of GfR and HIL … It is not the official code release for either paper.*

| 对象 | 状态 |
|------|------|
| TOG HIL 的 SMPL 角色动画 / Isaac Gym 官方代码 | **未发布** |
| RSS GfR / MTRG 官方代码 | **未在本仓** |
| 本仓 G1 climb / object 两阶段训练 + Viser 评测 | **可运行**（需 Isaac Sim + 运动资产） |

不要把本仓写成「HIL 官方代码」。

---

## 仓库入口（README / `docs/wbt-hybrid.md`）

| 组件 | 说明 |
|------|------|
| 安装 | `bash scripts/setup_isaacsim.sh`；`source scripts/source_isaacsim_setup.sh`；`wandb login` |
| 实验别名 | `exp:g1-29dof-wbt-hybrid-climb`、`exp:g1-29dof-wbt-hybrid-object` |
| 训练 | `python src/holosoma/holosoma/train_agent.py`；脚本副本 `scripts_run/run_train.sh` |
| 评测 | `python src/holosoma/holosoma/eval_agent.py` + Viser `:8012`；`scripts_run/run_play.sh` |
| 方法说明 | [`docs/wbt-hybrid.md`](https://github.com/jiashunwang/Hybrid-Motion-Imitation/blob/main/docs/wbt-hybrid.md) |
| 运动目录 | climb：`holosoma/data/motions/g1_29dof/whole_body_tracking/climbox1`；object：`…/movebox` |
| 默认并行 | 8192 envs |

**两阶段：** Stage 1 `track-ratio=1.0` 只跟参考；Stage 2 resume 后 `track-ratio=0.5`，并重置 `init-noise-std=0.45`。Object Stage 2 另开 `hybrid-gen-noise-curriculum`。

**Hybrid 要点（相对 TOG 论文）：** actor 仍不见逐步参考；critic 吃特权 tracking 信号；本仓 **不用 AMP 判别器**，gen 侧是稀疏 goal bonus（climb 2.0 / object 3.0）。设计更贴近 [GfR/MTRG](../../wiki/methods/mtrg-reference-goal-driven-rl.md) 的「参考塑形 + goal 泛化」，同时复用 HIL「统一观测、先 track 再混合」的日程。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-hil-hybrid-imitation-learning](../../wiki/entities/paper-hil-hybrid-imitation-learning.md) | TOG 论文实体；本仓是 G1 非官方延伸 |
| [HIL 方法页](../../wiki/methods/hil-hybrid-imitation-learning.md) | 仿真角色 tracking + AMP |
| [MTRG / GfR](../../wiki/methods/mtrg-reference-goal-driven-rl.md) | 同作者 RSS 2026 人形配方 |
| [holosoma](../../wiki/entities/holosoma.md) | 上游 Amazon FAR 训练栈 |
| [Unitree G1](../../wiki/entities/unitree-g1.md) | 29-DoF 实验机体 |

## 参考链接

- 仓库：<https://github.com/jiashunwang/Hybrid-Motion-Imitation>
- HIL 项目页：<https://jiashunwang.github.io/HIL/>
- GfR 项目页：<https://jiashunwang.github.io/GfR/>
- Holosoma 上游：<https://github.com/amazon-far/holosoma>
