# senlanke/mimic — G1 上的 SMP / CMoE / AME 移植（mjlab）

> 来源归档

- **标题：** SMP / CMoE / AME: Unitree G1 Motion-Control Reproductions and Ports
- **类型：** repo
- **链接：** https://github.com/senlanke/mimic
- **默认分支：** `master`
- **机构：** 清华大学 SUZ 课题组课程项目向移植（作者 GitHub: senlanke；SMP 同系 [SUZ-tsinghua/smp](https://github.com/SUZ-tsinghua/smp)）
- **入库日期：** 2026-08-25
- **复核日期：** 2026-08-29
- **一句话说明：** 在 **mjlab** 上把三条人形运动控制线接到 **Unitree G1**：SMP（完整）、CMoE（移植完成）、AME（未完成/未验证）。共享 `uv` 安装与 `scripts/train.py` / `play.py`。非任一上游官方实现。
- **许可：** 根目录无 SPDX；移植的 CMoE 代码保留上游 BSD-3-Clause（`LICENSES/CMoE.txt` + `NOTICE`）；其余须遵守各上游与数据集许可。
- **沉淀到 wiki：** 是 → [`wiki/entities/smp-g1-mjlab.md`](../../wiki/entities/smp-g1-mjlab.md)

---

## 开源状态（步骤 2.5，截至 2026-08-29）

| 项目 | 上游 | 本仓工作 | 状态 |
|------|------|----------|------|
| **SMP** | [SUZ-tsinghua/smp](https://github.com/SUZ-tsinghua/smp)（方法官方在 [MimicKit](https://github.com/xbpeng/MimicKit)） | 直接使用其 G1 运动特征、扩散先验与四类下游任务 | **Complete** |
| **CMoE** | [Hoshi-No-Ai/CMoE](https://github.com/Hoshi-No-Ai/CMoE)（README 仍写空仓 `Fudan-MAGIC-Lab/CMoE`） | 五专家复杂地形策略移植到 MuJoCo / mjlab，任务 `CMoE-G1` | **Port complete** |
| **AME** | [SII-FUSC/AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion) | Isaac Lab → mjlab 迁移中；见 `src/smp/rl/tasks/ame/MIGRATION.md` | **Incomplete / unverified** |

> README 明确：**AME is an unfinished task**。`MIGRATION.md` 写「No training, simulation or import validation was run」。不要把 `AME-G1*` 当可复现基线。

**权重：** SMP 内置三套 prior（`datasets/pretrain_ckpt/`）；CMoE **无**官方 checkpoint（需自训）；AME 尝试加载上游 `ame1.pt` / `ame2.pt`，行为等价**尚未验证**。2026-08-29 有 `add checkpoint` 提交，仍以 README 任务状态表为准。

---

## 与原始论文 / 参考实现的关系

| 资料 | 链接 | 角色 |
|------|------|------|
| SMP 论文 | [arXiv:2512.03028](https://arxiv.org/abs/2512.03028) | 方法定义（SDS、ESM、GSI） |
| SMP 项目页 | https://yxmu.foo/smp-page/ | [`sources/sites/smp-project.md`](../sites/smp-project.md) |
| MimicKit | https://github.com/xbpeng/MimicKit | 论文作者官方 SMP |
| CMoE 论文 / 官方代码 | [arXiv:2603.03067](https://arxiv.org/abs/2603.03067) · [Hoshi-No-Ai/CMoE](https://github.com/Hoshi-No-Ai/CMoE) | Isaac Gym 官方栈 — [`sources/repos/cmoe.md`](cmoe.md) |
| AME 论文 | [arXiv:2506.09588](https://arxiv.org/abs/2506.09588) | CNN+MHA 高程编码 |
| AME G1 复现（Isaac Lab） | https://github.com/SII-FUSC/AME_Locomotion | 本仓 AME 迁移源 |
| mjlab | https://github.com/mujocolab/mjlab | 共享 RL 环境骨架 |

本仓库是**课程项目向复现/移植**，不是 SMP / CMoE / AME 任一作者官方代码。

---

## 技术要点摘录

### 共享入口

- 包管理：`uv sync --frozen`（`uv.lock` 锁定 mjlab git rev；Python 3.13）
- 训练 / 回放：`uv run scripts/train.py <Task-ID>` / `uv run scripts/play.py <Task-ID>`
- 任务注册：`src/smp/rl/tasks/__init__.py` 副作用导入 `steering`（含 Forward）、`getup`、`location`、`cmoe`、`ame`

### 项目 1：SMP（完整）

三阶段：`scripts/csv_to_npz.py` → `scripts/compute_norm_stats.py` → `scripts/pretrain.py`（可跳过）→ PPO + 冻结 SDS。

| Checkpoint | 训练数据 | 默认任务 |
|------------|----------|----------|
| `pretrained_loco.pt` | walk / jog / run | `Smp-Forward-G1` |
| `pretrained_lafan_run.pt` | LAFAN run 子集 | `Smp-Steering-G1`、`Smp-Location-G1` |
| `pretrained_getup_f2s2.pt` | fall→stand get-up | `Smp-Getup-G1` |

**乘性奖励（相对 MimicKit 加性）：** `r = (Σ wᵢ taskᵢ) × r_smp`，`r_smp = exp(-w_s/|K| · Σ ‖ε̂−ε‖²)`。

**G1 运动特征 59 维/帧：** `root_pos(3) + root_rot(6) + joint_pos(29) + ee_pos(15) + root_lin_vel(3) + root_ang_vel(3)`。CSV 来自 [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) 的 `g1` split（30 FPS、36 列）。

### 项目 2：CMoE（移植完成，从零训）

任务 `CMoE-G1`；**不加载 SMP prior**。对齐官方核心：

- 12-DoF 下肢；10 帧本体历史；77 点地形高度扫描
- 非对称 actor / critic 观测；5 专家 + 状态/地形 estimator
- prototype 对比目标（`num_prototypes=32`，`temperature=0.2`）与 CMoE PPO
- 地形课程、域随机化、九类复杂地形；`play` 可切 `g1_cmoe_course_env_cfg(difficulty=0.5)` 沿 x 轴串场
- Runner：`smp.rl.cmoe.CMoERunner`；默认 `max_iterations=50_000`，`experiment_name=g1_cmoe`

MuJoCo 碰撞高程步长 10 cm、策略射线网格 5 cm（`MIGRATION.md` 与 AME 共用这条引擎边界）。

### 项目 3：AME（未完成）

意图：把 `SII-FUSC/AME_Locomotion` 的 Isaac Lab G1 复现迁到 mjlab。已搬入 CNN/MHA、33×21×3 高程、两阶段地形、AME PPO 与原 `model_state_dict` 布局；**端到端训练与行为对齐未做**。

| Task ID | 意图 | 状态 |
|---------|------|------|
| `AME-G1` | Stage-one | Unverified |
| `AME-G1-Global` | Stage-one + global-context | Unverified |
| `AME-G1-Finetune` | Stage-two，从 `.*_ame$` run resume | Unverified |

契约细节见 [`src/smp/rl/tasks/ame/MIGRATION.md`](https://github.com/senlanke/mimic/blob/master/src/smp/rl/tasks/ame/MIGRATION.md)。

---

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 本仓枢纽 | `wiki/entities/smp-g1-mjlab.md` |
| SMP 论文 / 方法 | `wiki/entities/paper-smp.md`、`wiki/methods/smp.md` |
| CMoE 官方 Isaac Gym | `wiki/entities/paper-cmoe.md`、`sources/repos/cmoe.md` |
| AME 论文 | `wiki/entities/paper-ame-attention-based-map-encoding.md` |
| MimicKit / mjlab | `wiki/entities/mimickit.md`、`wiki/entities/mjlab.md` |
