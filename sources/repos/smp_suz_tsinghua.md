# SMP on G1（SUZ-tsinghua/smp）

> 来源归档

- **标题：** SMP — Score-Matching Motion Priors (G1 reproduction on mjlab)
- **类型：** repo
- **机构：** 清华大学 SUZ 课题组（GitHub org: SUZ-tsinghua）
- **链接：** https://github.com/SUZ-tsinghua/smp
- **入库日期：** 2026-05-27
- **一句话说明：** 在 MimicKit 原版未提供 G1 配置的前提下，于 **mjlab** 上端到端复现 SMP（DDPM 预训练 + 冻结 SDS 引导奖励 + PPO），覆盖 Unitree G1 四类下游任务，并内置三套可跳过预训练的 prior checkpoint。
- **沉淀到 wiki：** 是 → [`wiki/entities/smp-g1-mjlab.md`](../../wiki/entities/smp-g1-mjlab.md)

---

## 与原始论文 / 参考实现的关系

| 资料 | 链接 | 角色 |
|------|------|------|
| SMP 论文 | [arXiv:2512.03028](https://arxiv.org/abs/2512.03028) | 方法定义（SDS、ESM、GSI、可复用先验） |
| 原论文项目页 | https://yxmu.foo/smp-page/ | 官方说明与结果 |
| MimicKit | https://github.com/xbpeng/MimicKit | 原版 SMP 实现（`docs/README_SMP.md`） |
| mjlab | https://github.com/mujocolab/mjlab | 本复现的 RL 环境骨架（`ManagerBasedRlEnv`） |

本仓库为**课程项目向的复现**，非论文作者官方代码；主要工程贡献是 **G1 运动特征、先验、任务与奖励** 的完整移植，以及 **任务奖励 × SMP 引导** 的乘性组合（见下）。

---

## 技术要点摘录

### 管线三阶段

1. **数据处理**：CSV → 窗口化 NPZ → 归一化统计（脚本：`csv_to_npz.py`、`compute_norm_stats.py`）。
2. **扩散预训练**：小 DDPM ε-预测器在 motion window 上训练（`pretrain.py`）；可跳过，使用 `datasets/pretrain_ckpt/` 内三套权重。
3. **RL**：PPO + 冻结 prior 的 SDS 风格 `r_smp`（`scripts/train.py` / `play.py`）。

### 预置 prior（免预训练）

| Checkpoint | 训练数据 | 默认任务 |
|------------|----------|----------|
| `pretrained_loco.pt` | walk / jog / run | `Smp-Forward-G1` |
| `pretrained_lafan_run.pt` | LAFAN run 子集 | `Smp-Steering-G1`、`Smp-Location-G1` |
| `pretrained_getup_f2s2.pt` | fall→stand get-up | `Smp-Getup-G1` |

### 四类下游任务（`mjlab.tasks.registry`）

| Task ID | 说明 |
|---------|------|
| `Smp-Forward-G1` | 固定 +x 朝向，速度 0.5–5 m/s 前进 |
| `Smp-Steering-G1` | 速度 + 朝向跟踪 |
| `Smp-Location-G1` | 世界系 xy 目标点 |
| `Smp-Getup-G1` | 跌倒姿态起身（GSI 初始化） |

### 与 MimicKit 的关键差异：乘性奖励

- **原版（加性）：** `r = w_task · task + w_smp · r_smp`
- **本复现（乘性）：** `r = (Σ wᵢ taskᵢ) × r_smp`，`r_smp = exp(-w_s/|K| · Σ ‖ε̂−ε‖²)`

动机：无需手调 task/smp 权重比；避免「只刷任务」或「只刷自然度」的单边最优。

### G1 运动特征（59 维/帧）

`root_pos(3) + root_rot(6) + joint_pos(29) + ee_pos(15) + root_lin_vel(3) + root_ang_vel(3)`，由 `MotionFeatureBuffer` 在线重建，与预训练布局一致；yaw-only 局部系锚定；reset 时用 GSI 从 prior 采样窗口初始化。

### 依赖与入口

- 包管理：`uv sync`（`uv.lock` 锁定 mjlab git rev）
- 训练示例：`uv run scripts/train.py Smp-Forward-G1 --env.scene.num-envs=4096`
- 回放：`uv run scripts/play.py Smp-Forward-G1 --wandb-run-path <org>/<project>/<run>`

---

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| SMP 方法本体 | `wiki/methods/smp.md`（补充 G1/mjlab 复现与乘性奖励设计） |
| 本仓库实体 | `wiki/entities/smp-g1-mjlab.md` |
| mjlab 生态 | `wiki/entities/mjlab.md` |
| AMP 对照（G1 + mjlab） | `wiki/entities/amp-mjlab.md` |
| MimicKit 原版栈 | `wiki/entities/mimickit.md` |
