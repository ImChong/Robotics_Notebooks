# senlanke/mimic — SMP on Unitree G1（mjlab 复现）

> 来源归档

- **标题：** SMP — Score-Matching Motion Priors (G1 reproduction on mjlab)
- **类型：** repo
- **链接：** https://github.com/senlanke/mimic
- **机构：** 清华大学 SUZ 课题组课程项目向复现（作者 GitHub: senlanke；同系仓库 [SUZ-tsinghua/smp](https://github.com/SUZ-tsinghua/smp)）
- **入库日期：** 2026-08-25
- **一句话说明：** 在 MimicKit 原版未提供 G1 的前提下，于 **mjlab** 上端到端复现 SMP（DDPM 预训练 + 冻结 SDS 引导奖励 + PPO），覆盖 Unitree G1 四类下游任务，内置三套可跳过预训练的 prior checkpoint；奖励采用 **task × r_smp** 乘性组合。
- **沉淀到 wiki：** 是 → [`wiki/entities/smp-g1-mjlab.md`](../../wiki/entities/smp-g1-mjlab.md)、[`wiki/entities/paper-smp.md`](../../wiki/entities/paper-smp.md)

---

## 与原始论文 / 参考实现的关系

| 资料 | 链接 | 角色 |
|------|------|------|
| SMP 论文 | [arXiv:2512.03028](https://arxiv.org/abs/2512.03028) | 方法定义（SDS、ESM、GSI、可复用先验） |
| 项目页 | https://yxmu.foo/smp-page/ | 官方说明与结果 — [`sources/sites/smp-project.md`](../sites/smp-project.md) |
| MimicKit | https://github.com/xbpeng/MimicKit | **论文作者官方** SMP 实现（`docs/README_SMP.md`） |
| SUZ-tsinghua/smp | https://github.com/SUZ-tsinghua/smp | 同系 G1 复现（org 镜像；README 与 senlanke/mimic 内容一致） |
| mjlab | https://github.com/mujocolab/mjlab | RL 环境骨架（`ManagerBasedRlEnv`） |

本仓库为**课程项目向复现**，非论文一作官方代码；工程贡献是 **G1 运动特征、先验、任务与奖励** 的完整移植。

---

## 技术要点摘录

### 管线三阶段

1. **数据处理**：`scripts/csv_to_npz.py` → 窗口化 NPZ；`scripts/compute_norm_stats.py` → q01/q99 归一化
2. **扩散预训练**：`scripts/pretrain.py` 训练 DDPM ε-预测器；可跳过，使用 `datasets/pretrain_ckpt/` 内三套权重
3. **RL**：`scripts/train.py` / `scripts/play.py` — PPO + 冻结 prior 的 SDS 风格 `r_smp`

### 预置 prior（免预训练）

| Checkpoint | 训练数据 | 默认任务 |
|------------|----------|----------|
| `pretrained_loco.pt` | walk / jog / run | `Smp-Forward-G1` |
| `pretrained_lafan_run.pt` | LAFAN run 子集 | `Smp-Steering-G1`、`Smp-Location-G1` |
| `pretrained_getup_f2s2.pt` | fall→stand get-up | `Smp-Getup-G1` |

### 四类下游任务

| Task ID | 说明 |
|---------|------|
| `Smp-Forward-G1` | 固定 +x 朝向，速度 0.5–5 m/s 前进 |
| `Smp-Steering-G1` | 速度 + 朝向跟踪 |
| `Smp-Location-G1` | 世界系 xy 目标点 |
| `Smp-Getup-G1` | 跌倒姿态起身（GSI 初始化） |

### 与 MimicKit 的关键差异：乘性奖励

- **原版（加性）：** `r = w_task · task + w_smp · r_smp`
- **本复现（乘性）：** `r = (Σ wᵢ taskᵢ) × r_smp`，`r_smp = exp(-w_s/|K| · Σ ‖ε̂−ε‖²)`

### G1 运动特征（59 维/帧）

`root_pos(3) + root_rot(6) + joint_pos(29) + ee_pos(15) + root_lin_vel(3) + root_ang_vel(3)`；`smp.rl` 包内 `MotionFeatureBuffer` 在线重建。

### 依赖与入口

- 包管理：`uv sync --frozen`（`uv.lock` 锁定 mjlab git rev；Python 3.13）
- 训练：`uv run scripts/train.py Smp-Forward-G1 --env.scene.num-envs=4096`
- 回放：`uv run scripts/play.py Smp-Forward-G1 --checkpoint-file <path>`

---

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 论文实体 | `wiki/entities/paper-smp.md` |
| SMP 方法本体 | `wiki/methods/smp.md` |
| G1 复现实体 | `wiki/entities/smp-g1-mjlab.md` |
| MimicKit 官方 | `wiki/entities/mimickit.md` |
