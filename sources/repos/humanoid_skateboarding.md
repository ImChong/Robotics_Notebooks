# humanoid_skateboarding（HUSKY 官方代码）

- **标题：** HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control
- **类型：** repo
- **仓库：** <https://github.com/TeleHuman/humanoid_skateboarding>
- **项目页：** <https://husky-humanoid.github.io/>
- **论文：** arXiv:[2602.03205](https://arxiv.org/abs/2602.03205)（RSS 2026）
- **机构：** 中国电信人工智能研究院（TeleAI）、香港大学（HKU）、中国科学技术大学（USTC）、上海科技大学（ShanghaiTech）
- **硬件：** Unitree G1（23 DoF，不含腕部）+ 滑板
- **收录日期：** 2026-11-22
- **核对日期：** 2026-11-22
- **核对范围：** `TeleHuman/humanoid_skateboarding` README、仓库目录结构、`src/mjlab_husky/scripts/train.py` 与 `test_scene/sim.py`
- **通读：** 已完成
- **许可：** CC BY-NC 4.0（仓内 `LICENSE-CC-BY-NC-4.0.md`）

## 一句话摘要

基于 **mjlab**（MuJoCo 物理 + IsaacLab API）与定制 **rsl_rl**（含 `amp_ppo.py`）的人形滑板官方实现，开放 `uv run train/play` 训练评估、`test_scene/sim.py` 轻量 MuJoCo ONNX 仿真（含 `ckpts/test.onnx` / `test.pt`）与滑板推地数据集。

## 关键入口与目录

| 路径 | 作用 |
|------|------|
| `src/mjlab_husky/scripts/train.py` | `uv run train Mjlab-Skater-Flat-Unitree-G1` 主训练入口（4096 并行环境，AMP-PPO） |
| `src/mjlab_husky/scripts/play.py` | `uv run play` Viser 回放与 checkpoint 评估 |
| `src/mjlab_husky/tasks/skater/` | 滑板环境配置（`skater_env_cfg.py`）、MDP 观测/奖励/终止与 `rl/runner.py` |
| `src/mjlab_husky/asset_zoo/robots/skateboard/` | G1 + 滑板 MJCF 与 `g1_skater_constants.py` |
| `rsl_rl/` | 内嵌定制 RSL-RL，含 `amp_on_policy_runner.py` 与 `amp_ppo.py` |
| `test_scene/sim.py` · `test_scene/sim.sh` | 轻量 MuJoCo 场景与 ONNX 推理验证（键盘方向键控板） |
| `ckpts/` · `dataset/` | 预训练检查点（`test.pt` / `test.onnx`）与推地 AMP 参考 pose |

## 对 Wiki 的映射

- [HUSKY 实体页](../../wiki/entities/paper-amp-survey-14-husky.md)
- [husky_humanoid_skateboarding_arxiv_2602_03205.md](../papers/husky_humanoid_skateboarding_arxiv_2602_03205.md)

## 参考来源（原始）

- 代码仓库：<https://github.com/TeleHuman/humanoid_skateboarding>
- arXiv:[2602.03205](https://arxiv.org/abs/2602.03205)
