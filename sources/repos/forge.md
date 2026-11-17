# FORGE 官方代码仓库：bb0928/FORGE

> 一手源码归档（2026-11-17 核对）

- **类型：** repo / Isaac Gym / 强化学习全身控制
- **代码：** https://github.com/bb0928/FORGE
- **项目：** https://bb0928.github.io/forge/
- **论文：** https://arxiv.org/abs/2611.17142
- **开源状态：** 部分开源；第二阶段残差策略训练和仿真评估可读；真机部署链仍待发布
- **核对范围：** 官方 README、`scripts/train_stage2_residual.sh`、`scripts/test_res_pf.sh`、`legged_gym_repo/legged_gym/scripts/train.py`、`play_res.py`、`legged_gym/utils/task_registry.py`、`legged_gym/envs/__init__.py`、`rsl_rl_repo/rsl_rl/runners/on_policy_runner.py`。
- **通读：** 已完成
- **入库日期：** 2026-11-17

## 可核验的运行入口

| 入口 | 代码中的实际调用/作用 |
|---|---|
| `scripts/train_stage2_residual.sh` | 检查 Python 依赖、设置 PYTHONPATH 和基础模型路径，再调用 `legged_gym/scripts/train.py --task g1_ee_residual --num_envs 4096 --max_iterations 20000 --base_model_path model_17000.pt` |
| `legged_gym/scripts/train.py` | `task_registry.make_env` → `task_registry.make_alg_runner` → `ppo_runner.learn` |
| `legged_gym/envs/__init__.py` | 将 `g1_ee_residual` 注册到 `LeggedRobotEEResidual` 与相应配置 |
| `task_registry.py` | 组装仿真环境与训练 Runner，并处理 checkpoint 的 resume / 加载 |
| `on_policy_runner.py` | 循环采样 `alg.act` → `env.step`，然后计算 returns、执行 `alg.update`、保存模型 |
| `scripts/test_res_pf.sh` | 设置已有 checkpoint 路径，调用 `legged_gym/scripts/play_res.py --task g1_ee_residual --resume --num_envs 1 --headless` |
| `play_res.py` | 创建环境及 runner、加载推理策略，用观测循环推理并推进 Isaac Gym 环境 |

## 复现边界与注意事项

1. README 需要手动安装 Isaac Gym；其环境示例为 Python 3.8 / PyTorch 2.4.1 / CUDA 12.1，需按本机平台核对兼容性。
2. `model_17000.pt` 在仓库根目录；测试脚本的 `RESUMEPATH` 是作者当时的具体日志路径，复现者必须改为自己的 checkpoint。
3. **论文 §III-B / IV-A 写 2,048 并行环境，公开 shell 默认写 4,096**；不能将两者不加说明地视为完全相同的训练设置。
4. README TODO：基础策略训练、MuJoCo 部署和真机部署脚本尚未发布；公开的仿真回放不等于已经开放完整真机链路。
5. 上述为源码静态核对，**未运行**训练、评测或真机部署，不宣称成功复现原文数值。

## 对 wiki 的映射

- [FORGE 论文唯一实体节点](../../wiki/entities/paper-forge-force-aware-loco-manipulation.md)
- [原文归档](../papers/forge-force-aware-loco-manipulation_arxiv_2611_17142.md)
- [项目页归档](../sites/forge.md)
