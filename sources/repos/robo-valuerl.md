# Robo-ValueRL 官方训练与部署仓库

- **仓库：** https://github.com/Open-X-Humanoid/Robo-ValueRL
- **论文：** https://arxiv.org/abs/2607.09866
- **项目：** https://gewu-lab.github.io/Robo-ValueRL/
- **核验日期：** 2026-11-12
- **开放结论：** 训练、评测、离线/在线迭代与部署入口已发布；所需真机、示教数据、模型 checkpoint 与多卡资源须单独准备。

## 实际运行入口

1. `install_simple.sh` 建环境（安装含 Transformers 兼容补丁）。
2. `scripts/train_robo_valuerl_value_estimator.sh` 调用 `train_robo_valuerl_value_estimator.py`。
3. `annotate_value_function_sparse_part.py` 标注剩余时间，`generate_action_quality/generate_quality.py` 排序产出动作质量字段。
4. `scripts/train_robo_valuerl_offline.sh` 调用 `train_robo_value_rl_offline_rl.py` 训练质量条件策略。
5. GPU 端 `deploy_robo_valuerl.py` 与机器人端 `rtc_env_client_inference_with_delta.py` 用 socket 通信、实时执行动作块；机器人环境 `rl_envs/realworld_x_humanoid_env.py`。
6. `scripts/train_robo_valuerl_online.sh` 调用 `train_robo_value_rl_online_rl.py` 训练残差适配。

官方 README 的硬件参考：8×A100 训练、RTX4090 推理；这些是作者配置，非实测最低配置。

- **关联：** [来源论文](../papers/robo_value_rl_arxiv_2607_09866.md)；[wiki](../../wiki/entities/paper-robo-valuerl.md)
