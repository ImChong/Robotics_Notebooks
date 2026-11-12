# Robot-Learning-Collective/behavior-1k-solution-2026｜源码核对

- **官方仓库：** https://github.com/Robot-Learning-Collective/behavior-1k-solution-2026
- **论文：** https://arxiv.org/abs/2611.14432
- **官网：** https://robot-learning-collective.github.io/behavior-1k-2026/
- **权重：** https://huggingface.co/IliaLarchenko/BEHAVIOR2026VLA
- **许可证：** Apache-2.0（仓库 LICENSE）
- **核对日期：** 2026-11-12
- **核对范围：** 读取 README、agent/README.md；逐文件阅读 scripts/train.py、scripts/serve_b1k.py、scripts/run_parallel_eval.py、scripts/eval_worker.py 与 src/b1k/models/pi_behavior.py、src/b1k/shared/eval_b1k_wrapper.py、src/b1k/policies/policy_config.py 的实现入口和核心符号；未实际启动 GPU 仿真/训练。
- **通读：** 已完成

## 可执行路径（静态核对）

1. **训练：** `uv run scripts/train.py --yaml configs/train_b1k_2026.yaml`。入口为 `train.py::main`，通过 `b1k.training.data_loader` 获取观测/动作，建立 `PiBehavior` 并用 `train_step` 优化损失。配方需要额外数据、权重和 GPU。
2. **服务：** `uv run scripts/serve_b1k.py --policy.config configs/train_b1k_2026.yaml --policy.dir models/behavior2026_vla`。入口 `main` 加载 checkpoint (`b1k.policies.policy_config.create_trained_policy`)，创建 `MultiSessionPolicyServer`。WebSocket 会话维护 `B1KPolicyWrapper` 等实例，批量策略推理并返回动作。
3. **仿真评测：** `uv run scripts/run_parallel_eval.py --task-name turning_on_radio --instance-indices 0 1 2 3 --num-envs 4`。启动 `scripts/eval_worker.py`，由 `b1k.shared.slot_evaluator.SlotEvaluator` 驱动 OmniGibson worker，与策略服务交互。
4. **策略细节：** `B1KPolicyWrapper.begin_step/end_step/act_batched` 负责阶段、动作分块与调整；`PiBehavior.sample_actions` 输出流匹配动作块，`PiBehavior.compute_loss` 供训练。
5. **附加 agent：** 服务端在 `--agent` 允许时将 `agent/tasks/tasks.json` 选定任务交给 `b1k.policies.agent.AgentRoute`；其实践时的冻结会话资源**没有发布**。

## 资源与复现边界

- 需要拉取仓库子模块（openpi、BEHAVIOR-1K）、配置 Isaac Sim/OmniGibson 与 checkpoint。
- Hugging Face 权重与训练数据**是不同 repo**，数据 repo 为 `behavior-1k/2026-challenge-demos`。
- 已发布训练 recipe 不构成对最终模型 checkpoint 的 bit-exact 重建说明；README 称训练过程中跨硬件迁移、更新过配置。
- 本次为源码静态审阅，**未执行任何基准测试**。

## 对 wiki 的映射

- [论文归档](../papers/behavior_2026_solution_arxiv_2611_14432.md)
- [项目页归档](../sites/behavior-1k-2026.md)
- [统一 wiki 实体](../../wiki/entities/paper-behavior-2026-one-policy-100-tasks.md)
