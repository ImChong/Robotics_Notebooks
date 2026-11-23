# FADA 官方代码：Planner–IDM 训练、目标域 LoRA 与部署

> 官方 GitHub 一手资源归档；核查日期：2026-11-23。

- **仓库：** <https://github.com/LeCAR-Lab/FADA>
- **论文：** <https://arxiv.org/abs/2606.28476v2>（[论文归档](../papers/fada_arxiv_2606_28476.md)）
- **项目页：** <https://lecar-lab.github.io/FADA-humanoid/>（[主页归档](../sites/fada-humanoid.md)）
- **权重：** <https://huggingface.co/AngchenXie/fada-checkpoints>（README 提供的 G1 / T1 oracle checkpoint）
- **许可证：** Apache-2.0；基于 Amazon FAR 的 Holosoma 扩展
- **通读：** 已完成
- **核对范围：** 官方 README 全部六个阶段、源码 train.py / eval_checkpoint.py / finetune_idm_lora.py / run_policy.py 的入口与模块引用；运行命令未在本次环境实际执行。

## 开源边界

- **已开放：** oracle PPO 源域训练、Planner–IDM DAgger 蒸馏、ONNX 评估导出、MuJoCo/真机推理及 H5 采集、LoRA IDM 目标域适应、oracle 权重。
- **未开放：** 仓库 Status 中 whole-body tracking 仍为未完成项。论文的跟踪实验有研究结果，不等于该任务源码已经可运行。
- **运行要求：** README 提供 Ubuntu 22.04+、IsaacSim（站点徽标 5.1.0）、MuJoCo（3.0+）、单独的 inference 环境；实际依赖及设备须按官方 setup 脚本配置。

## README 复现入口（六步）

1. **源域 oracle：** `python src/holosoma/holosoma/train_agent.py exp:g1_29dof_oracle simulator:isaacsim logger:wandb --logger.base-dir logs/g1_oracle`；T1 使用 `exp:t1_23dof_waist50_oracle`。
2. **DAgger 学生：** `python -m holosoma.fada.planner_idm.train --expert-checkpoint <oracle-checkpoint>`。默认弱策略数据采样依赖同目录的中间 oracle checkpoints；可通过 `--suboptimal-data-ratio 0` 关闭该采样需求。
3. **评估导出：** `python -m holosoma.fada.planner_idm.eval_checkpoint --checkpoint <dagger-run>/model_final.pt` → `planner_idm_policy.onnx`。
4. **MuJoCo 与真机采集：** `src/holosoma/holosoma/run_sim.py` 启动模拟器；`src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco-fada` 加 `--task.model-path`、`--task.collect-data` → `dataset.h5`。真机使用 `--task.interface` 指定网络接口并遵守硬件部署文档。
5. **适应：** `python -m holosoma.fada.planner_idm.finetune_idm_lora --checkpoint <dagger-run>/model_final.pt --target-datasets <dataset.h5>` → 新的 ONNX。
6. **对照评估：** 同一命令序列分别运行适应前后 ONNX，产生各自的 `mocap_unified.npz`；官方 README **不自动计算比较指标**，由使用者后处理。

## 关键源码锚点

- `src/holosoma/holosoma/fada/planner_idm/train.py`：加载 oracle、构建学生与 FADATrainer。
- `src/holosoma/holosoma/fada/planner_idm/eval_checkpoint.py`：恢复模型并调用 eval / ONNX 导出。
- `src/holosoma/holosoma/fada/planner_idm/finetune_idm_lora.py`：加载 H5 轨迹、注入 LoRA、目标域优化与合并导出。
- `src/holosoma_inference/holosoma_inference/run_policy.py`：选择 `LocomotionPolicy_FADA` 进行推理及数据采集。

## 对 wiki 的映射

- [FADA 论文独立实体](../../wiki/entities/paper-fada-humanoid.md) — 对齐源码运行时序与论文三阶段算法
- [LoRA](../../wiki/concepts/lora.md) — IDM 低秩参数适应
