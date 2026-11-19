# Diffusion Policy（real-stanford/diffusion_policy）

- **URL：** <https://github.com/real-stanford/diffusion_policy>
- **项目页：** <https://diffusion-policy.cs.columbia.edu/>
- **组织：** real-stanford
- **关联论文：** [diffusion_policy_arxiv_2303_04137](../papers/diffusion_policy_arxiv_2303_04137.md)
- **最近复核：** 2026-11-19
- **核对范围：** `README.md`、`train.py`、`eval.py`、`eval_real_robot.py`、`diffusion_policy/workspace/train_diffusion_{unet,transformer}_hybrid_workspace.py`、`diffusion_policy/policy/diffusion_{unet,transformer}_hybrid_image_policy.py`、`diffusion_policy/model/diffusion/{conditional_unet1d,transformer_for_diffusion}.py`、`diffusion_policy/env_runner/pusht_image_runner.py`
- **通读：** 已完成

## 一句话说明

RSS 2023 Diffusion Policy 官方 Hydra + PyTorch 实现，覆盖 CNN（1D U-Net + FiLM）与 Transformer 两类去噪策略、仿真异步评测及真实机械臂共享内存实时控制管线。

## 关键入口与模块

| 路径 / 符号 | 角色 | 关键实现细节（源码核对） |
|-------------|------|--------------------------|
| `train.py` | 训练总入口 | `@hydra.main` 加载配置，实例化 `BaseWorkspace` 并调用 `.run()` |
| `eval.py` | 仿真评测入口 | `@click.command` 读取 checkpoint，调用 `workspace.load_payload` 与 `env_runner.run(policy)` 输出 `eval_log.json` |
| `eval_real_robot.py` | 真机闭环控制入口 | 基于 `SharedMemoryRingBuffer` / `SharedMemoryQueue` 与 `RealEnv` 异步执行，代码中固定设 `policy.num_inference_steps = 16`（DDIM）与 receding-horizon 调度 |
| `TrainDiffusionUnetHybridWorkspace` / `TrainDiffusionTransformerHybridWorkspace` | 训练编排 | 管理 EMA 模型、余弦学习率调度、`LinearNormalizer` 与定时 `env_runner.run` 验证 |
| `DiffusionUnetHybridImagePolicy` | CNN 去噪策略 | 使用 robomimic `bc_rnn` `algo_factory` 构造多相机 `obs_encoder`（GroupNorm + `CropRandomizer`），经 `ConditionalUnet1D` 与 `LowdimMaskGenerator` / `DDPMScheduler` 执行 `compute_loss` 与 `conditional_sample` |
| `DiffusionTransformerHybridImagePolicy` | Transformer 去噪策略 | 使用 `TransformerForDiffusion`（MinGPT 风格因果 encoder-decoder 或 BERT 编码器，带因果 `mask` / `memory_mask` 与 `ln_f`/`head`），`get_optimizer` 拆分 weight decay |
| `ConditionalUnet1D` / `TransformerForDiffusion` | 去噪骨干网络 | `ConditionalResidualBlock1D` 通过 FiLM 调制注入观测与时间步；Transformer 版采用交叉注意力与时间因果掩码 |
| `PushTImageRunner` | 仿真评测封装 | 基于 `AsyncVectorEnv`、`MultiStepWrapper` 与 `VideoRecordingWrapper` 并行 rollout |

## 交叉链接

- [Diffusion Policy 论文实体](../../wiki/entities/paper-diffusion-policy.md)
- [Diffusion Policy 方法页](../../wiki/methods/diffusion-policy.md)
