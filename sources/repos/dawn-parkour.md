# dawn-parkour：DAWN 官方实现

- **类型：** repo
- **代码：** <https://github.com/DocyNoah/dawn-parkour>
- **项目页：** <https://dawn-parkour.github.io/>；[项目页归档](../sites/dawn-parkour.md)。
- **论文：** <https://arxiv.org/abs/2609.29092>；[论文归档](../papers/dawn_arxiv_2609_29092.md)。
- **核查日期：** 2026-11-08
- **许可证：** MIT。
- **开源状态：** 训练与仿真回放源码已发布；README 使用本地训练 checkpoint，未列独立权重下载；目录未见 Go1 SDK / Jetson 真机通信入口。

## 可辨识入口

| 文件 / 模块 | 用途 |
|---|---|
| `INSTALL.md`、`setup_isaaclab_env.sh` | Isaac Sim 5.1.0、固定 IsaacLab 子模块与 uv 安装 |
| `dawn/train.py` | 启动模拟器、创建环境与 runner，调用训练 |
| `dawn/rl/runners/dawn_runner.py` | rollout、世界模型特征、动作历史与 PPO/AMP 更新 |
| `dawn/rl/modules/dawn_world_model.py` | `_target_key` 选干净重建目标、`_extra_losses` 构造双分支、`_compute_contrastive_loss` 计算双向 CE |
| `dawn/sim/utils/depth_noise_model.py` | 深度噪声实现 |
| `dawn/play.py`、`dawn/rl/runners/dawn_eval_runner.py` | checkpoint 仿真回放 |
| `dawn/datasets/mocap_motions/` | hop/trot 动作文件 |

## 对比学习实现核验

已阅读官方 `dawn/rl/modules/dawn_world_model.py`、`dawn/rl/runners/dawn_runner.py` 和 `dawn/train.py` 的相关训练路径。

1. `DAWNWorldModel.__init__` 构建共享 Projection Head：`Linear → LayerNorm → SiLU → Linear`；输入维度为 `dyn_stoch * dyn_discrete + dyn_deter`，其参数加入世界模型优化器。
2. `_extra_losses` 用 `data_clean = {**chunk, "image": chunk["image_clean"].clone()}` 构造干净分支，调用 `rssm.obs_embedder → rssm.observe_traj`；对原带噪 `latent_traj.s` 和 `latent_clean.s` 分别经同一个 `_projection_head`。
3. `_compute_contrastive_loss` 将 `[B,T,D]` 展为 `[B*T,D]`，先归一化，再计算 `logits = z_noisy @ z_clean.T / temperature`；以对角线为正样本，正向和转置方向的 `cross_entropy` 取均值，乘 `contrastive_coef`。
4. **与论文式 (9) 的区别：** 论文 NT-Xent 分母包含 2N 投影特征；源码采用 clean/noisy 跨分支 `BT×BT` 矩阵，没有把同分支的 clean-clean / noisy-noisy 负样本逐项加入，并把其他时间步纳入非对角矩阵元素。复现应以源码为准。
5. `_target_key("image", chunk)` 在存在 `image_clean` 时使用干净深度作为解码监督；对比损失在投影特征上求值，但梯度能回传世界模型。部署推理不调用该对比分支。

`DawnRunner.train_world_model` 从 `DAWNRolloutStorage` 采样序列，设置 `image`（noisy）、`image_clean`（clean），随后调用 `self._world_model._train(batch_data)`。训练由 `dawn.train → runner.learn` 进入。

README 使用 RTX 5090、默认 4,096 环境；不能承诺 8 GB 显存直接运行默认配置。此次未执行 GPU 训练或 Go1 真机部署。

## 复现入口

按 `INSTALL.md` 初始化子模块、安装 Isaac Sim、执行 `uv sync --extra cu128` 与 setup 脚本后：

```bash
source .venv/bin/activate
python -m dawn.train dawn
python -m dawn.play --checkpoint.model-dir "logs/DAWN/<run>/model_<iteration>" --env.task Go1-DAWN-Play-v0 --env.num-envs 4 --video.enable
```

checkpoint 路径需替换为实际训练输出。

**知识页：** [DAWN](../../wiki/entities/paper-dawn.md)。
