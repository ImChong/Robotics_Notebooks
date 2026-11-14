# GitHub 仓库：leggedrobotics/sru-pytorch-spatial-learning & sru-navigation-sim

- **仓库：** https://github.com/leggedrobotics/sru-pytorch-spatial-learning 与 https://github.com/leggedrobotics/sru-navigation-sim（ETH Zurich RSL 官方发布；注：早期草稿曾误引用已 404 的 `ManifoldTechLtd/sru-pytorch-spatial-learning`，实测不存在，正确官方归属为 `leggedrobotics` 组织，而 `ManifoldTechLtd/SRU-Odin` 为社区第三方适配分支）
- **关联项目：** SRU（Spatially-Enhanced Recurrent Memory）
- **开源状态：** 已开源（`sru-pytorch-spatial-learning` 公开核心 PyTorch 空间循环单元与合成点云/螺旋路径记忆基准；`sru-navigation-sim` 公开基于 Isaac Lab v2.1.1 的 B2W 与 AoW-D 导航仿真任务、`vae_pretrain_new.pth` 及运动控制策略资产；底层 RL 算法 PPO/MDPO 依赖分离包 `rsl_rl`）
- **核对范围：** 已通读官方仓库 `leggedrobotics/sru-pytorch-spatial-learning`（`README.md`、`IMPORT_EXAMPLES.md`、`run_pointcloud.py`、`network/lstm_sru_gate.py` 等核心模块）与 `leggedrobotics/sru-navigation-sim`（`docs/README.md`、`scripts/train.py` 及资产结构）
- **通读：** 已完成

## 代码结构与核心模块

1. **`leggedrobotics/sru-pytorch-spatial-learning`（核心空间记忆算子库）**
   - `network/lstm_sru_gate.py`：`LSTMSRUGateCell` 与 `LSTM_SRU_Gate` 对应论文 `SRU-Ours`，在前向计算中并发求解 `gates = linear_all([x, h])` 与空间变换门 `tx = transform_gate(x)`，执行逐元素乘积 `g_t = tanh(tx * g)` 并采用 Refine Gate `f = i * (1 - (1 - f)^2) + (1 - i) * f^2` 更新细胞态 `c_next`。
   - 同时提供 `network/lstm_sru.py`（`LSTM_SRU`）、`network/gru_sru.py`（`GRU_SRU`）以及基线 `MambaNet`、`S4Model` 与标准 `LSTM`/`GRU`。
   - `run_pointcloud.py`：实现论文 Appendix A 空间-时序记忆评估流程，`TransformBaseModel` 经 `PoseEncoder`（单位阵初始化）与 `ForwardLayer` 联合优化坐标均方误差 `coord_loss`（MSE）与类别标签 `info_loss`（10× BCEWithLogits）。

2. **`leggedrobotics/sru-navigation-sim`（Isaac Lab v2.1.1 分层视觉导航扩展）**
   - `scripts/train.py`：通过 `AppLauncher` 初始化 Isaac Sim 4.5 / Isaac Lab，加载注册任务（例如 `--task Isaac-Navigation-B2W-v0`，亦支持 `AoW-D`），经 `RslRlVecEnvWrapper` 包装环境并由 `rsl_rl.runners.OnPolicyRunner` 启动多环境（如 `--num_envs 2048`）PPO/MDPO 训练与日志归档。
   - **分层与边界说明**：高层策略以 **5 Hz** 输出 $SE(2)$ 速度指令 `(v_x, v_y, omega)` 驱动自带的 **50 Hz** 底层预训练运动策略；`sru-navigation-sim` 本身不包含自定义网络与 DML 调度代码，需搭配包含 SRU + 空间注意力 + DML 的 `rsl_rl` 分支包运行。
