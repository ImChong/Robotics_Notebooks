# engineai_amp（众擎机器人）开源仓库与 AMP 理论基础精读归档

- **类型：** 开源仓库精读（配合基础论文核对）
- **仓库：** <https://github.com/engineai-robotics/engineai_amp>
- **本地检出提交：** `83ba64b`（`main` 分支）
- **所属机构：** EngineAI（众擎机器人）
- **理论对照原文：** *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control*（Peng et al., arXiv:2104.02180v2 / ACM TOG 2021）
- **通读：** 已完成
- **核对范围：** `engineai-robotics/engineai_amp@83ba64b` 全部核心源码（`README.md`、`scripts/train.py`、`scripts/play.py`、`dataset/config/dataset.yaml`、`source/engineai_lab/tasks/velocity/config/pm01/__init__.py`、`source/engineai_lab/tasks/velocity/config/pm01/flat_amp_env_cfg.py`、`source/engineai_lab/tasks/velocity/config/pm01/agents/amp_ppo_cfg.py`、`source/engineai_lab/algorithms/amp_ppo.py`、`source/engineai_lab/utils/AMP_discriminator.py`、`source/engineai_lab/utils/AMP_data_loader.py`），并对照 Peng et al.（`arXiv:2104.02180v2`）第 5–6 节与 Algorithm 1。

---

## 一、理论基线对照（Peng et al., arXiv:2104.02180v2）

1. **LSGAN 判别目标与有界风格奖励（Eq. 6–7）**：为缓解标准 Sigmoid GAN 的梯度饱和，AMP 采用最小二乘 GAN（LSGAN）：判别器 `D(Φ(s), Φ(s'))` 将专家动捕转移回归至 `+1`、策略采样转移回归至 `-1`；策略获得的风格奖励经偏移与截断归一化至 `[0, 1]`：
   `r^S(s_t, s_{t+1}) = max(0, 1 - 0.25 * (D(Φ(s_t), Φ(s_{t+1})) - 1)^2)`，总奖励为 `r_t = w_G r_t^G + w_S r_t^S`。
2. **专家流形梯度惩罚（Eq. 8）**：针对真实专家特征 `φ = (Φ(s), Φ(s'))` 施加梯度惩罚 `(w_gp / 2) E_{d^M}[||∇_φ D(φ)||_2^2]`，抑制判别器在专家流形上的非零梯度引起的振荡。

---

## 二、仓库源码实现核对（`engineai_amp@83ba64b`）

1. **推荐依赖与注册任务（`README.md`、`pm01/__init__.py`）**：
   - 基于 Isaac Sim >= 5.0、Isaac Lab（推荐提交 `4df6560e`）与 `rsl-rl-lib >= 5.0`。
   - 注册两个面向 EngineAI **PM01** 人形机器人的 Gym 环境：`Flat-PM01-v0`（基础平地行走 PPO）与 `Flat-AMP-PM01-v0`（`PM01AMPFlatEnvCfg` + `PM01FlatAMPPPORunnerCfg` + `AMPPPO`）。
2. **观测、参考运动加载与特征构造**：
   - `flat_amp_env_cfg.py` 在 `PolicyCfg`/`CriticCfg` 之外新增 `AMPCfg`，以 `history_length = 5` 采样 `dummy_history_term(env)`；该函数将名称匹配 `.*WAIST_YAW.*` 的腰部偏航关节位置和速度置零（兼容不同构型数据集），并将关节位置与基座线速度（`robot_base_lin_vel_b`）按经验系数缩放（`* 9`、`* 7`）拼接为单帧特征。
   - `amp_ppo_cfg.py` 指定 `frame_dim = 26`、`frame_length = 5`、`discriminator_hidden_dims = [512, 256, 128]`、`style_reward_weight = 2.0`、`frame_normalization = True`、`dataset_path = "dataset/config/dataset.yaml"`（指向 `dataset/data/locomotion.npz`）。
   - `AMPDataLoader` 从 `.npz` 加载 `fps`、`joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`，通过 `subtract_frame_transforms` 与 `quat_inv` 将速度变换至锚点局部坐标系，并在 `mini_batch_generator` 中按 `old -> new` 顺序拼接 5 帧特征。
3. **训练、判别器更新与导出部署（`amp_ppo.py`、`AMP_discriminator.py`、`scripts/train.py`、`scripts/play.py`）**：
   - `AMPPPO.process_env_step()` 计算 `amp_reward = 0.01 * self.style_reward_weight * self.discriminator.get_amp_reward(obs["amp"])`，并与环境任务奖励相加传入基类 `PPO.process_env_step()`。
   - `AMPPPO.update()` 每 4 次 PPO 更新（`ppo_update_counter % 4 == 0`）执行一次判别器训练：对专家与策略特征计算 `0.5 * (MSE(expert, +1) + MSE(policy, -1))` 再加上 `compute_grad_pen(amp_ref_batch, lambda_=10)`，并用 `EmpiricalNormalization` 对每帧切片更新归一化统计；随后调用 `super().update()` 更新策略与价值网络。
   - `scripts/play.py` 加载检查点后自动调用 `ppo_runner.export_policy_to_onnx(...)` 在 `exported/` 目录导出 ONNX；`README.md` 部署章节进一步说明通过阿里巴巴 `MNN-Converter` 将 ONNX 转为 MNN 模型以便真机部署（仓库代码侧不包含 C++ 真机端运行时源码）。
