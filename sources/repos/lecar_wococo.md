# LeCAR-Lab/wococo 官方仓库核对笔记

> 来源归档（official-repo · GitHub 代码与训练/推理入口源码级核对）

- **仓库：** <https://github.com/LeCAR-Lab/wococo>
- **对应论文：** *WoCoCo: Learning Whole-Body Humanoid Control with Sequential Contacts*（arXiv:2406.06005v2 / CoRL 2024 Oral）
- **机构：** CMU LeCAR Lab / ETH Zurich
- **许可证：** CC BY-NC 4.0（代码库），并继承 ETH Zurich / Nikita Rudin（`legged_gym`、`rsl_rl`）与 NVIDIA Corporation & Affiliates（Isaac Gym）原有许可证；禁止用于商业用途或商业产品宣传演示。
- **核对范围：** 通读 `README.md`、目录树及核心运行与算法源码：`legged_gym/legged_gym/envs/__init__.py`、`legged_gym/legged_gym/scripts/train.py`、`legged_gym/legged_gym/scripts/play.py`、`legged_gym/legged_gym/envs/base/h1_jumpjack.py`、`legged_gym/legged_gym/envs/base/curiosity.py`、`legged_gym/legged_gym/envs/h1/h1_jumpjack_config.py`、`rsl_rl/rsl_rl/runners/on_policy_runner.py` 与 `rsl_rl/rsl_rl/algorithms/ppo_sym.py`。
- **通读：** 已完成

## 一、开源边界与目录结构

- **仓库明确「去工程化（de-engineered）」展示核心范式：** README 说明未把特定地形课程、所有任务配置或真机 C++ 推理栈一并打包发布，而是提供 **Dynamic Clap-and-Tap Dancing（`h1:jumpjack`）** 作为顺序接触 MDP、稠密接触奖励、阶段计数、Hash 好奇心与对称 PPO 的完整参考实现。
- **未包含在公开仓库的论文资产：** Parkour Jumping、Anywhere-to-Anywhere Box Loco-Manipulation、Bidirectional Cliffside Climbing 与 22-DoF Dinosaur 任务的独立环境类、真机部署/动捕接入 C++ 代码以及论文中修正躯干质量后的自定义 H1 URDF 参数说明外，核心算法模块完整可用。
- **关键目录：**
  - `legged_gym/legged_gym/envs/__init__.py`：通过 `task_registry.register("h1:jumpjack", H1JumpJack, H1JumpJackCfg(), H1JumpJackCfgPPO())` 注册示例任务。
  - `legged_gym/legged_gym/envs/base/h1_jumpjack.py`：顺序接触环境主实现。
  - `legged_gym/legged_gym/envs/base/curiosity.py`：实现论文 `NHashCuriosity` 与对照用的 `RNDCuriosity`。
  - `legged_gym/legged_gym/envs/base/lpf.py`：实现 Butterworth 低通滤波器（`ActionFilterButter`、`ActionFilterButterTorch` 等）。
  - `legged_gym/legged_gym/envs/h1/h1_jumpjack_config.py`：任务、奖励权重、curiosity 边界与 `PPO_SYM` 超参数配置。
  - `legged_gym/legged_gym/scripts/train.py` / `play.py`：训练与可视化/JIT 导出入口。
  - `rsl_rl/rsl_rl/runners/on_policy_runner.py` & `algorithms/ppo_sym.py`：带左右对称数据增强的 PPO 训练器。

## 二、核心源码与论文公式对齐核对

1. **顺序接触状态机与阶段推进（`h1_jumpjack.py`）：**
   - `_init_buffers()` 分配 `contact_sequence` 形状 `(num_envs, 4, num_seq)`（`num_seq = 10`）、`current_contact_goal` 与 `period_contact = [0.4, 0.8]` s。
   - `post_physics_step()` 仅当 `self.current_contact_reached` 为真时累加 `time_since_contact += self.dt`，超过 `period_contact` 后清零并推进 `current_contact_goal`（对应论文第 2 节「满足后延迟切换阶段」）。
2. **WoCoCo 奖励实现（`h1_jumpjack.py` L272–340 & `h1_jumpjack_config.py` L212–240）：**
   - `_reward_on_box()` 计算双脚、双手接触与任务位置/拍手判定 `correct_all`，更新 `current_contact_reached[:, 0] |= correct_all`，并在 `current_contact_goal > 0` 时对 `wrong_contact` 施加 `-4.0` 惩罚（对应论文 Eq. 3）。
   - `_reward_prev_fulfill()` 返回 `current_contact_goal * _whole_body_fulfillment()`（对应论文 Eq. 4：$r_{\mathrm{stage}} = n_{\mathrm{stage}} F_{\mathrm{task}}$）。
   - `_reward_curiosity()` 构造 `29` 维世界系基座+末端位置与接触观测 `curio_obs`，传入 `curiosity_handler.update_curiosity(curio_obs, shift)`。
3. **随机神经网络 Hash 好奇心（`curiosity.py` L25–79）：**
   - `NHashCuriosity.normalize()` 将 `29` 维观测按 `obs_lb`/`obs_ub` 截断并缩放到 $[0, \pi]$，拼接 `cos`/`sin` 为 `58` 维输入。
   - `hashnn` 为单隐层 `32` 单元 ReLU MLP 输出 `16` 维符号布尔向量（`2^{16} = 65536` 桶），`bin2int()` 转为桶索引后按 `1 / sqrt(1 + bin_cnt[bin_idx])` 计算奖励。
   - `update_curiosity()` 对原始观测与 `sym_map(obs)` 左右对称观测各计算一次并取 `0.5 * rew_orig + 0.5 * rew_sym`（对齐 Appendix D）。
4. **对称 PPO 与观测配置（`ppo_sym.py`、`h1_jumpjack_config.py`）：**
   - `H1JumpJackCfg` 设 `num_envs = 4096`、`num_observations = 187`（含 19 DoF 本体、phase 4 维、world xyyaw 3 维、2 步历史堆叠共 114 维等）、`num_actions = 19`、`decimation = 4`（50 Hz 策略 / 200 Hz PD）、`action_filt = True` 与 `action_cutfreq = 4.0` Hz。
   - `OnPolicyRunner.__init__()` 检测 `alg == PPO_SYM` 时自动注入 `self.alg.symmetric_fn = self.env.symmetric_fn`；`PPO_SYM.update()` 在每个 minibatch 内调用 `self.symmetric_fn(...)` 将 `batch_size` 翻倍并合并原始/对称样本同步反传梯度。注意 `H1JumpJack.symmetric_fn()` 显式断言 `not self.privileged_obs_buf`（因此代码默认关闭特权 Critic 观测或要求其与策略对称）。

## 三、官方训练与评估入口

- **环境依赖：** Python 3.8 + PyTorch + Isaac Gym Preview 4 + `pip install -r requirements.txt`（附注：`legged_gym/legged_gym/scripts/train.py` L12 遗留了一行作者本地路径 `sys.path.append("/home/wenli-run/unitree_rl_gym")`，不影响模块导入）。
- **训练命令（工作目录 `legged_gym/legged_gym`）：**
  - `python3 scripts/train.py --task=h1:jumpjack` -> `task_registry.make_env` & `make_alg_runner`（`PPO_SYM`，`ActorCritic [512, 256, 128]`，`max_iterations = 100000`）。
- **回放/验证与 JIT 导出（工作目录 `legged_gym/legged_gym`）：**
  - `python scripts/play.py --task=h1:jumpjack --num_envs=3`
  - `play.py` 默认 `EXPORT_POLICY = False`、`env_cfg.rewards.scales.curiosity = 0`；将 `EXPORT_POLICY` 设为 `True` 可调用 `export_policy_as_jit` 导出 TorchScript 策略用于 C++ 部署，另有 `groupdance = True` 与 `--joystick` 交互测试分支。

## 对 wiki 的映射

- [paper-loco-manip-161-116-wococo](../../wiki/entities/paper-loco-manip-161-116-wococo.md)
