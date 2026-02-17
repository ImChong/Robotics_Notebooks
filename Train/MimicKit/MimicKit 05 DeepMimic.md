https://xbpeng.github.io/projects/DeepMimic/index.html
https://ar5iv.labs.arxiv.org/html/1804.02717?_immersive_translate_auto_translate=1
https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf
https://github.com/xbpeng/MimicKit/blob/main/docs/README_DeepMimic.md

# 《DeepMimic》论文

在《DeepMimic》论文的第 4 部分（Background），作者设定了标准**强化学习（RL）**问题的数学框架，用于训练物理仿真角色。以下是该部分公式的详细解释：

### 1. 期望回报（Expected Return）

角色的目标是通过学习最优参数 $\theta^*$ 来最大化其**期望回报 $J(\theta)$**： $$J(\theta) = \mathbb{E}_{\tau \sim p_{\theta}(\tau)} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

- **$J(\theta)$**：表示在策略参数为 $\theta$ 时，智能体在一个回合内能获得的平均总奖励。
- **$\mathbb{E}_{\tau \sim p_{\theta}(\tau)}$**：表示对所有可能的**轨迹（Trajectory）$\tau$** 求期望。
- **$\sum_{t=0}^{T} \gamma^t r_t$**：这是轨迹的回报总和。其中 $r_t$ 是第 $t$ 步的标量奖励，反映了该动作的优劣。
- **$\gamma \in$**：**折扣因子（Discount Factor）**，用于确保在无限时长的情况下回报依然有限，同时也体现了对即时奖励的偏好。

### 2. 轨迹概率分布（Trajectory Distribution）

轨迹 $\tau$ 是由状态和动作组成的序列 $(s_0, a_0, s_1, \dots, s_T)$，其分布由下式诱导： $$p_{\theta}(\tau) = p(s_0) \prod_{t=0}^{T-1} p(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t|s_t)$$

- **$p(s_0)$**：初始状态分布。
- **$p(s_{t+1}|s_t, a_t)$**：**环境动力学（Dynamics）**，即在当前状态执行某动作后转移到下一状态的概率。
- **$\pi_{\theta}(a_t|s_t)$**：**参数化策略**，即在给定状态下选择动作 $a$ 的概率分布。

### 3. 策略梯度（Policy Gradient）

为了优化参数 $\theta$，作者使用了策略梯度方法，通过对期望回报求梯度来更新参数： $$\nabla_{\theta} J(\theta) = \mathbb{E}_{s_t \sim d_{\theta}(s_t), a_t \sim \pi_{\theta}(a_t|s_t)} \left[ \nabla_{\theta} \log(\pi_{\theta}(a_t|s_t)) \mathcal{A}_t \right]$$

- **$\nabla_{\theta} J(\theta)$**：期望回报相对于参数 $\theta$ 的梯度。
- **$d_{\theta}(s_t)$**：在策略 $\pi_{\theta}$ 下的状态分布。
- **$\log(\pi_{\theta}(a_t|s_t))$**：动作概率的对数，梯度运算会沿此方向调整动作概率。
- **$\mathcal{A}_t$**：**优势函数（Advantage Function）**。

### 4. 优势函数与价值函数（Advantage and Value Functions）

优势函数用于衡量在特定状态下采取某个动作比平均水平好多少： $$\mathcal{A}_t = R_t - V(s_t)$$

- **$R_t = \sum_{l=0}^{T-t} \gamma^l r_{t+l}$**：从时间 $t$ 开始到回合结束的**实际折扣回报**。
- **$V(s_t)$**：**价值函数**，它估计了从状态 $s_t$ 开始遵循当前策略所能获得的平均预期回报： $$V(s_t) = \mathbb{E} [R_t | \pi_{\theta}, s_t]$$。

**核心逻辑**：策略梯度的作用是**增加**那些导致比预期回报更高（$\mathcal{A}_t > 0$）的动作的可能性，同时**减少**那些回报低于预期（$\mathcal{A}_t < 0$）的动作的可能性。

在本文的具体实现中，作者采用了**近端策略优化算法（PPO）**，并结合 **TD($\lambda$)** 训练价值函数，使用**广义优势估计（GAE($\lambda$)）**来计算梯度所需的优势值。

如果您需要更深入的公式推导或想了解这些公式在 PPO 算法中是如何具体变形的，我可以为您生成一份**定制报告**。



# MimicKit DeepMimic - g1_walk

```JSON
		{
            "name": "MimicKit: 训练 (Train)",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/mimickit/run.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "stopOnEntry": true,
            "python": "${userHome}/miniconda3/envs/env_mimickit_isaacgym/bin/python",
            "cwd": "${workspaceFolder}",
            "env": {
                "LD_LIBRARY_PATH": "${userHome}/miniconda3/envs/env_mimickit_isaacgym/lib:${env:LD_LIBRARY_PATH}"
            },
            "args": [
                "--mode", "train",
                "--num_envs", "4096",
                "--arg_file", "args/deepmimic_g1_ppo_args.txt",
                "--visualize", "false"
            ]
        }
```

```bash
tensorboard --logdir /home/chong/Desktop/Project/RobotMimic/MimicKit/output
```

```JSON
		{
            "name": "MimicKit: 测试 (Test)",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/mimickit/run.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "stopOnEntry": false,
            "python": "${userHome}/miniconda3/envs/env_mimickit_isaacgym/bin/python",
            "cwd": "${workspaceFolder}",
            "env": {
                "LD_LIBRARY_PATH": "${userHome}/miniconda3/envs/env_mimickit_isaacgym/lib:${env:LD_LIBRARY_PATH}"
            },
            "args": [
                "--mode", "test",
                "--num_envs", "8",
                "--arg_file", "args/deepmimic_g1_ppo_args.txt",
                "--visualize", "true",
                "--model_file", "output/g1_deepmimic_20260215/model.pt"
            ]
        }
```

# MimicKit DeepMimic - g1_walk 理解

