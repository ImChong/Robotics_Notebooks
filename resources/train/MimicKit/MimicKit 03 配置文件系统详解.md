### 1. 配置文件的整体架构

```mermaid
flowchart TB
    subgraph "入口: args/*.txt 参数文件"
        ARGS["deepmimic_humanoid_ppo_args.txt<br/>━━━━━━━━━━━━━━━━━<br/>--num_envs 4096<br/>--engine_config data/engines/xxx.yaml<br/>--env_config data/envs/xxx.yaml<br/>--agent_config data/agents/xxx.yaml<br/>--mode train<br/>--out_dir output/"]
    end

    ARGS -->|"--engine_config"| EC
    ARGS -->|"--env_config"| EVC
    ARGS -->|"--agent_config"| AC

    subgraph ENGINE_CFG ["engine_config — 物理引擎配置"]
        EC["isaac_gym_engine.yaml<br/>━━━━━━━━━━━━━━━━<br/>engine_name: isaac_gym<br/>control_mode: pos<br/>control_freq: 30<br/>sim_freq: 120<br/>env_spacing: 5"]
    end

    subgraph ENV_CFG ["env_config — 环境/任务配置"]
        EVC["deepmimic_humanoid_env.yaml<br/>━━━━━━━━━━━━━━━━<br/>env_name: deepmimic<br/>char_file: humanoid.xml<br/>motion_file: xxx.pkl<br/>episode_length: 10.0<br/>key_bodies: [...]<br/>reward_pose_w: 0.5<br/>..."]
    end

    subgraph AGENT_CFG ["agent_config — 智能体/算法配置"]
        AC["deepmimic_humanoid_ppo_agent.yaml<br/>━━━━━━━━━━━━━━━━<br/>agent_name: PPO<br/>model: {actor_net, critic_net, ...}<br/>optimizer: {type, lr}<br/>discount: 0.99<br/>steps_per_iter: 32<br/>ppo_clip_ratio: 0.2<br/>..."]
    end

    EC -->|"传入"| ENV_BUILD["env_builder.build_env()"]
    EVC -->|"传入"| ENV_BUILD
    ENV_BUILD -->|"创建"| ENV_INST["具体 Env 实例<br/>(如 DeepMimicEnv)"]

    AC -->|"传入"| AGENT_BUILD["agent_builder.build_agent()"]
    ENV_INST -->|"传入"| AGENT_BUILD
    AGENT_BUILD -->|"创建"| AGENT_INST["具体 Agent 实例<br/>(如 PPOAgent)"]

    style ARGS fill:#E3F2FD,stroke:#1565C0,color:#000
    style ENGINE_CFG fill:#FFF3E0,stroke:#E65100
    style ENV_CFG fill:#E8F5E9,stroke:#2E7D32
    style AGENT_CFG fill:#F3E5F5,stroke:#6A1B9A
```

### 2. 三类配置的职责划分

```mermaid
mindmap
    root((配置文件系统))
        engine_config<br/>物理引擎层
            engine_name<br/>引擎选择
                isaac_gym
                isaac_lab
                newton
            sim_freq<br/>仿真频率 Hz
            control_freq<br/>控制频率 Hz
            control_mode<br/>控制模式 pos/vel
            env_spacing<br/>环境间距
            ground_contact_height
        env_config<br/>环境/任务层
            env_name<br/>环境类型路由
                view_motion
                deepmimic
                amp / ase / add
                task_location / task_steering
            char_file<br/>角色模型 XML
            motion_file<br/>动作数据 PKL
            key_bodies<br/>关键身体部位
            episode_length<br/>回合时长
            奖励权重 reward_*_w
            奖励缩放 reward_*_scale
            终止条件 termination
        agent_config<br/>算法/网络层
            agent_name<br/>算法类型路由
                Dummy / PPO / AWR
                AMP / ASE / ADD
            model 网络结构
                actor_net
                critic_net
                disc_net 判别器
            optimizer 优化器
                type SGD
                learning_rate
            RL 超参数
                discount
                steps_per_iter
                ppo_clip_ratio
                td_lambda
            AMP 特有参数
                disc_buffer_size
                disc_reward_scale
                task_reward_weight
```

### 3. 配置加载与覆盖机制

```mermaid
sequenceDiagram
    participant CLI as 命令行 / args.txt
    participant AP as ArgParser
    participant EB as env_builder
    participant YAML as YAML 文件

    CLI->>AP: --arg_file args/deepmimic_humanoid_ppo_args.txt
    AP->>AP: load_args(命令行参数)
    AP->>AP: load_file(arg_file) 读取文件中的参数
    Note over AP: _table = {<br/> "num_envs": ["4096"],<br/> "engine_config": ["data/engines/...yaml"],<br/> "env_config": ["data/envs/...yaml"],<br/> "agent_config": ["data/agents/...yaml"],<br/> "out_dir": ["output/"]<br/>}

    AP->>EB: parse_string("env_config") → env_file 路径
    AP->>EB: parse_string("engine_config") → engine_file 路径

    EB->>YAML: load_config(env_file)
    YAML-->>EB: env_config (dict)
    EB->>YAML: load_config(engine_file)
    YAML-->>EB: engine_config (dict)

    Note over EB: 检查 env_config 中是否有<br/>"engine" 字段

    alt env_config 包含 "engine" 字段
        EB->>EB: override_engine_config()<br/>env 中的 engine 参数覆盖 engine_config
        Note over EB: 优先级: env_config.engine > engine_config<br/>例如 ViewMotionEnv 会强制<br/>sim_freq = control_freq
    end

    EB-->>EB: 返回最终的 (env_config, engine_config)
```

### 4. 不同实验方案的配置组合

```mermaid
flowchart LR
    subgraph "引擎选择 (3 种)"
        E1["🔧 isaac_gym_engine.yaml<br/>sim_freq: 120"]
        E2["🔧 isaac_lab_engine.yaml<br/>sim_freq: 120"]
        E3["🔧 newton_engine.yaml<br/>sim_freq: 240"]
    end

    subgraph "环境 × 角色 (31 种)"
        direction TB
        EV1["view_motion_humanoid"]
        EV2["deepmimic_humanoid"]
        EV3["amp_humanoid"]
        EV4["ase_humanoid"]
        EV5["add_humanoid"]
        EV6["amp_g1 / go2 / smpl / ..."]
        EV7["task_location / steering"]
    end

    subgraph "算法选择 (18 种)"
        direction TB
        A1["PPO (deepmimic_humanoid)"]
        A2["AWR (deepmimic_humanoid)"]
        A3["AMP (humanoid / g1 / ...)"]
        A4["ASE (humanoid)"]
        A5["ADD (humanoid / g1 / ...)"]
        A6["Dummy (无 agent_config)"]
    end

    E1 & E2 & E3 -.->|任选一个| COMBO
    EV1 & EV2 & EV3 & EV4 & EV5 & EV6 & EV7 -.->|任选一个| COMBO
    A1 & A2 & A3 & A4 & A5 & A6 -.->|任选一个| COMBO

    COMBO((组合成<br/>args.txt))

    style COMBO fill:#FFEB3B,stroke:#F57F17,color:#000
```

---

### 每个配置文件的字段详解

#### engine_config — 物理引擎配置

| 字段 | 类型 | 含义 | 示例 |
|------|------|------|------|
| `engine_name` | string | 物理引擎选择，决定用哪个仿真器 | `"isaac_gym"` / `"isaac_lab"` / `"newton"` |
| `control_mode` | string | 控制方式 | `"pos"` (位置控制) |
| `control_freq` | int | 控制频率 (Hz)，Agent 做决策的频率 | `30` |
| `sim_freq` | int | 仿真频率 (Hz)，物理模拟步进频率 | `120` (即每个控制步内仿真 4 次) |
| `env_spacing` | float | 多环境间的空间间距 (m) | `5` |
| `ground_contact_height` | float | 地面接触检测高度阈值 | `0.3` |

#### env_config — 环境/任务配置

| 字段 | 类型 | 含义 | 谁消费 |
|------|------|------|--------|
| `env_name` | string | **路由键**，决定创建哪个 Env 类 | `env_builder` |
| `char_file` | string | 角色 MJCF/XML 模型文件路径 | `CharEnv` |
| `motion_file` | string | 参考动作数据文件 | `ViewMotionEnv` / `DeepMimicEnv` 等 |
| `key_bodies` | list | 关键身体部位名称，用于奖励计算和可视化 | 各 Env |
| `contact_bodies` | list | 允许接触地面的身体部位 | `DeepMimicEnv` |
| `episode_length` | float | 回合最大时长 (秒) | `BaseEnv` |
| `reward_*_w` | float | 各奖励分量权重 | 各 Env |
| `reward_*_scale` | float | 各奖励分量缩放因子 | 各 Env |
| `init_pose` | list | 初始姿态 (根位置 + 关节角) | `CharEnv` |
| `engine` (可选) | dict | 覆盖 engine_config 的参数 | `env_builder.override_engine_config()` |

#### agent_config — 智能体/算法配置

| 字段 | 类型 | 含义 | 谁消费 |
|------|------|------|--------|
| `agent_name` | string | **路由键**，决定创建哪个 Agent 类 | `agent_builder` |
| `model.actor_net` | string | Actor 网络结构 | `BaseAgent._build_model()` |
| `model.critic_net` | string | Critic 网络结构 | `BaseAgent._build_model()` |
| `model.disc_net` | string | 判别器网络 (AMP 专属) | `AMPAgent` |
| `optimizer.type` | string | 优化器类型 | `MPOptimizer` |
| `optimizer.learning_rate` | float | 学习率 | `MPOptimizer` |
| `discount` | float | 折扣因子 γ | `BaseAgent` |
| `steps_per_iter` | int | 每次迭代采集步数 | `BaseAgent._rollout_train()` |
| `iters_per_output` | int | 每多少次迭代输出/评估一次 | `BaseAgent.train_model()` |
| `ppo_clip_ratio` | float | PPO 裁剪比率 | `PPOAgent` |
| `td_lambda` | float | GAE lambda 参数 | `PPOAgent` |
| `disc_*` | various | 判别器相关超参数 (AMP/ASE/ADD) | 对应 Agent |

### 核心设计理念

**三层解耦，自由组合**：配置系统将"用什么引擎仿真"、"仿真什么环境和任务"、"用什么算法训练"三个维度彻底分离。你可以像搭积木一样，通过修改 `args.txt` 中的三行路径来自由组合不同的引擎、环境和算法，而不需要修改任何代码。

**覆盖机制**：`env_config` 中可选的 `engine` 字段能够覆盖 `engine_config` 中的值（如 `ViewMotionEnv` 强制让 `sim_freq = control_freq`），这使得特殊环境可以对引擎参数做强制约束，同时保持通用引擎配置的复用性。