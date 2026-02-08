### 1. 总体流程（顺序图）

```mermaid
sequenceDiagram
    participant User as 用户/命令行
    participant Main as main()
    participant Run as run()
    participant ArgParser as ArgParser
    participant EnvBuilder as env_builder
    participant AgentBuilder as agent_builder
    participant Agent as BaseAgent
    participant Env as BaseEnv

    User->>Main: python run.py --arg_file xxx.txt
    Main->>ArgParser: load_args(argv)
    ArgParser->>ArgParser: load_args(命令行参数)
    ArgParser->>ArgParser: load_file(arg_file)
    ArgParser-->>Main: args (参数字典)

    Main->>Main: 解析 devices, master_port
    Main->>Run: run(rank, num_procs, device, master_port, args)

    Note over Run: 初始化阶段
    Run->>Run: mp_util.init() / set_rand_seed() / set_np_formatting()
    Run->>Run: create_output_dir(out_dir)

    Note over Run: 构建环境
    Run->>EnvBuilder: build_env(env_file, engine_file, num_envs, device, visualize)
    EnvBuilder->>EnvBuilder: load_configs(env_file, engine_file)
    EnvBuilder->>Env: 根据 env_name 实例化具体 Env
    EnvBuilder-->>Run: env

    Note over Run: 构建智能体
    Run->>AgentBuilder: build_agent(agent_file, env, device)
    AgentBuilder->>AgentBuilder: load_agent_file(agent_file) → agent_config
    AgentBuilder->>Agent: 根据 agent_name 实例化具体 Agent(config, env, device)
    AgentBuilder-->>Run: agent

    Note over Run: 加载模型(可选)
    Run->>Agent: agent.load(model_file) [如果指定了model_file]

    alt mode == "train"
        Run->>Agent: train_model(max_samples, out_dir, ...)
        loop while sample_count < max_samples
            Agent->>Agent: _train_iter()
            Agent->>Env: step(action) / reset()
            Agent->>Agent: _update_model()
            Agent->>Agent: test_model() [定期评估]
            Agent->>Agent: save(model_file) [定期保存]
        end
    else mode == "test"
        Run->>Agent: test_model(test_episodes)
        loop 收集足够 episodes
            Agent->>Agent: _decide_action(obs, info)
            Agent->>Env: step(action)
        end
        Agent-->>Run: {mean_return, mean_ep_len, num_eps}
    end
```

### 2. 类关系图

```mermaid
classDiagram
    class ArgParser {
        -_table: dict
        +load_args(arg_strs)
        +load_file(filename)
        +parse_string(key, default)
        +parse_int(key, default)
        +parse_float(key, default)
        +parse_bool(key, default)
        +has_key(key)
    }

    class env_builder {
        +build_env(env_file, engine_file, num_envs, device, visualize)$
        +load_configs(env_file, engine_file)$
        +override_engine_config(env_engine_config, engine_config)$
    }

    class agent_builder {
        +build_agent(agent_file, env, device)$
        +load_agent_file(file)$
    }

    class BaseAgent {
        <<abstract>>
        #_env: BaseEnv
        #_device: str
        #_iter: int
        #_sample_count: int
        #_config: dict
        #_exp_buffer: ExperienceBuffer
        #_obs_norm: Normalizer
        #_optimizer: MPOptimizer
        +train_model(max_samples, out_dir, ...)
        +test_model(num_episodes)
        +save(out_file)
        +load(in_file)
        #_train_iter()
        #_rollout_train(num_steps)
        #_rollout_test(num_episodes)
        #_decide_action(obs, info)*
        #_build_model(config)*
        #_update_model()*
    }

    class DummyAgent {
        +_decide_action(obs, info)
        +_build_model(config)
        +_update_model()
    }

    class PPOAgent {
        +_decide_action(obs, info)
        +_build_model(config)
        +_update_model()
    }

    class AMPAgent {
        +_decide_action(obs, info)
        +_build_model(config)
        +_update_model()
    }

    class ASEAgent { }
    class ADDAgent { }
    class AWRAgent { }

    BaseAgent <|-- DummyAgent
    BaseAgent <|-- PPOAgent
    BaseAgent <|-- AMPAgent
    BaseAgent <|-- ASEAgent
    BaseAgent <|-- ADDAgent
    BaseAgent <|-- AWRAgent

    class BaseEnv {
        <<abstract>>
        +step(action)
        +reset(env_ids)
        +get_obs_space()
        +get_action_space()
    }

    class CharEnv { }
    class DeepMimicEnv { }
    class AMPEnv { }
    class ViewMotionEnv { }

    BaseEnv <|-- CharEnv
    CharEnv <|-- DeepMimicEnv
    CharEnv <|-- AMPEnv
    CharEnv <|-- ViewMotionEnv

    BaseAgent o-- BaseEnv : _env
    agent_builder ..> BaseAgent : 创建
    env_builder ..> BaseEnv : 创建
```

### 3. 训练循环详细流程

```mermaid
flowchart TD
    A[train_model 开始] --> B[创建 Logger / 输出目录]
    B --> C[reset_envs → 获取初始 obs]
    C --> D[_init_train: iter=0, sample_count=0]
    D --> E{sample_count < max_samples?}
    
    E -->|是| F[_train_iter]
    F --> G[_init_iter]
    G --> H[设置 TRAIN 模式]
    H --> I[_rollout_train: 收集 steps_per_iter 步数据]
    
    I --> I1[_decide_action: 策略网络推理]
    I1 --> I2[_record_data_pre_step: 记录 obs, action]
    I2 --> I3[env.step: 执行动作]
    I3 --> I4[_record_data_post_step: 记录 reward, done]
    I4 --> I5[_reset_done_envs: 重置结束的环境]
    I5 --> I6{收集完 steps_per_iter 步?}
    I6 -->|否| I1
    I6 -->|是| J[_build_train_data: 准备训练数据]
    
    J --> K[_update_model: 策略梯度更新]
    K --> L[_update_normalizers: 更新观测归一化]
    L --> M{到输出间隔?}
    
    M -->|是| N[test_model: 评估当前策略]
    N --> O[记录日志 + 保存模型]
    O --> P[reset 环境 & return_tracker]
    P --> Q[iter += 1]
    
    M -->|否| Q
    Q --> E
    
    E -->|否| R[训练结束]

    style A fill:#4CAF50,color:white
    style R fill:#f44336,color:white
    style K fill:#FF9800,color:white
    style N fill:#2196F3,color:white
```

### 4. 配置文件与工厂模式映射

```mermaid
flowchart LR
    subgraph "args.txt 参数文件"
        A1["--env_config<br/>环境配置YAML路径"]
        A2["--engine_config<br/>引擎配置YAML路径"]
        A3["--agent_config<br/>智能体配置YAML路径"]
        A4["--mode train/test"]
        A5["--num_envs 4"]
    end

    subgraph "env_builder 工厂"
        direction TB
        E0{"env_name?"}
        E1[CharEnv]
        E2[DeepMimicEnv]
        E3[AMPEnv]
        E4[ASEEnv]
        E5[ViewMotionEnv]
        E6[TaskLocationEnv]
        E0 --> E1 & E2 & E3 & E4 & E5 & E6
    end

    subgraph "agent_builder 工厂"
        direction TB
        AG0{"agent_name?"}
        AG1[DummyAgent<br/>无agent_config时]
        AG2[PPOAgent]
        AG3[AWRAgent]
        AG4[AMPAgent]
        AG5[ASEAgent]
        AG6[ADDAgent]
        AG0 --> AG1 & AG2 & AG3 & AG4 & AG5 & AG6
    end

    A1 -->|YAML → env_config| E0
    A2 -->|YAML → engine_config| E0
    A3 -->|YAML → agent_config| AG0
```

---

### 流程总结

| 阶段 | 入口函数 | 核心逻辑 |
|------|---------|---------|
| **load_args** | `run.py → load_args()` | `ArgParser` 先解析命令行 `--key value`，再加载 `arg_file` 文件补充参数，存入 `_table` 字典 |
| **build_env** | `run.py → build_env()` | 读取 `env_config` + `engine_config` 两个 YAML，根据 `env_name` 通过工厂模式创建对应的环境实例（如 `ViewMotionEnv`, `AMPEnv` 等） |
| **build_agent** | `run.py → build_agent()` | 读取 `agent_config` YAML，根据 `agent_name` 工厂创建智能体；无配置时默认 `DummyAgent`。Agent 初始化时自动构建 normalizer → model → optimizer → exp_buffer |
| **train** | `BaseAgent.train_model()` | 核心循环：`rollout_train`（收集数据） → `_build_train_data` → `_update_model`（策略更新） → 定期 `test_model` 评估并保存 |
| **test** | `BaseAgent.test_model()` | 切换到 TEST 模式，使用 `torch.no_grad()` 运行策略，收集足够 episodes 后统计 mean_return 等指标 |

**关键设计模式**：
- **工厂模式**：`env_builder` 和 `agent_builder` 都根据 YAML 配置中的名称字符串动态创建具体实例
- **模板方法**：`BaseAgent` 定义了训练/测试的骨架流程，子类只需实现 `_decide_action()`、`_build_model()`、`_update_model()` 三个抽象方法
- **多进程支持**：`main()` 通过 `torch.multiprocessing` 支持多 GPU 训练，每个进程独立调用 `run()`