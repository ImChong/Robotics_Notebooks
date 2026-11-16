---
type: entity
project_id: xpeng-dial
tags: [paper, vla, world-model, robocasa, manipulation, repo, open-source, china-embodied-opensource, project]
status: complete
updated: 2026-11-16
arxiv: "2603.29844"
code: https://github.com/xpeng-robotics/DIAL
project: https://xpeng-robotics.github.io/dial/
institutions:
  - xpeng
related:
  - ./light-o1.md
  - ./robocasa.md
  - ../methods/vla.md
  - ./paper-pi05-open-world-vla.md
  - ./paper-thinkworld.md
  - ../methods/generative-world-models.md
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
  - ../queries/hmi-opensource-projects-coverage.md
sources:
  - ../../sources/papers/dial_arxiv_2603_29844.md
  - ../../sources/repos/dial.md
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
summary: "DIAL（arXiv:2603.29844，小鹏机器人 / HKU / UNC，已开源 xpeng-robotics/DIAL）：在 VLA 中解耦 intent 与 action，用 latent world 建模把语言 ground 为 visual roadmap 再由 flow-matching DiT 生成动作块；收录 RoboCasa GR-1 对照与源码推理评测时序。"
---

# DIAL

**DIAL**（*Decoupling Intent and Action via Latent World Modeling for End-to-End VLA*，Chen et al.，[arXiv:2603.29844](https://arxiv.org/abs/2603.29844)，[项目页](https://xpeng-robotics.github.io/dial/)，[GitHub 代码](https://github.com/xpeng-robotics/DIAL)）由**小鹏机器人（XPENG Robotics）**与香港大学、UNC 合作提出，并作为 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 第一层代表项目公开（Apache-2.0）。论文指出：多数端到端 VLA 把预训练 VLM **仅当 encoder 直出 action**，既浪费 **高层决策**，又带来 **训练不稳定与语义退化**。DIAL 让 VLM 侧学习 **latent world / visual roadmap**，再解耦生成低层动作。

## 一句话定义

**在 VLA 里用 latent 世界模型先把语言指令 ground 成结构化「视觉计划」，再出动作，而不是 VLM 特征一步映射到 motor。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 预训练多模态大模型骨干 |
| GR-1 | General Robot 1 | RoboCasa 人形桌面操作基准平台 |
| E2E | End-to-End | 端到端联合训练 |
| WM | World Model | 预测/表征环境动态的 latent 模块 |

## 为什么重要

- **Light-O1 对照：** [Light-O1 Tech Blog](https://www.lightorigins.com/en/blog/light-o1) 在 **RoboCasa GR-1 24 任务** 上与 GR00T N1.7、π0.5、**DIAL** 等 **同数据训练** 对比；Light-O1 报告 **79.3% macro success**。
- **VLA 结构论战：** 与「VLM+action expert / flow head」路线并列，强调 **intent 层 world modeling**。
- **国内开源全栈坐标：** 收录于 [国内具身智能开源全景](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 第一层「VLA/操作模型」，并与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 及 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 互联，提供可运行推理与评测服务代码。

## 流程总览

```mermaid
flowchart TB
    IN["语言指令 l_t + 当前观测 o_t + 本体状态 q_t"]
    subgraph S2["System-2 (VLM, 如 Qwen2.5-VL-3B)"]
        direction TB
        VIT["冻结的预训练 ViT 提取 o_t 视觉特征"]
        LLM["LLM 处理 视觉 patch + 指令 + N 个可学习 query"]
        MLP["MLP 投影头 → 潜在意图 x_t (latent foresight)"]
        VIT --> LLM --> MLP
    end
    subgraph S1["System-1 (flow-matching 反应式策略 / 潜在逆动力学)"]
        direction TB
        FUS["4 层 self-attention 融合<br/>Enc_ViT(o_t) 与 x_t"]
        DIT["16 层 DiT 动作解码器<br/>(cross-attention 条件 + q_t 投影 token + 噪声动作 token)"]
        FUS --> DIT
    end
    IN --> S2
    MLP -->|"x_t (可微瓶颈)"| FUS
    IN --> FUS
    DIT --> ACT["动作块 A_t = [a_t ... a_(t+H-1)]"]
    GT["真值未来特征 Enc_ViT(o_(t+H))"] -.->|"L_world: MSE 监督 x_t"| MLP
    GT -.->|"Stage 1 解耦热身: 替代 x_t 喂给 System-1"| FUS
    ACT -.->|"Stage 2 端到端: L_fm 梯度经 x_t 回传至 VLM"| MLP
```

图依据：论文 §3.1-3.4 / Fig.1 / Fig.3

## 核心机制

1. **Intent 分支：** VLM 生成/维护 **latent visual roadmap** — 抽象语言 → 连贯空间-时间计划。
2. **Action 分支：** 在 latent 结构上输出 **低层控制**，避免语义特征被 action 梯度破坏。
3. **可视化：** 论文展示 latent 如何把 linguistic instruction **ground** 到 structurally aligned 路线图。

## 源码运行时序图

入口：`bash examples/eval.sh <model_path> <eval_type>`（仿真在线评测）。脚本先以后台方式运行 `python scripts/inference_service_dial.py --server --embodiment_tag gr1 --model_path ... --port ...`，再对每个 GR1 任务调用 `python scripts/simulation_service_dial_augPosRot.py --client --env_name ...`（失败最多重试 5 次）。

```mermaid
sequenceDiagram
    autonumber
    participant SH as examples/eval.sh
    participant SRV as inference_service_dial.py --server
    participant DP as DialPolicy gr00t/model/policy_dial.py
    participant RIS as RobotInferenceServer gr00t/eval/service.py
    participant CLI as simulation_service_dial_augPosRot.py --client
    participant SIC as SimulationInferenceClient gr00t/eval/simulation_augPosRot.py
    participant ENV as gym 环境 robocasa GR1
    participant CS as compute_success_rate.py

    SH->>SRV: nohup 启动推理服务
    SRV->>DP: DialPolicy(model_path, embodiment_tag) 加载 checkpoint
    SRV->>RIS: RobotInferenceServer(policy, port).run() 注册 get_action 与 get_modality_config
    SH->>CLI: 对每个任务调用 client
    CLI->>SIC: SimulationInferenceClient(host, port)
    CLI->>SIC: get_modality_config()
    SIC->>RIS: 请求 get_modality_config 经 ZMQ msgpack
    RIS-->>SIC: 模态配置
    CLI->>SIC: run_simulation(SimulationConfig)
    SIC->>ENV: setup_environment 与 gym.make 及视频录制 wrapper
    SIC->>ENV: env.reset()

    loop 直到完成 n_episodes
        SIC->>RIS: get_action(obs) 经 ZMQ msgpack
        RIS->>DP: get_action(observations)
        DP-->>RIS: actions
        RIS-->>SIC: actions
        SIC->>ENV: env.step(actions) 并累计 success 与奖励
        alt terminations 或 truncations
            SIC->>ENV: env.reset() 开始下一回合
        end
    end
    SIC-->>CLI: env_name 与 episode_successes
    SH->>CS: 汇总 client.log 写入 results.json
```

图依据：xpeng-robotics/DIAL@b611fed，读过 `README.md`（前 120 行）、`examples/eval.sh`（开头与任务循环、末尾段）、`examples/train.sh`、`scripts/inference_service_dial.py`（`main` 段与 grep）、`scripts/simulation_service_dial_augPosRot.py`（`--server` / `--client` 分支）、`gr00t/eval/service.py`（传输与 server 结构 grep）、`gr00t/eval/robot.py`（grep）、`gr00t/eval/simulation_augPosRot.py`（`run_simulation` 与 `_get_actions_from_server` 段）、`gr00t/model/policy_dial.py`（类与方法签名 grep）。`scripts/dual_system_train.py` 训练链路未画、未读。时序图依据以上文件，未运行代码。

## 工程实践

- **官方仓库与服务架构**：[xpeng-robotics/DIAL](https://github.com/xpeng-robotics/DIAL) 基于 GR00T 服务协议组织在线评测与 ZMQ/msgpack 推理服务（`inference_service_dial.py` / `DialPolicy`）。使用前请核对官方 README 的依赖、checkpoint 获取及 Apache-2.0 许可说明。
- **同公司全栈对照**：可通过 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 与 [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md) 对照小鹏其它开源仓是否共用推理与评测底座。

## 实验与评测

- **RoboCasa GR1 Tabletop Simulation** 为主要仿真 benchmark 之一（与 Light-O1 同设定：24 厨房任务、1000 demo/task 训练、50 ep 评测）。
- 具体 macro success **以原文 Table 为准**；Light-O1 blog 将其列为 **published baseline** 而非本库复现数值。

## 与其他工作对比

| 维度 | DIAL（本文） | 直出式端到端 VLA（VLM 当 encoder） | [π0.5](./paper-pi05-open-world-vla.md) |
|------|--------------|------------------------------------|-----------------------------------------|
| VLM 的角色 | 产出 **latent visual roadmap**（高层意图） | 特征提取器，一步映射到 motor | 预训练多模态骨干 + 动作专家 |
| 意图/动作耦合 | **解耦**，动作梯度不直接冲刷语义特征 | 强耦合，易训练不稳/语义退化 | 分层但以动作 token 为主 |
| 主张的收益 | 语言 grounding 更结构化、训练更稳 | 结构简单 | 开放世界泛化 |
| 主要代价 | 多一层 latent 世界建模与其监督 | — | 预训练规模成本 |

- **同 benchmark 才可比：** 本页数值口径是 **RoboCasa GR-1 Tabletop**（24 厨房任务 / 1000 demo per task / 50 ep 评测）；[Light-O1](./light-o1.md) 在同设定下报告更高 macro success，但那是 **该博客引用的 published baseline**，不是本库复现值。
- **与世界模型路线的边界：** DIAL 的 latent world 服务的是 **意图 grounding**，不是像素级未来 rollout；与 [Generative World Models](../methods/generative-world-models.md) 一支的评测口径（预测保真度）不同，别拿视频质量指标横比。
- **逐项数值：** macro success 与消融 **以原文 Table 为准**（[arXiv:2603.29844](https://arxiv.org/abs/2603.29844)）。

## 结论

**DIAL 代表 VLA 设计谱系中「先 latent 计划、后 motor」一支，与 Light-O1 的 human prior + BFM 路线正交但共享 RoboCasa 评测坐标。**

1. **解耦 intent/action** 缓解 VLM 语义被 low-level 梯度侵蚀。
2. **Latent roadmap** 使语言指令可检验是否 **ground 到空间结构**。
3. **RoboCasa GR-1** 是 humanoid tabletop 的 **标准对照场** — 读 Light-O1 79.3% 必对照 DIAL 等发表线。
4. **开源已可查验**：官方仓库 [xpeng-robotics/DIAL](https://github.com/xpeng-robotics/DIAL) 公开了服务化推理与 RoboCasa 仿真评测链路，训练链路与具体复现以官方仓最新提交为准。
5. **部署** 仍依赖仿真 teleop 数据分布；真机 zero-shot 未作为本文主 claim。

## 关联页面

- [ThinkWorld](./paper-thinkworld.md) — 用未来特征预测强化动作推理，并利用人类视频学习技能组合
- [Light-O1](./light-o1.md) — 同 benchmark 更强 reported macro success
- [RoboCasa](./robocasa.md) — 仿真环境与数据协议
- [π0.5](./paper-pi05-open-world-vla.md) — 同类 VLA 对照
- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [dial_arxiv_2603_29844.md](../../sources/papers/dial_arxiv_2603_29844.md)
- [DIAL 源码归档](../../sources/repos/dial.md)（<https://github.com/xpeng-robotics/DIAL>）
- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 论文：<https://arxiv.org/abs/2603.29844>
- [arXiv PDF](https://arxiv.org/pdf/2603.29844)：已通读方法部分（实验与附录以略读为准）：摘要、§3（模型总览、System-2、System-1、两阶段训练）、§4.1 实验设定
- 源码核对（2026-11-02）：xpeng-robotics/DIAL@b611fed，读过 `README.md`（前 120 行）、`examples/eval.sh`、`examples/train.sh`、`scripts/inference_service_dial.py`、`scripts/simulation_service_dial_augPosRot.py`、`gr00t/eval/service.py`、`gr00t/eval/robot.py`（grep）、`gr00t/eval/simulation_augPosRot.py`（部分）、`gr00t/model/policy_dial.py`（grep）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [DIAL 官方 GitHub 仓库](https://github.com/xpeng-robotics/DIAL)
- [小鹏机器人 官方组织](https://github.com/xpeng-robotics)
- [Light-O1 Tech Blog — Humanoid Manipulation 节](https://www.lightorigins.com/en/blog/light-o1)
