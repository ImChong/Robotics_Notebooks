---
type: entity
tags: [paper, world-action-models, wam, vla, world-models, embodied-ai, robot-manipulation, long-context, nvidia, mit, hku, ucsd]
status: complete
updated: 2026-10-11
arxiv: "2610.10528"
code: https://github.com/NVlabs/LongLive/tree/main/Long-WAM
project: https://nvlabs.github.io/LongLive/Long-WAM/
code_scope: "Long-WAM/"
project_distinction: "Separately maintained research project in the NVlabs/LongLive monorepo; distinct from LongLive video-generation releases."
project_id: long-wam
summary: "Long-WAM 将因果视频预测专家与动作专家结合，在固定动作块和未来预测跨度下扩展观测历史；代码位于 NVlabs/LongLive 的 Long-WAM 子目录。"
related:
  - ../concepts/world-action-models.md
  - ../methods/vla.md
  - ../methods/generative-world-models.md
  - ../tasks/manipulation.md
  - ./paper-uniwam-unified-world-action-model.md
  - ../overview/frontier-manipulation-2026-10-05-10-09.md
sources:
  - ../../sources/papers/long-wam-arxiv-2610-10528.md
  - ../../sources/repos/nvlabs-longlive-long-wam.md
  - ../../sources/repos/lerobot.md
  - ../../sources/sites/long-wam-project-page.md
  - ../../sources/blogs/frontier_manipulation_2026_10_05_10_09.md
---

# Long-WAM：Scaling the Context of World-Action Models

**Long-WAM**（*Long-WAM: Scaling the Context of World-Action Models*）研究因果世界–动作模型（WAM）如何利用更长的观测历史。它保持未来预测跨度与动作块长度固定，只扩展模型可见的过去上下文，并将因果视频预测专家与动作专家组合成一套策略。

- **论文：** [arXiv:2610.10528](https://arxiv.org/abs/2610.10528) · [HTML v1](https://arxiv.org/html/2610.10528v1) · [PDF](https://arxiv.org/pdf/2610.10528)
- **项目页：** [Long-WAM](https://nvlabs.github.io/LongLive/Long-WAM/)
- **官方代码：** [NVlabs/LongLive/Long-WAM](https://github.com/NVlabs/LongLive/tree/main/Long-WAM)（项目子目录；Apache-2.0，第三方组件按各自许可）
- **模型：** [Hugging Face collection](https://huggingface.co/collections/Efficient-Large-Model/long-wam)
- **作者：** Wei Huang、Bohan Zhang、Chenzhi Liu、Isabella Liu、Shuai Yang、Weian Mao、Luozhou Wang、Yicheng Xiao、Weifeng Lin、Qixin Hu、Bryan Chu、Sifei Liu、Linxi Fan、Xiaojuan Qi、Song Han、Yukang Chen
- **机构：** NVIDIA、MIT、The University of Hong Kong、UC San Diego；Wei Huang 与 Bohan Zhang 为共同第一作者。
- **范围：** Long-WAM 与同仓库的 LongLive 视频生成代码是不同项目；本节点把论文、Long-WAM 代码子目录和模型资源合并为一个独立实体。

## 一句话理解

**在稳定阶段继续较长动作，在抓取、接触等关键阶段利用更长视觉历史预测未来，再据此生成动作。**

## 英文缩写速查

| 缩写 | 英文全称 | 本文含义 |
|---|---|---|
| WAM | World-Action Model | 联合学习环境未来变化与机器人动作的具身模型 |
| AR | Autoregressive | 自回归；按时间顺序逐步预测未来视频 latent |
| MoT | Mixture-of-Transformers | 多个模态专家通过注意力接口交换上下文的架构 |
| IDM | Inverse Dynamics Model | 从状态变化估计动作监督或动作条件的逆动力学模型 |
| SR | Success Rate | 成功率，需结合具体 benchmark 协议解释 |
| KV cache | Key-Value cache | Transformer 历史注意力键值缓存 |
| GPT | Generative Pre-trained Transformer | 此处指承担高层规划、恢复和纠正的 GPT-6 Astra agent |

## 流程总览

下图是 LeRobot 移植里两种推理模式（`LongWAMConfig.denoising` 取 `idm` 或 `cod`）的数据流，来自 `modeling_longwam.py: LongWAMPolicy._predict`；论文级流程图见下文「方法」一节。

```mermaid
flowchart TD
    OBS["每个控制 tick 的当前帧 + 状态 + 指令"] --> HIS["_observe 入队，_history 按 video_stride 取历史帧"]
    HIS --> VAE["Wan VAE 编码历史 -> clean latent"]
    OBS --> CTX["UMT5 文本 + proprio 编码 -> context"]
    VAE --> MODE{"denoising 模式"}
    CTX --> MODE
    MODE -->|"idm"| IM["先对未来视频 latent 去噪（imagine_steps），再缓存视频 KV"]
    IM --> AD1["动作 expert 读视频 KV 去噪动作"]
    MODE -->|"cod"| CO["视频与动作每一步联合去噪"]
    AD1 --> OUT["动作 chunk -> 执行 n_action_steps 步"]
    CO --> OUT
    OUT --> OBS
```

## 方法：记住过去、预测未来、生成动作

常见策略根据当前观测生成一段动作（action chunk），再按固定步数重规划。稳定搬运时频繁重规划会增加成本；接触、抓取和放置时，细小状态变化又可能改变后续动作。Long-WAM 的问题是怎样让因果策略利用更长历史，而不是把“长动作块”与“长观测历史”混为一谈。

视频专家在因果约束下预测未来视觉 latent；动作专家读取观测历史和预测 latent，生成动作块。视频专家不读取动作 token，以保持从观测历史向未来预测的方向。推理时在 latent 空间工作，无需解码完整未来 RGB 视频。

```mermaid
flowchart TB
  data["机器人与第一视角视频"] --> pretrain["LongLive 2.0 Robot 视频自回归预训练"]
  pretrain --> adapt["因果双专家适配"]
  history["长观测历史"] --> video["视频专家预测未来 latent"]
  adapt --> video
  video --> action["动作专家生成动作块"]
  task["任务指令与机器人状态"] --> action
  action --> chunk["执行一个动作块"]
  chunk --> history
```

- **视频专家：** 按自回归次序从历史视觉 latent 预测未来 latent，学习场景和交互变化。
- **动作专家：** 条件去噪/动作生成，利用历史、任务信息、未来表征生成连续动作。
- **非对称注意力：** 动作侧读取预测未来；视频侧不读取动作 token。两侧通过 MoT 风格接口交换上下文。
- **上下文扩展：** 将历史扩展至 38.4 秒，同时不同比例增加动作块或未来跨度。评估配置包括 0、2.4、4.8、9.6、19.2、38.4 秒。
- **训练数据：** 项目页称 LongLive-2.0-Robot 从 LongLive-2.0 checkpoint 继续训练，汇集约 10,000 个窗口等价小时的机器人/第一视角视频，包括 RoVid-X、AgiBot World、EgoDex、EgoVerse、VITRA；未来视频预测适配不需要动作标签。

## 论文报告结果

以下是作者/项目页报告值，本次 ingest 未在模拟器或真机上复测。

| 设置 | 报告结果 | 阅读边界 |
|---|---:|---|
| RoboCasa GR-1，历史从 0 秒扩至 19.2 秒 | SR 从 **63.3%** 升至 **78.7%** | 项目页另列 2.4 秒为 66.3%，应按原始配置区分 |
| LIBERO-Long | **99.5% SR** | 特定仿真基准与评估协议 |
| RoboTwin 2.0 | **94.4% SR** | 不代表任意机器人平台 |
| 动态杯叠放真机演示 | **19/20（95%）** | 单项任务、样本量有限 |
| 单动作块延迟 | **107.4 ms** | 作者报告 RTX 5090 测量，包含未来 latent 预测，不是独立复测 |
| G1 移动杯抓取 | 3.0 / 4.5 / 6.0 / 7.5 cm/s 下为 100 / 100 / 95 / 90%（各 20 次） | 两个基线在 6.0 与 7.5 cm/s 均未成功；作者报告，样本量有限 |
| YAM 长时操作 | >40 秒任务三任务均值 **81.7%**（各 20 次） | 长时任务样例，不等于任意任务时长下的成功率 |
| GPT-6 Astra + Long-WAM，RoboCasa365 | 总体 31.4%→**54.4%**；未见组合 6.1%→**35.0%** | 官网称 Long-WAM checkpoint 不变；属于高层规划器 + 执行器组合结果，不是 Long-WAM 单体收益 |
| 38.4 秒历史配置 | 相较 19.2 秒结果回落 | 页面指出长窗口受较短训练轨迹和 padding 影响；不能直接推断记忆上限 |

项目页还展示 G1/YAM 真机结果和 GPT-6 Astra 组合。规划器组合中 Long-WAM 负责执行带历史条件的动作块，高层规划器负责任务分解、失败恢复与动作纠正；因此 54.4% 是组合系统成绩。需区分真实执行任务结果与未来视频预测演示：生成的视频展示可能未来，不表示机器人已按该序列成功执行动作。

## 高层规划器与 Long-WAM 的组合

官网展示 GPT-6 Astra 与同一 Long-WAM checkpoint 组合，在 RoboCasa365 的 50 个任务上总体成功率从 Long-WAM 单独运行的 31.4% 提高至 54.4%；未见组合从 6.1% 提高至 35.0%。该设计将全任务规划、子指令分解、失败恢复和动作纠正交给高层 agent，由 Long-WAM 执行带长视觉历史的动作块。它说明 WAM 可以作为层级机器人系统的执行器；这些数字属于组合系统，不能作为 Long-WAM checkpoint 单体的改进。

## 与其他工作对比

| 工作 | 怎样处理长历史 / 未来预测 | 与 Long-WAM 的差别 |
|---|---|---|
| [MemoryWAM](./paper-memorywam.md) | 滑窗近期帧 + 任务起点 anchor + gist token 压缩长历史，推理复杂度 O(N)→O(N/d) | MemoryWAM 主动压缩历史；Long-WAM 直接拉长因果视频专家可见的原始 latent 历史，并系统扫描 0–38.4 秒的上下文长度 |
| [UniWAM](./paper-uniwam-unified-world-action-model.md) | 8B MoT 联合物理语义推理、未来视觉与动作生成 | 同为 MoT 式多专家；UniWAM 的重点是加入推理模态与多源预训练，Long-WAM 的重点是上下文长度这一单一变量 |
| [Fast-WAM](./paper-fast-wam.md) | 训练期保留视频共训，推理期跳过未来视频去噪 | 两者都在推理期不解码完整未来 RGB；Fast-WAM 直接省略未来预测，Long-WAM 保留未来 latent 预测并作为动作专家输入 |

阅读边界：上表各项的 RoboTwin 2.0 / LIBERO 数字来自各自论文设定，未在统一协议下复测；Long-WAM 的贡献更适合理解为“固定动作块与未来跨度、只改历史长度”的受控实验，而不是与上述方法的直接排名。

## 代码、权重与复现范围

代码位于 [NVlabs/LongLive 的 Long-WAM/ 子目录](https://github.com/NVlabs/LongLive/tree/main/Long-WAM)，不是独立 Git 仓库。README 提供模型配置、训练/评估脚本和异步部署组件；任务涉及 LIBERO、RoboTwin 2.0、DOMINO、RoboCasa GR-1 / RoboCasa365。官方 Hugging Face collection 有不同机器人和上下文长度的模型。

- 各 benchmark 使用独立配置；README 描述 LIBERO、RoboTwin 的 IDM 与 CodeDenoise 路径，以及 RoboCasa GR-1 多历史长度评估。
- 异步部署将模型推理和机器人动作执行流水化；包含 LeRobot policy interface 和 LIBERO、RoboTwin 2、YAM、Franka、Unitree G1 等适配/转换器。
- 项目描述 RTX 5090、DGX Spark、Jetson AGX Thor 部署路径；实际运行仍依赖 checkpoint、依赖、模拟器资产和适配的机器人接口。
- 单张 GR-1 checkpoint 卡示例采用 20 Hz、动作块 16、单路第一视角 RGB、58 维本体状态和 29 维动作。不可泛化到所有本体或 checkpoint。
- Long-WAM 代码标注 Apache-2.0；第三方代码、模型、数据和模拟器资产按各自许可。部分 Hugging Face 模型卡标注为 `other`，使用前应逐个检查权重条款。

## 复现边界与局限

- `Long-WAM/docs/VERIFICATION.md` 记录源代码快照上的 CPU 回归测试和 CLI 检查通过；GPU benchmark、完整模拟器评测和物理机器人任务未在该验证记录中运行。
- `Long-WAM/docs/BENCHMARKS.md` 将完整 benchmark reproduction 标为未验证。CPU 测试不能验证论文成功率、GPU 延迟或真机结果；验证环境也不是针对官方依赖锁定的全新安装。
- 长于训练轨迹的历史可能大量由 padding 占据，因此 38.4 秒结果下降不能简单归因为上下文机制失效或模型记忆上限。
- 增长观测历史不等于自适应改变动作块长度；动作何时中断或重规划还取决于部署控制逻辑。
- 基准结果与特定本体、数据和任务划分绑定；不能据此推断在所有环境下都优于短历史策略。

## LeRobot 实现

- **身份核对**：LeRobot `docs/source/longwam.mdx` 链接的论文即 arXiv 2610.10528，arXiv 标题 *Long-WAM: Scaling the Context of World-Action Models*（2026-10-07，API 核对），与本页一致；LeRobot 策略源自官方 `NVlabs/LongLive/Long-WAM`。
- **模式**：支持 IDM（先想象未来视频再去噪动作）与 COD（视频和动作联合去噪）两种，对应两个原生检查点 `Efficient-Large-Model/Long-WAM-LIBERO-IDM-LeRobot` 与 `...-COD-LeRobot`；原始 `model.pt` 包不能直接加载（文档）。
- **LIBERO 配置**（文档）：`past_obs_steps=48`（20 Hz 下 2.4 s）、`video_stride=4`、`future_video_steps=32`、`action_horizon=32`、`inference_action_horizon=80`、`n_action_steps=10`；两路 224x224 相机，7 维动作，8 维状态；相机顺序必须是 `observation.images.image`、`observation.images.image2`。
- **LeRobot 文档报告的 LIBERO 结果**（每模型每套件 500 episode，GB200，BF16）：IDM 平均 99.00%，COD 平均 98.45%（分套件数字见文档）。这是 LeRobot 移植检查点的结果，不等同于论文表中的数字。
- **使用要点**：必须在每个控制 tick 调 `select_action`（包括执行队列动作的 tick），否则历史过期；`reset()` / `drop_queued_actions()` 清空历史与队列；通用 async/RTC 未验证（文档）；默认 `inference_seed=1`，每个 chunk 重置同一 seed 以保持发布时的推理行为。
- **精度与夹爪**：`--policy.dtype` 在加载前决定参数存储精度；发布的夹爪后处理会把最后一维从 [0,1] 转成带符号指令，微调数据已是带符号约定时设 `--policy.toggle_action_dimensions=[]`。
- **一致性提示**：下方「源码运行时序图」是官方子目录视角的通用抽象（Async control runner 等），不是 LeRobot 源码函数名；LeRobot 真实调用链见本节后的图。

### LeRobot 移植的运行时序（固定提交 `9e5ee98`）

来源：`src/lerobot/policies/longwam/modeling_longwam.py`、`mot_longwam.py`（`LongWAMDiT`、`ActionExpert`）。

```mermaid
sequenceDiagram
    participant TR as "lerobot_train.py: train()"
    participant EV as "lerobot_eval.py: rollout()"
    participant P as "LongWAMPolicy"
    participant D as "mot_longwam.py: LongWAMDiT"
    TR->>P: "policy(batch) -> forward"
    P->>P: "_context (UMT5 + proprio_encoder), _encode_video (Wan VAE)"
    P->>P: "_add_noise 加噪视频与动作；前 clean_frames 帧保持干净"
    P->>D: "dit(noisy_video, noisy_action, video_time, action_time, context)"
    D-->>P: "video_prediction, action_prediction"
    P->>P: "_flow_loss 加权 -> loss_video, loss_action"
    EV->>P: "select_action(observation) 每 tick"
    P->>P: "_observe: 当前帧入 deque（maxlen = past_obs_steps+1）"
    P->>P: "队列空 -> _predict"
    P->>P: "_history -> _encode_video -> clean latent；_context(cache=True)"
    alt "denoising == idm"
        loop "imagine_steps"
            P->>D: "dit(clean+future, None, ...) 去噪未来视频"
        end
        P->>D: "dit(..., return_cache=True) 得到 video_cache"
        loop "num_inference_steps"
            P->>D: "dit(None, action, ..., video_cache) 去噪动作"
        end
    else "denoising == cod"
        loop "num_inference_steps"
            P->>D: "dit(clean+future, action, time, time, context) 联合去噪"
        end
    end
    P-->>EV: "取前 n_action_steps 步入队，popleft"
```

## 关联页面

- [World Action Models（WAM）](../concepts/world-action-models.md) — 定义、谱系与代表项目索引。
- [UniWAM](./paper-uniwam-unified-world-action-model.md) — 将物理语言推理、未来视频和动作生成联合到 MoT 的另一种配方。
- [VLA](../methods/vla.md) — 视觉语言动作策略。
- [生成式世界模型](../methods/generative-world-models.md) — 视频/latent 未来预测背景。
- [机器人操作](../tasks/manipulation.md) — 操作任务和评测背景。
- [论文来源归档](../../sources/papers/long-wam-arxiv-2610-10528.md)
- [代码来源归档](../../sources/repos/nvlabs-longlive-long-wam.md)
- [项目页归档](../../sources/sites/long-wam-project-page.md)
- [【10.5-10.9前沿论文动态】Manipulation34篇](../overview/frontier-manipulation-2026-10-05-10-09.md)

## 结论

Long-WAM 展示了增加可见历史可以提升若干 WAM 长程操作基准表现，并让未来视频 latent 为动作生成提供预测线索。论文报告了 LIBERO-Long、RoboTwin 2.0、RoboCasa GR-1 和真机动态杯叠放结果；但仓库说明完整 benchmark 尚未复现，长窗口收益依赖训练轨迹覆盖，真机任务样本有限。使用它作基线时，应核对 checkpoint、历史长度、动作块和机器人接口，并分别验证仿真成功率与实际部署延迟。

## 源码运行时序图

```mermaid
sequenceDiagram
  participant Env as Robot or simulator
  participant Runner as Async control runner
  participant Video as Causal video expert
  participant Action as Action expert
  participant Policy as LeRobot policy adapter
  loop Each replanning cycle
    Env->>Runner: Observation state instruction
    Runner->>Video: History latents and task context
    Video-->>Runner: Predicted future latents
    Runner->>Action: History future latents and robot state
    Action-->>Runner: Denoised action chunk
    Runner->>Policy: Convert chunk to robot command
    Policy->>Env: Execute selected action steps
    Env-->>Runner: Next observation and state
  end
```

## 参考来源

- [`sources/repos/lerobot.md`](../../sources/repos/lerobot.md) — LeRobot 固定提交上的 `longwam` 移植与文档
- [论文 arXiv](https://arxiv.org/abs/2610.10528) · [HTML v1](https://arxiv.org/html/2610.10528v1)
- [Long-WAM 官方项目页](https://nvlabs.github.io/LongLive/Long-WAM/)
- [代码与复现文档](https://github.com/NVlabs/LongLive/tree/main/Long-WAM)
- [Hugging Face 检查点集合](https://huggingface.co/collections/Efficient-Large-Model/long-wam)
- [GR-1 检查点示例卡](https://huggingface.co/Efficient-Large-Model/Long-WAM-RoboCasa-GR1-4.8s)
- [【10.5-10.9前沿论文动态】Manipulation34篇 综述条目](../../sources/blogs/frontier_manipulation_2026_10_05_10_09.md)

