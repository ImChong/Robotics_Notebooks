---
type: entity
tags: [vla, open-source, moe, flow-matching, multi-embodiment, humanoid, qwen, manipulation, loco-manipulation, depth-distillation, robbyant]
status: complete
updated: 2026-11-05
arxiv: "2607.06403"
code: https://github.com/robbyant/lingbot-vla-v2
related:
  - ./lingbot-vla.md
  - ../entities/paper-sa-2607-07534-infinite-worlds-with-versatile-interactions-ling.md
  - ../methods/vla.md
  - ../methods/being-h07.md
  - ../methods/egoscale.md
  - ../tasks/manipulation.md
  - ../tasks/loco-manipulation.md
  - ./humannet.md
  - ./qwen-vla.md
  - ./xiaomi-robotics-0.md
  - ./paper-greenvla-staged-vla-humanoid.md
  - ../methods/lingbot-map.md
  - ../entities/lerobot.md
  - ./letools.md
sources:
  - ../../sources/papers/lingbot_vla_v2_tech_report.md
  - ../../sources/repos/lingbot-vla-v2.md
  - ../../sources/sites/lingbot-vla-v2-technology-robbant.md
  - ../../sources/repos/lerobot.md
summary: "LingBot-VLA 2.0（6B）：Qwen3-VL-4B + 稀疏 MoE action expert，约 6 万小时异构预训练（5 万 h 机器人 ×20 本体 + 1 万 h egocentric 人视频）、55 维统一全身动作空间与 Dual-Query 深度/视频蒸馏；GM-100 与长程移动操作 generalist 评测超 π₀.₅ / GR00T N1.7，开源权重与真机部署脚本。"
---

# LingBot-VLA 2.0

**LingBot-VLA 2.0**（*From Foundation to Application: Improving VLA Models in Practice*，[arXiv:2607.06403](https://arxiv.org/abs/2607.06403)，[Robbyant](https://github.com/robbyant/lingbot-vla-v2)）是面向 **真机规模化应用** 的 **Vision-Language-Action 基础模型**：在 **Qwen3-VL-4B-Instruct** 上叠加 **稀疏 MoE action expert** 与 **native depth** 蒸馏分支，用 **重设计数据管线**、**55 维统一动作空间** 与 **Dual-Query 预测性蒸馏**，把 LingBot-VLA 1.0 的「双臂操作」扩展到 **全身 DoF + 移动底盘 + 灵巧手**，并在 **GM-100** 与 **长程移动操作** 的 **generalist 联合训练** 设定下报告优于 **π₀.₅**、**GR00T N1.7** 与 **1.0** 的对照结果。

## 一句话定义

**6B 务实 VLA 基础模型：用 6 万小时过滤后的机器人+人视频预训练、55 维跨本体动作槽位与 MoE 专家，把 Qwen3-VL 语义能力接到可部署的全身操纵与移动操作闭环。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| MoE | Mixture of Experts | 混合专家：稀疏激活子网络以扩展容量 |
| EEF | End-Effector | 末端执行器位姿/状态 |
| VLM | Vision-Language Model | 视觉-语言多模态理解模型，VLA 的上游 |
| Ego | Egocentric Vision | 第一人称视角感知与控制 |
| GM-100 | General Manipulation 100 | 论文/项目使用的双臂桌面 generalist 评测套件（9 任务混合训练） |
| SFT | Supervised Fine-Tuning | 用监督数据将通用模型适配到特定任务分布 |

## 为什么重要

- **从基础到应用的完整开源链：** 除技术报告外，公开 **6B 预训练权重**（[HF](https://huggingface.co/robbyant/lingbot-vla-v2-6b) / [ModelScope](https://modelscope.cn/models/Robbyant/lingbot-vla-v2-6b)）、**LeRobot 后训练范例**（RoboTwin 2.0 50 任务）与 **`deploy.lingbot_vla_v2_policy` 真机服务**，降低「只读论文、难复现部署」的落差。
- **数据工程与模型同等重要：** 约 **60,000 h** 原始池经 **机器人 / egocentric 双流分阶段过滤**（video–state 对齐、运动质量、手轨迹重建等），强调 **质量对齐** 而非单纯堆小时——与 [HumanNet](./humannet.md) 讨论的「人视频小时能否替代早期真机预训练」形成同团队实证语境。
- **动作空间从双臂走向全身：** **55 维** 规范向量统一 **臂、EEF、夹爪、灵巧手、腰、头、移动底盘**，使 **20 种 embodiment** 可在同一 MoE 专家预算下共训——对照 [Green-VLA](./paper-greenvla-staged-vla-humanoid.md) 的 **64 维语义槽位** 与 [Qwen-VLA](./qwen-vla.md) 的 **embodiment prompt** 路线。
- **预测性蒸馏而非纯 BC：** **Dual-Query** 从 **LingBot-Depth**（几何）与 **DINO-Video**（语义时序）蒸馏 **当前+未来感知 query**，把「世界会怎么变」作为辅助表征目标，与 [Being-H0.7](../methods/being-h07.md) 类未来观测监督同族但走 **显式教师蒸馏** 路径。

## 核心结构/机制

| 模块 | 作用 |
|------|------|
| **VLM 骨干** | **Qwen3-VL-4B-Instruct**：理解多视角图像、语言指令与场景上下文。 |
| **MoE Action Expert** | 稀疏 **MoE** 层：细粒度专家 + **shared expert** 隔离通用/专精模式；后训练可用 **sequence-wise aux loss** + **router z-loss** 稳路由。 |
| **55 维统一动作** | 臂 **14** + EEF **14** + 夹爪 **2** + 手 **12** + 腰 **4** + 头 **2** + 移动 **3** + 保留 **4**；各 embodiment 经 **robot config YAML** 映射。 |
| **Dual-Query 蒸馏** | 视觉/文本 token 后接 **当前+未来 query**；教师：**LingBot-Depth**（深度/几何）、**DINO-Video**（语义视频表征）；另需 **MoGe-2** 等（见 `Training_Config.md`）。 |
| **Native depth** | 发布权重为 **native-depth** 变体；真机配置见 `configs/vla/real_robot/real_robot.yaml`。 |
| **后训练栈** | **LeRobot v2.1/v3.0** → `robot_configs/*.yaml` → `norm_stats`；优化器可选 **Muon** 或 **AdamW**。 |

## 流程总览（数据 → 预训练 → 后训练 → 部署）

```mermaid
flowchart TB
  subgraph pool [原始池 ~60k h]
    R[机器人轨迹\n50k h × 20 本体]
    E[Egocentric 人视频\n10k h]
  end
  subgraph filt [双流质量过滤]
    RF[机器人：对齐/清晰度/运动质量]
    EF[人视频：VLM 筛 manipulation\n手轨迹重建与标准化]
  end
  subgraph pt [预训练 LingBot-VLA 2.0-6B]
    VLM[Qwen3-VL-4B]
    MOE[MoE Action Expert\n55 维统一动作]
    DQ[Dual-Query 蒸馏\nDepth + DINO-Video]
    VLM --> MOE
    VLM --> DQ
  end
  subgraph ft [后训练范例]
    LR[LeRobot 数据集]
    CFG[robot config 映射]
    NS[norm statistics]
    LR --> CFG --> NS --> MOE
  end
  subgraph dep [评测与部署]
    OL[开环 eval]
    RT[RoboTwin 50 任务闭环]
    RR[真机 policy server\n~130 ms / 4090D]
  end
  R --> RF
  E --> EF
  RF --> pt
  EF --> pt
  MOE --> ft
  ft --> dep
```

## 论文核对（2026-11-05，已读 arXiv:2607.06403 方法与评测节）

- 论文 §4.1 称预训练采用 **token 级 loss-free MoE**：sigmoid 路由、每层 1 个共享专家 + Nr 个路由专家（每 token 激活 K 个）、DeepSeek-V3 式 **无辅助损失** 的 bias 负载均衡；§4.2 为 Dual-Query 蒸馏（对齐 LingBot-Depth 深度 token 与 DINO-Video 语义时序表征）。
- **论文与 LeRobot 默认配置的差异（请勿混读）：** LeRobot `LingbotVLAV2Config` 注释写明 released recipe 的**主**负载均衡手段是 DeepSeek-V3 的 sequence-wise **辅助损失**（`sequence_wise_loss_coeff=1e-3`）加 router z-loss（`router_z_loss_coeff=1e-4`），路由 `sigmoid`；这与论文 §4.1 描述的「auxiliary-loss-free bias」口径不同，哪一个对应哪个训练阶段本次未能核实（推测：论文描述预训练，released recipe 描述后训练）。
- 论文 Table 3：DINO-Video 在 LARYBench 四项中三项最优。
- 「6B」来自权重仓库名 `lingbot-vla-v2-6b` 与项目 README 口径；论文正文中本次未检索到明确的总参数量表述，MoE 对照实验用的是「Dense 0.6B vs MoE 1.6B-A-0.6B」的小尺度设置。
- 论文长程移动操作表（冰箱分拣 77.1/60.0 等）与上表一致。

## 源码运行时序图

LeRobot 固定提交 `9e5ee98` 的 `policy.type=lingbot_vla_v2`（`src/lerobot/policies/lingbot_vla_v2/`，核心在 `modeling_lingbot_vla_v2.py` 与 `model_core/`）：

```mermaid
sequenceDiagram
    autonumber
    participant Tr as lerobot_train.train / lerobot_eval.rollout
    participant Fa as policies.factory.make_policy / make_pre_post_processors
    participant Pre as make_lingbot_vla_v2_pre_post_processors
    participant Pol as LingbotVLAV2Policy
    participant FM as FlowMatchingV2
    participant X as QwenvlWithExpertV2Model (Qwen3-VL + MoE 动作专家)

    Tr->>Fa: --policy.path=lerobot/lingbot_vla_v2_base (或 _robotwin)
    Fa->>Pre: rename → add_batch → RelativeActionsProcessorStep → normalize → LingbotVLAV2SlotMappingProcessorStep(55 维规范槽位) → LingbotVLAV2ImageProcessorStep → LingbotVLAV2ChatTemplateProcessorStep → TokenizerProcessorStep
    Pre-->>Tr: images / image_grid_thw / lang_tokens / 55 维 state
    Tr->>Pol: policy(batch) = forward
    Pol->>Pol: _teacher_targets（仅配置 align_params 时，DistillationTeachers 取深度/视频教师目标）
    Pol->>FM: forward：make_flow_matching_inputs(NOISE_AT_ONE) → embed_prefix + embed_suffix
    FM->>X: forward([prefix_embs, suffix_embs])：VLM 前缀 + MoE 专家后缀，_block_suffix_to_queries_ 屏蔽 query
    X-->>FM: suffix_out + router_logits
    FM-->>Pol: 逐元素 MSE(L1_fm 可选) + sequence-wise 损失 + router z-loss + _align_losses
    Tr->>Pol: 推理 select_action：队列空 → predict_action_chunk
    Pol->>FM: sample_actions：前缀 KV 缓存一次 → num_steps=10 次 predict_velocity（可选 CUDA Graph / torch.compile / RTC）
    FM-->>Pol: 规范空间动作 (B,50,55)
    Pol-->>Tr: 逆槽位映射 → unnormalize → AbsoluteActionsProcessorStep（若相对动作）→ 取前 n_action_steps
```

## LeRobot 实现

- **安装与权重：** `pip install -e ".[training,lingbot_vla_v2]"`；Qwen3-VL tokenizer/图像处理器取自 `Qwen/Qwen3-VL-4B-Instruct`；不需要 `flash-attn`（改用 PyTorch SDPA）。检查点：`lerobot/lingbot_vla_v2_base`（`robbyant/lingbot-vla-v2-6b` 的格式转换）与 `lerobot/lingbot_vla_v2_robotwin`（RoboTwin 2.0，14 维，step 50000）。
- **动作专家（文档）：** 稀疏 MoE Qwen2 专家，32 个路由专家、top-4；默认 `chunk_size=n_action_steps=50`、去噪 10 步；55 维规范空间用 `--policy.state_slots` / `--policy.action_slots` 把自己的机器人映射进去。
- **RoboTwin 数字：** LeRobot 文档转述上游 README：50 任务平均 clean 93.52%、randomized 92.80%（π0.5 为 82.74% / 76.76%）；文档明说这些是上游数字，并非通过此移植测得。
- **推理加速开关（默认关）：** `use_cudagraph_denoise`、`use_cudagraph_prefix`、`use_cudagraph_prefix_full`、`compile_predict_velocity`；文档在一张 RTX PRO 6000 上给出 50 步 chunk 约 756 ms（bf16，无开关）→ 约 120 ms（全开+compile）。此处的 ~130 ms/4090D 与那条是不同硬件、不同口径的数字。
- **蒸馏教师许可：** MoGe-2 (MIT)、LingBot-Depth (Apache-2.0)、DINO-Video（来自 DINOv3，受 DINOv3 License 约束）；仅蒸馏训练才下载。
- 许可：代码和转换权重 Apache-2.0。

## 公开结果（generalist 联合训练，README / 项目页）

> 指标为 **progress score / success rate**；数字以 [官方 README](https://github.com/robbyant/lingbot-vla-v2) 为准。

### GM-100 双臂桌面

| 平台 | GR00T N1.7 | π₀.₅ | LingBot-VLA 1.0 | LingBot-VLA 2.0 |
|------|----------:|-----:|----------------:|----------------:|
| AgileX Cobot Magic | 36.3 / 17.8 | 59.1 / 32.2 | 58.2 / 30.0 | **66.2 / 34.4** |
| Galaxea R1Pro | 16.4 / 5.6 | 27.4 / 8.9 | 32.7 / 15.6 | **34.6 / 15.6** |

### 长程移动操作

| 本体 | 任务 | 设定 | LingBot-VLA 2.0 | π₀.₅ |
|------|------|------|----------------:|-----:|
| Astribot S1 | 冰箱分拣 | In-domain | **77.1 / 60.0** | 65.3 / 46.7 |
| Astribot S1 | 冰箱分拣 | Out-of-domain | **37.0 / 13.3** | 30.3 / 6.7 |
| Cobot Magic-ARX X5 | 炉灶清洁 | In-domain | **84.3 / 66.7** | 79.9 / 60.0 |
| Cobot Magic-ARX X5 | 炉灶清洁 | Out-of-domain | **67.5 / 40.0** | 62.5 / 33.3 |

## 部署要点

- **推理延迟：** `deploy.lingbot_vla_v2_policy` 在 **RTX 4090D**、**10 步去噪**、`--use_compile` 下单次约 **130 ms**——仍需与 [action chunking](../methods/action-chunking.md) / 异步缓冲组合以满足高频控制。
- **环境：** Python **3.12**、PyTorch **2.8.0**、`flash-attn==2.8.3`；训练环境脚本 `tools/create_train_env.sh`。
- **权重下载：** `python3 scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-v2-6b`。

## 常见误区或局限

- **误区：6B 权重可直接零样本上任意新机器人。** 仍需 **robot config 特征映射**、**norm statistics** 与针对目标平台的 **后训练**；预训练提供的是跨本体先验，不是免适配魔法。
- **误区：人视频与机器人数据可混用同一监督。** 管线对 **egocentric 流** 单独过滤与手轨迹标准化；与机器人流的 **video–state 对齐** 要求不同，不能当作同一 loss 的简单拼接。
- **误区：Dual-Query 等于独立世界模型 rollout。** 蒸馏目标是 **表征层几何+时序先验**，推理默认仍是 **VLA 前馈/flow 去噪**，不像 τ₀-WM 类内置仿真选优环。
- **局限：** 相对 [Qwen-VLA](./qwen-vla.md) / [Xiaomi-Robotics-0](./xiaomi-robotics-0.md)，公开叙事更偏 **Robbyant 生态真机任务**；跨机构 benchmark 复现与 MoE 路由可解释性仍待社区验证。
- **局限：** [LingBot-Map](../methods/lingbot-map.md) 同属 Robbyant 几何线，但解决 **流式 3D 重建**；与本文 **操纵 VLA** 互补，非同一模型。

## 与其他工作对比

| 对照对象 | LingBot-VLA 2.0 的差异 |
|----------|------------------------|
| **LingBot-VLA 1.0** | 数据 **6 万 h** 管线、**全身 55 维** 动作、**MoE** 与 **Dual-Query**；GM-100 / 移动操作数值全面提升 |
| **π₀.₅** | 同 **flow 去噪动作头 + Qwen 系 VLM** 族；2.0 强调 **数据过滤 + MoE 跨本体 + 深度/视频蒸馏** |
| **GR00T N1.7** | Generalist 表格上 progress/success 明显落后 2.0（Cobot Magic 行） |
| **Green-VLA** | 同为 **统一动作槽位 + 大规模预训练**；Green 走 **五阶段课程 + R2 RL**，LingBot 走 **MoE + 教师蒸馏 + LeRobot 后训练范例** |
| **HumanNet 实验设定** | HumanNet 论文用 **LingBot-VLA 架构** 做「人视频 vs 真机小时」对照；2.0 把该路线的数据规模与全身动作正式产品化 |

## 关联页面

- [Robbyant（蚂蚁灵波）](./robbyant.md) — 发布方公司页与 LingBot 全栈
- [VLA（Vision-Language-Action）](../methods/vla.md) — 方法总览与 2026 开源 VLA 谱系
- [Manipulation](../tasks/manipulation.md) — GM-100 / RoboTwin 双臂 generalist 语境
- [Loco-Manipulation](../tasks/loco-manipulation.md) — 冰箱分拣、炉灶清洁等长程移动操作
- [HumanNet](./humannet.md) — 人中心视频作 VLA 预训练来源的对照实验
- [Qwen-VLA](./qwen-vla.md) — 同 **Qwen3-VL + flow 动作头** 族的通才路线
- [Green-VLA](./paper-greenvla-staged-vla-humanoid.md) — 另一 **统一动作 + 分阶段训练** 人形向实例
- [LeRobot](../entities/lerobot.md) — 后训练数据格式与工具链
- [LeTools](./letools.md) — 乐聚 Learning 仓 2026-07-09 起接入 LingbotVLA-v2 外部模型
- [LingBot-Map](../methods/lingbot-map.md) — 同团队 **流式 3D 几何** 基础模型（问题域不同）

## 参考来源

- [lerobot.md](../../sources/repos/lerobot.md)：固定提交 9e5ee98 的 `docs/source/lingbot_vla_v2.mdx` 与 `src/lerobot/policies/lingbot_vla_v2/`
- [LingBot-VLA 2.0 技术报告摘录](../../sources/papers/lingbot_vla_v2_tech_report.md)
- [LingBot-VLA 2.0 仓库归档](../../sources/repos/lingbot-vla-v2.md)
- [LingBot-VLA 2.0 官方项目页](../../sources/sites/lingbot-vla-v2-technology-robbant.md)
- Wu et al., *From Foundation to Application: Improving VLA Models in Practice*, [arXiv:2607.06403](https://arxiv.org/abs/2607.06403)
- [robbyant/lingbot-vla-v2（GitHub）](https://github.com/robbyant/lingbot-vla-v2)

## 推荐继续阅读

- [LingBot-VLA 2.0 项目页](https://technology.robbyant.com/lingbot-vla-v2) — 数据管线可视化与真机演示
- [robbyant/lingbot-vla-v2-6b（Hugging Face）](https://huggingface.co/robbyant/lingbot-vla-v2-6b) — 预训练权重与 depth/DINO 教师子目录
- [configs/vla/Training_Config.md](https://github.com/robbyant/lingbot-vla-v2/blob/main/configs/vla/Training_Config.md) — 批大小、蒸馏、MoE 与优化器细节
