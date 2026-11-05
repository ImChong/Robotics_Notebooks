---
type: entity
tags:
- paper
- robbyant
- world-action-model
- vla
status: complete
updated: '2026-10-17'
arxiv: '2601.21998'
venue: arXiv 2026
code: https://github.com/Robbyant/lingbot-va
summary: LingBot-VA 在因果交错序列中建模视觉动态与动作，以真实观测更新缓存，并通过异步执行减少视频–动作联合推理的控制停顿。
related:
- ../entities/awesome-world-models.md
- ../overview/sun-awesome-wm-technology-map.md
- ../methods/generative-world-models.md
- ../methods/model-based-rl.md
- ../tasks/manipulation.md
- ../tasks/locomotion.md
- ./robbyant.md
- ./lingbot-vla.md
sources:
- ../../sources/papers/sun_awesome_wm_2601_21998_lingbot-va-causal-video-action-world-mod.md
- ../../sources/papers/sun_awesome_wm_catalog.md
- ../../sources/repos/awesome-world-models.md
- ../../sources/repos/lingbot-va.md
- ../../sources/repos/lerobot.md
---

# LingBot-VA：因果视频–动作世界模型

## 一句话定义

LingBot-VA 在因果交错序列中建模视觉动态与动作，以真实观测更新缓存，并通过异步执行减少视频–动作联合推理的控制停顿。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
| --- | --- | --- |
| VLA | Vision-Language-Action | 视觉和语言条件下生成动作 |
| MoT | Mixture of Transformers | 视频/动作双流建模 |
| KV | Key-Value | 注意力历史缓存 |
| IDM | Inverse Dynamics Model | 把动作潜表示解码为执行动作 |

## 为什么重要

- 把世界预测与动作学习放入同一时间结构，便于研究视觉动态先验如何影响控制。
- 强调执行后真实观测回写，避免长程预测把错误持续带入闭环。

## 流程总览

```mermaid
flowchart TD
    OBS["首帧观测 + 任务文本"] --> ENC["Wan2.2 VAE 编码为视频 latent；UMT5 编码文本"]
    ENC --> VID["视频流去噪（CFG，默认 20 步）：预测未来 latent"]
    VID --> ACT["动作流去噪（默认 50 步）：同一 transformer 的动作分支"]
    ACT --> EXE["执行动作块，采集真实关键帧"]
    EXE --> FB["真实关键帧 + 已执行动作写回 KV cache"]
    FB --> VID
    subgraph TR["训练"]
        TS["视频 latent 与动作各自加噪"] --> BC["block-causal 双流 transformer"]
        BC --> LOSS["latent_loss + action_loss（流匹配）"]
    end
```

图依据：LeRobot `lingbot_va.mdx` 与 `modeling_lingbot_va.py` 的 `_infer` / `_compute_kv_cache` / `training_loss_from_streams`。

## 核心原理

**因果交错建模 → 动作解码 → 执行 → 真实观测替换视频缓存 → 下一轮推理**。视频流与动作流保持概念区分，MoT、KV cache 和异步执行共同降低重复计算。

公开仓目前提供 **shared backbone** 的基座与 RoboTwin / LIBERO-Long 后训练权重；分离双流版本不能仅凭论文描述认定已经发布。VA 2.0 的 semantic visual-action tokenizer、因果 DiT 与 MoE 另见[公司家族页](./robbyant.md)。

## 源码运行时序图

```mermaid
sequenceDiagram
    autonumber
    participant D as LeRobot dataset
    participant T as script/run_va_posttrain.sh
    participant M as wan_va 模型与配置
    participant C as 仿真或机器人客户端
    D->>T: 轨迹与归一化统计
    T->>M: 后训练 batch
    loop 闭环推理
        C->>M: 当前视觉与本体状态
        M->>M: 因果视频/动作生成与缓存更新
        M-->>C: 解码动作块
        C->>C: 执行动作并采集真实帧
        C->>M: 新真实观测校正历史
    end
```

模型进程与客户端是分离运行环境；部署细节以 README 对应 benchmark 的服务脚本为准。

### LeRobot 移植的运行时序（固定提交 `9e5ee98`）

来源：`src/lerobot/policies/lingbot_va/modeling_lingbot_va.py`，工具函数在同目录 `utils.py`。上图是官方原仓库的入口；下图是 LeRobot 中真实的类与函数。

```mermaid
sequenceDiagram
    participant E as "lerobot_eval.py: rollout()"
    participant P as "LingBotVAPolicy.select_action"
    participant C as "LingBotVAPolicy._predict_chunk"
    participant I as "LingBotVAPolicy._infer"
    participant K as "LingBotVAPolicy._compute_kv_cache"
    participant X as "utils.py: WanTransformer3DModel.forward"
    E->>P: "policy.select_action(observation)"
    P->>P: "_maybe_init_prompt: UMT5 编码任务文本"
    P->>C: "首帧: _predict_and_decode"
    C->>C: "_encode_frames: Wan VAE 编码首帧， _init_streaming_cache"
    C->>I: "init_latent, frame_st_id=0"
    I->>X: "视频流去噪循环 (action_mode=False, update_cache)"
    I->>X: "动作流去噪循环 (action_mode=True)"
    I-->>C: "actions, latents"
    C-->>P: "裁剪 used_action_channel_ids -> 入队"
    P-->>E: "popleft 单步动作"
    Note over P: "后续步：缓冲真实关键帧；队列耗尽后"
    P->>K: "_compute_kv_cache(obs_buffer, executed_actions)"
    K->>X: "update_cache=2 写入真实 latent 与动作"
    P->>I: "下一块预测 (frame_st_id 递增)"
```

训练路径：`lerobot_train.py` 调 `LingBotVAPolicy.forward` -> `_build_training_streams`（VAE + UMT5）-> `training_loss_from_streams`（要求 `attn_mode='flex'`）-> `WanTransformer3DModel.forward(train_mode=True)` -> `_flow_matching_loss`。

## LeRobot 实现

- **身份核对**：arXiv 2601.21998 的 arXiv 标题是 *Causal World Modeling for Robot Control*（2026-01-29，API 核对），LingBot-VA 是其模型名；本页标题为库内沿用的索引名。LeRobot 文档把它称为基于 Wan2.2 的自回归视频–动作世界模型策略。
- **规模与组成**（文档）：可训练 Wan 双流 transformer 约 5B 参数（`WanTransformer3DModel`），冻结 `AutoencoderKLWan`（`z_dim=48`）与 `UMT5EncoderModel`；冻结部分约 20 GB 在加载时按 `wan_pretrained_path` 拉取，默认指向上游 `robbyant/*`。
- **权重**：`lerobot/lingbot_va_libero_long`、`lerobot/lingbot_va_robotwin`、`lerobot/lingbot_va_base`。
- **推理**：视频流 20 步、动作流 50 步，各自独立的流匹配调度器；仅实现单环境评测（`--eval.batch_size=1`）；约需 18–24 GB 显存（文档）。
- **动作空间**：30 维多本体布局，LIBERO 用通道 0–6；RoboTwin 为末端位姿控制（`--env.action_mode=ee`），由 CuRobo IK 转关节；相机顺序固定且敏感。
- **训练限制**：需 `--policy.attn_mode=flex`；完整 5B 在 24–32 GB 单卡上不能用 AdamW 全量微调，文档推荐 LoRA，且无内置默认 LoRA 目标，需手动指定 `--peft.target_modules`。
- **可选**：`save_predicted_video` / `return_predicted_video` 解码想象视频。
- 许可：Apache-2.0（文档）。

## 工程实践

1. 先锁定 `wan_va/configs/va_libero_cfg.py` 中 `action_snr_shift`、动作 channel 和 `norm_stat`，它们必须与 checkpoint 对齐。
2. 自有数据按 LeRobot 处理；`script/run_va_posttrain.sh` 选择 `robotwin_train` 或 `libero_train`。
3. 仿真与策略环境分离，通过 server-client 通信，避免依赖冲突。
4. **2026-10-05 核查**：Apache-2.0 代码、shared-backbone 权重、RoboTwin 与 LIBERO 后训练数据公开；不代表完整预训练池与所有论文结构开放。

## 评测与指标

公开提供 RoboTwin-2.0、LIBERO-Long 的评测与后训练路径。对比时明确 shared backbone、任务划分与数据增广，分别测长程成功率、数据效率和异步闭环时延；本页不混用 VA 2.0 数字。

## 结论

**LingBot-VA 的核心是因果视频–动作建模与真实观测校正构成闭环。**

1. 配置必须与发布权重的动作归一化一致。
2. 分别记录预测误差、控制成功率与时延。
3. 按已发布 shared-backbone 范围复现，不推定全部结构已开放。

## 与其他工作对比

与 [LingBot-VLA](lingbot-vla.md) 的直接动作生成相比，本作强调因果视觉动态与动作的联合时间结构。与 [FastWAM](paper-fast-wam.md) 的测试时跳过未来视频不同，本作需要分析视频–动作交错推理、异步执行和真实观测校正各自的代价。

## 局限与风险

- 预测视频存在累积误差，真实观测回写是闭环的一部分。
- 视频去噪、通信和动作执行的吞吐须一起测；不能以单模型帧率代替端到端时延。

## 关联页面

- [Robbyant 家族与 VA 2.0](./robbyant.md)
- [LingBot-VLA](./lingbot-vla.md)
- [世界模型](../methods/generative-world-models.md)
- [R²-WAM](./paper-r2-wam.md) — 让 WAM 自身重生成旧任务 rollout 作回放以缓解持续学习遗忘；作者指出遗忘集中在逆动力学部分

## 参考来源

- [`sources/repos/lerobot.md`](../../sources/repos/lerobot.md) — LeRobot 固定提交上的 `lingbot_va` 移植与文档
- 论文：<https://arxiv.org/abs/2601.21998>
- [官方项目与运行入口](../../sources/repos/lingbot-va.md)

## 推荐继续阅读

- [官方项目](https://technology.robbyant.com/lingbot-va/)
- [官方源码](https://github.com/robbyant/lingbot-va)
