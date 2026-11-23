---
type: entity
tags: ['paper', 'humanoid', 'tsinghua', 'galbot', 'bit', 'hit', 'pku', 'shanghai-pil', 'social-hri', 'co-speech-gesture', 'diffusion', 'unitree-g1']
status: complete
updated: 2026-11-21
arxiv: "2608.28693"
summary: "RoboGesture（arXiv:2608.28693，ECCV 2026，清华/银河通用等）：300+ 类手势 + 1000 h 半合成机器人音–动数据；Mimi token 分层语义–声学对齐 + DiT-CFM + Anti-Inertia CFG Masking；OSQP 自碰撞滤波将碰撞帧率由 4.16% 降至 0.13%；已开源推理、G1 部署、54 类权重与训练核心示例。"
related:
  - ../tasks/loco-manipulation.md
  - ./paper-pamor.md
  - ./paper-diffsheg.md
  - ./paper-notebook-semantic-co-speech-gesture-synthesis-and-real-ti.md
  - ./unitree-g1.md
  - ../methods/diffusion-motion-generation.md
code: https://github.com/GalaxyGeneralRobotics/RoboGesture
original:
  - ../../sources/papers/robogesture_arxiv_2608_28693.md
  - ../../sources/sites/robogesture-arxiv.md
  - ../../sources/repos/robogesture.md
sources:
  - ../../sources/papers/robogesture_arxiv_2608_28693.md
  - ../../sources/sites/robogesture-arxiv.md
  - ../../sources/repos/robogesture.md
---

# RoboGesture：人形实时语义对齐伴随语音手势

**RoboGesture: Real-Time Semantic-aligned Co-Speech Gestures Generation for Humanoid Interaction**（[arXiv:2608.28693](https://arxiv.org/abs/2608.28693)）由 **Zifan Wang、Ziang Ren、Pengyang Shi、Zirui Wang、Chenghuai Lin、Tianze Wang、Zekun Qi、Liangliang Zhao、He Wang、Li Yi**（清华 / Galbot / 北理工 / 哈工大 / 北大 / 上海期智研究院）提出；早期映射见 [公众号周更](../../sources/blogs/wechat_shenlan_weekly_papers_2026-09-04.md)，**2026-09-28** 按 arXiv 元数据与项目页复核更新。

## 一句话定义

让人形从 **听** 到 **做手势** 走 **原始音频 token → 机器人运动** 的端到端流式管线，并用 MPC 保证真机无碰撞。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DiT | Diffusion Transformer | 连续运动空间的扩散 Transformer |
| CFM | Conditional Flow Matching | 条件流匹配生成运动块 |
| CFG | Classifier-Free Guidance | 本文 Anti-Inertia 变体防历史惯性 |
| MPC | Model Predictive Control | 在线凸 QP 安全滤波（碰撞/速度/跟踪） |
| HRI | Human-Robot Interaction | 社交人机交互 |

## 为什么重要

文本中间表示丢韵律；**modality eclipse** 让模型抄运动惯性、忽视音频；avatar→retarget 在线不安全——RoboGesture 在 **机器人空间** 联合设计数据、模型与控制，并作为 **listen–respond–gesture** 闭环的运动核心。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 清华大学、银河通用机器人（Galbot）、北京理工大学、哈尔滨工业大学、北京大学、上海期智研究院 |
| **作者** | Zifan Wang, Ziang Ren, Pengyang Shi, Zirui Wang, Chenghuai Lin, Tianze Wang, Zekun Qi, Liangliang Zhao, He Wang, Li Yi |
| **开源** | 见 [工程实践](#工程实践) |

## 核心原理

三模块：**Hierarchical Semantic-Acoustic Aligner**（Mimi codec 多粒度 token + beat/300 类语义头）、**Streaming DiT+CFM**（Cross-Attn 微对齐 + FiLM 宏调制；15% 历史 mask 的 Anti-Inertia CFG）、**MPC 安全滤波**（5.6 ms/帧，离线亦用于清洗训练集）。平台：G1 + BrainCo 灵巧手，41 DoF 上身。

### 流程总览

```mermaid
flowchart LR
  audio[流式语音] --> align[语义-声学对齐器]
  align --> dit[DiT-CFM 运动生成]
  hist[历史运动] --> dit
  dit --> mpc[MPC 安全滤波]
  mpc --> g1[Unitree G1]
```

## 源码运行时序图

图依据：`GalaxyGeneralRobotics/RoboGesture@2913f1d` 的 `robogesture/cli.py`、`robogesture/offline.py`、`robogesture/model.py`、`robogesture/motion.py`、`robogesture/collision.py`、`robogesture/app.py`、`robogesture/session.py` 与 `robogesture/robot.py`（说明文档核对，未连接实机或运行 GPU checkpoint）。公开仓库分两条可执行路径：**离线 WAV 生成与渲染**，以及 **真机整句英文对话播放**（另有合成张量演示 `python -m training_examples.demo`）。

```mermaid
sequenceDiagram
    autonumber
    participant CLI as "robogesture CLI (cli.py)"
    participant App as "app.py / offline.py"
    participant ASRLLM as "ASR + Qwen3 LoRA + TTS"
    participant Gen as "MotionGenerator (model.py)"
    participant Col as "CollisionFilter OSQP (collision.py)"
    participant Sess as "MotionSession + Robot DDS"

    CLI->>App: "infer wav -> .npz / stream --execute --confirm SEND_MOTION"
    opt "stream 真机对话链路 (app.run)"
        Sess->>Sess: "Robot.start() FSM 501 + RobotAudioClient.probe()"
        App->>ASRLLM: "listen_once -> transcribe -> LocalAssistant.reply -> synthesize"
        ASRLLM-->>App: "24 kHz PCM 语音"
    end
    App->>Gen: "generate(pcm): 2x1s 静音历史预热 + Mimi 2-codebook + 50 步 CFM"
    Gen-->>App: "raw_motion [T, 41] (30 FPS)"
    App->>App: "model_to_robot + smooth_model_blocks / build_timeline"
    App->>Col: "filter(frames, legs, state.frame)"
    Col-->>App: "safe/corrected/held frames (53-DoF qpos)"
    alt "infer 离线导出"
        App-->>CLI: "写入 60 列 qpos NPZ + 对齐 WAV（可选 render 输出 MP4）"
    else "stream 实机播放 (MotionSession.play)"
        Sess->>Sess: "audio.stage(utterance_id, audio)"
        loop "逐帧 30 Hz 节拍"
            Sess->>Sess: "robot.send(frame) 与到点触发 audio.play"
        end
    end
```

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | **推理与部署代码已开源（2026-11-21 复核 `@2913f1d`）**；含 Git LFS 54 类 `MotionModel`（`checkpoints/motion/`）、Qwen LoRA（`checkpoints/qwen_lora/`）与 15 组配对示例，**1000 h 数据集与完整预处理/大规模训练脚本未公开** |
| 离线复现入口 | `robogesture infer examples/sample_06.wav --output outputs/generated.npz` 与 `robogesture render outputs/generated.npz --output outputs/generated.mp4`；`python -m training_examples.demo` 运行合成张量 Stage 0/1 训练核心验证 |
| 实机对话与控制链路 | `robogesture stream --execute --confirm SEND_MOTION` 级联 Volcengine ASR → `Qwen/Qwen3-4B-Instruct-2507` + LoRA（带动作标签）→ Volcengine TTS → `MotionGenerator`（2 s 静音预热后每秒出 30 帧）→ `CollisionFilter`（OSQP）→ G1 DDS FSM 501 与 BrainCo Revo2 灵巧手 30 Hz 同步播放 |
| 安全与损失参数 | Stage 1 采用 15% 概率整批置零 Past Motion（Anti-Inertia CFG Masking）、手部联合空间权重 `w_hand = 4.0`、语义关键帧权重 `{5, 10}`；OSQP 自碰撞滤波在论文评测中将自碰撞帧率从 `4.16%` 降至 `0.13%` |

## 实验与评测

| 基准 | FGD↓ | Col.↓ | 读法 |
|------|------|-------|------|
| BEAT | **0.845** | 0.88% | 优于 LivelySpeaker/DiffSHEG 等 |
| SemanticBEAT | BC **0.295** | **0.13%** | 语义手势对齐 SOTA |

## 结论

RoboGesture 把 **语义–韵律–安全** 绑成可部署的 listen–respond–gesture 闭环；Anti-Inertia Masking 是打破「运动抄历史」的关键训练技巧。

1. **300+** 手势类 + **1000 h** 半合成训练对（LLM 场景 → 标注 → TTS/节拍 → 0.4 s 语义预期 → MPC 清洗）。
2. 真机四场景：社交协助 / 情感支持 / 知识分享 / 办公沟通（G1 演示 clip）。
3. 与 Speech-LLM 管线级联；**≈120 FPS** motion，系统 latency 瓶颈在上游语音而非运动生成。
4. **ECCV 2026** Poster（2026-09-10 Session 2）。
5. 截至 **2026-09-28** **未开源** — 选型应假设无法复现训练，仅可对照方法与 BEAT 数字。

## 与其他工作对比

「让人形做出得体的上身动作」有几条路，差别在 **条件信号** 与 **谁保证真机安全**：

| 工作 | 条件输入 | 生成空间 | 真机安全由谁保证 | 与本文 |
|------|----------|----------|------------------|--------|
| **RoboGesture** | **原始音频 token**（Mimi codec，多粒度 + beat/300 类语义头） | **直接在机器人空间** | **MPC 安全滤波**（5.6 ms/帧），在线可用 | 本页；G1 + BrainCo 手，41 DoF 上身，≈120 FPS |
| [PAMoR](./paper-pamor.md) | **文本 + 效价–唤醒（V-A）** | 机器人全身运动（G1） | 未报安全滤波层 | 同为社交人形运动生成、同平台家族；条件是 **情感标签** 而非声学信号 |
| [DiffSHEG](./paper-diffsheg.md) / LivelySpeaker 一类 avatar 方法 | 语音/文本 | **虚拟人空间**，再 retarget | retarget 后无在线保证 | 动机：**avatar→retarget 在线不安全**；BEAT FGD 0.845 优于 DiffSHEG 等 |
| [Semantic Co-Speech → G1](./paper-notebook-semantic-co-speech-gesture-synthesis-and-real-ti.md) | 语义检索 + 生成 | 人体→GMR→RL 跟踪 | 跟踪策略 | 同为 G1 共语手势；本文 **text-free 流式音频 token**，彼为 PNB 深读索引 |
| 文本中间表示管线 | ASR→文本→动作 | 各式 | — | **丢韵律**；本文用音频 token 绕开 |
| [人形语音交互流水线](../queries/humanoid-voice-interaction-pipeline.md) | 完整 listen–respond 链路 | — | — | RoboGesture 是该链路的 **动作输出端** |

**技巧层面的可迁移点：** **Anti-Inertia CFG Masking**（15% 历史 mask）针对「历史运动 + 外部条件」流式生成的通病——**模型抄历史、忽视条件信号**。

**证据边界：** BEAT / SemanticBEAT 数字来自论文；**代码与数据截至 2026-09-28 未开源**，真机为四场景演示而非统计评测。

## 局限与风险

上身为主；未覆盖全身行走协同；半合成数据依赖 LLM 场景与 TTS；MPC 假设与 G1+BrainCo  kinematics 绑定，换 embodiment 需重跑数据管线。

## 关联页面

- [loco-manipulation](../tasks/loco-manipulation.md)
- [paper-pamor.md](./paper-pamor.md)
- [paper-diffsheg.md](./paper-diffsheg.md)
- [Semantic Co-Speech → G1（PNB）](./paper-notebook-semantic-co-speech-gesture-synthesis-and-real-ti.md)
- [unitree-g1.md](./unitree-g1.md)
- [diffusion-motion-generation](../methods/diffusion-motion-generation.md)
- [人形语音交互流水线](../queries/humanoid-voice-interaction-pipeline.md)

## 参考来源

- [robogesture_arxiv_2608_28693.md](../../sources/papers/robogesture_arxiv_2608_28693.md)
- [robogesture-arxiv.md](../../sources/sites/robogesture-arxiv.md)
- [robogesture.md](../../sources/repos/robogesture.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_papers_2026-09-04.md)

## 推荐继续阅读

- [RoboGesture 项目页](https://RoboGesture.github.io)
- [arXiv:2608.28693 PDF](https://arxiv.org/pdf/2608.28693)
