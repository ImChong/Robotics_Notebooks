---
type: entity
tags: [framework, wam, manipulation, open-source, black-forest-labs, huggingface]
status: complete
updated: 2026-09-24
related:
  - ../concepts/world-action-models.md
  - ./lerobot.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/repos/flux-action.md
  - ../../sources/repos/lerobot.md
summary: "FLUX 3 Action：BFL 7B world action model 训练/推理/tooling 已开源；HF 公开 DROID/SO-101/base 权重；SO-101 LoRA 走 LeRobot。"
---

# FLUX 3 Action

**FLUX 3 Action** 是 [Black Forest Labs](https://github.com/black-forest-labs) 发布的 **world action model** 工程栈：在 FLUX 3 视觉生成骨干上做 **standalone 全量微调** 与 **动作预测**，覆盖数据准备、分布式训练、checkpoint/resume、export 与 **DROID / SO-101** 推理部署。SO-101 任务 LoRA 通过 **[LeRobot](./lerobot.md)** 集成路径发布。

## 一句话定义

BFL 把 video world model 迁到 robot action：独立训练/推理代码 + Hub 上 DROID/SO-101/base 权重族，SO-101 下游可接 LeRobot LoRA 工作流。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 联合未来视觉与动作预测 |
| BF16 | Brain Float 16 | 默认全精度推理/训练 dtype |
| FP8r | FP8 reduced | 量化推理变体（Hub `variants/fp8r`） |
| GD / SD | Guidance / Step Distilled | 4-step 与 1-step 蒸馏配方 |
| DROID | Distributed Robot Interaction Dataset | 桌面臂大规模数据集 |

## 为什么重要

- **工业级 WAM tooling：** 相对论文-only WAM，提供 **可克隆仓库 + 公开 checkpoint 矩阵**（base / DROID / SO-101 × BF16/FP8r × GD/SD）。
- **训练/推理代码分离：** `transformer.py` vs `transformer_inf_bf16.py` / `transformer_inf_fp8r.py` — 便于 agent/工程师直读路径。
- **LeRobot 桥接：** SO-101 **task LoRA** 不在孤立脚本里，而挂在 LeRobot 生态 — 降低 sim2real 栈拼接成本。

## 核心信息

| 字段 | 内容 |
|------|------|
| GitHub | [black-forest-labs/flux-action](https://github.com/black-forest-labs/flux-action)（**已开源**） |
| HF Collection | [black-forest-labs/flux-3-action](https://huggingface.co/collections/black-forest-labs/flux-3-action)（权重 **公开**，gated: false） |
| 规模 | ~7B 级 world action model（项目命名 FLUX 3 Action） |
| 平台 | Linux + NVIDIA GPU；Python 3.12 / torch 2.10 cu128 示例 |

## 流程总览

```mermaid
flowchart LR
  data["DROID parquet / SO-101 数据"]
  prep["数据准备\nprepare"]
  train["分布式全量微调\ntransformer.py"]
  export["checkpoint / export"]
  infer["推理\ntransformer_inf_*"]
  hub["HF flux-3-action-*"]
  lerobot["LeRobot SO-101 LoRA\n可选"]
  robot["DROID / SO-101 部署"]
  data --> prep --> train --> export --> hub
  hub --> infer --> robot
  hub --> lerobot --> robot
```

## 源码运行时序图

```mermaid
sequenceDiagram
  autonumber
  participant Dev as 开发者
  participant Setup as uv sync + NATTEN wheel
  participant Prep as docs/prepare.md
  participant Train as 训练入口
  participant Hub as Hugging Face Hub
  participant Inf as docs/setup.md 推理
  Dev->>Setup: clone flux-action + extras
  Dev->>Prep: DROID episode / 自定义 embodiment
  Dev->>Train: droid-finetune 配方
  Train->>Hub: 上传或拉取 checkpoint
  Dev->>Inf: BF16 或 FP8r 变体 predict
  Inf-->>Dev: action chunk / serve RoboLab
```

节点对齐 [`sources/repos/flux-action.md`](../../sources/repos/flux-action.md) 与 `docs/setup.md`、`docs/droid-finetune.md`、`docs/so101-lora.md`。

### LeRobot 移植的运行时序（固定提交 `9e5ee98`）

上图是 standalone `flux-action` 仓库的入口视角；下图来自 LeRobot `src/lerobot/policies/flux3/modeling_flux3.py`（`Flux3Policy`），打包与采样函数在 `f3/packing.py`、`f3/sampling.py`。

```mermaid
sequenceDiagram
    participant TR as "lerobot_train.py: train()"
    participant EV as "lerobot_eval.py / lerobot_rollout"
    participant P as "Flux3Policy"
    participant PK as "f3/packing.py"
    participant D as "f3/transformer.py: JointSingleSeq (DiT)"
    participant S as "f3/sampling.py"
    TR->>P: "policy(batch) -> forward"
    P->>P: "_cameras / _conditioning_inputs / _captions / _valid_windows"
    P->>PK: "materialize_video -> pack_video(冻结 Video VAE) / pack_actions / pack_text"
    P->>PK: "sample_timesteps -> build_forward_kwargs"
    P->>D: "self.dit(**kwargs) 同时预测视频 latent 与动作"
    P->>PK: "flow_loss(video_loss_weight, action_loss_weight)"
    PK-->>TR: "loss, {video_mse, action_mse}"
    EV->>P: "select_action(batch)"
    P->>P: "队列空 -> predict_action_chunk"
    P->>PK: "pack_video(targets=False) 条件 latent + pack_actions"
    P->>P: "_sample: 视频与动作噪声联合去噪"
    P->>S: "cosmos_unipc_order2 或 euler (num_inference_steps)"
    S->>D: "predict(): 无分类器引导时 cfg_two_pass"
    S-->>P: "动作 latent / action_scale"
    P-->>EV: "取前 n_action_steps 入队；视频分支结果丢弃"
```

## LeRobot 实现

- **身份核对**：未找到一手论文。`src/lerobot/policies/flux3/README.md` 写有 “Technical Report ... XXX”，是未填的占位，没有 arXiv 链接；LeRobot 文档与该 README 指向 Black Forest Labs 的模型卡与 `flux-action` 仓库。本页因此无 arXiv ID，数据以文档/源码为准。
- **规模与能力**（LeRobot 策略 README，厂商表述，未核实）：7B world action model，输入相机帧、状态、文本，返回未来 32 个动作，与未来 32 帧一同去噪；DROID 微调版在 RoboLab-120 上 42.6% 任务成功率（排名第一，厂商表述）。
- **LeRobot 支持范围**：推理与 PEFT（LoRA 适配主干，动作/条件头全量训练）；全量微调与新本体训练回到 standalone 仓库。SO-101 微调检查点 `black-forest-labs/flux-3-action-so101`，DROID `flux-3-action-droid`，基座 `flux-3-action-base`（文档）。
- **SO-101 检查点配置**（文档）：`scene` / `wrist` 双相机，8 个观测时间步，2 个独立编码的视觉快照，过去动作/状态条件；预测 42 个动作，执行前 32 个（30 Hz）；末通道保持绝对（`delta_absolute_dims=[-1]`）。DROID：3 相机，8 维状态/动作，15 Hz 绝对关节位置指令。
- **只构建 video 与 video_cond 流**：未使用的图像/音频权重加载前被过滤，文档称省约 28% DiT 参数而不改变动作预测。
- **限制**：不支持 RTC / 异步推理（`select_action` 在 `use_relative_actions` 时直接抛 `NotImplementedError`）；依赖 NATTEN 0.21.6+ 且与 torch/CUDA 精确匹配的 wheel；DROID 的 PEFT 端到端训练“尚未验证”；文档声明本版本仍需 GPU 与机器人验证。
- **后处理**：策略输出为归一化动作，必须经 postprocessor 反归一化并按 delta 积分，首个指令以实测状态为锚；episode 边界要重置策略与两个处理器。
- 仅动作被返回；`_sample` 同时去噪视频但丢弃（`modeling_flux3.py`）。

## 工程实践

| 检查项 | 建议 |
|--------|------|
| Checkpoint 选型 | Base 4-step / GD / SD 与 BF16/FP8r 六组合 — 见 README 表与 [setup#choose-a-checkpoint](https://github.com/black-forest-labs/flux-action/blob/main/docs/setup.md) |
| Config 文件 | LeRobot 读 `config.json`；原生 FP8r 优先 `config.native.json` |
| NATTEN | 必须匹配 torch/CUDA 的 whl — 视频 VAE neighborhood attention 依赖 |
| SO-101 | 任务适配走 **LeRobot LoRA** 文档，勿与 standalone DROID 全量微调混环境 |
| 自定义 embodiment | 见 `docs/embodiments.md`（游戏示例作 worked example） |

## 与其他工作对比

| 对照 | 差异 |
|------|------|
| Cosmos / DreamZero 等 WAM | BFL 栈强调 **独立 repo + 多 variant checkpoint 矩阵** + LeRobot 下游 |
| [LeRobot](./lerobot.md) | LeRobot 是通用 IL/VLA 框架；FLUX Action 是 **专用 WAM 训练/推理** + 可选 LeRobot LoRA |
| [THAW-VLA](./paper-thaw-vla.md) | THAW 蒸馏 WAM 特征进小 VLA；FLUX Action 是 **全量 WAM 训练部署** |

## 结论

**FLUX 3 Action 是目前少数「仓库 + 公开权重 + 训练/推理/LoRA 文档齐全」的 WAM 工程发布，适合作为 DROID/SO-101 world-action 基线与自定义 embodiment 起点。**

1. **先选 checkpoint 配方** — GD/SD/FP8r 影响 latency 与质量，不是单一权重。
2. **SO-101 走 LeRobot** — 与 standalone DROID 全量微调环境分离。
3. **NATTEN + FFmpeg AV1** — 环境门禁，CI/云镜像需预装。
4. **Hub 权重已公开** — 无需 gated 申请（2026-09-24 API 核查）。
5. 与 [world-action-models](../concepts/world-action-models.md) 概念页对照读「生成式 world→action」位置。

## 关联页面

- [World Action Models](../concepts/world-action-models.md)
- [LeRobot](./lerobot.md)
- [VLA](../methods/vla.md)
- [Manipulation](../tasks/manipulation.md)

## 推荐继续阅读

- [flux-action GitHub](https://github.com/black-forest-labs/flux-action)
- [HF FLUX 3 Action collection](https://huggingface.co/collections/black-forest-labs/flux-3-action)

## 参考来源

- [`sources/repos/lerobot.md`](../../sources/repos/lerobot.md) — LeRobot 固定提交上的 `flux3` 策略与文档
- [FLUX 3 Action 仓库归档](../../sources/repos/flux-action.md)
