---
type: entity
tags:
  - paper
  - progress-reward
  - reward-modeling
  - vlm
  - zero-shot
  - manipulation
  - imitation-learning
  - awr
  - oxe
  - uw
  - ai2
  - amazon
  - unc-chapel-hill
  - lerobot
status: complete
updated: 2026-11-05
arxiv: "2602.19313"
code: https://github.com/TOPReward/TOPReward
related:
  - ../concepts/progress-reward-modeling.md
  - ../entities/paper-progress-reward-modeling-survey.md
  - ../methods/awr.md
  - ../methods/imitation-learning.md
  - ../methods/vla.md
  - ./paper-prm-as-a-judge.md
  - ../concepts/open-x-embodiment.md
  - ../entities/lerobot.md
  - ./paper-sarm.md
  - ./paper-robometer.md
sources:
  - ../../sources/papers/topreward_arxiv_2602_19313.md
  - ../../sources/sites/topreward-github-io.md
  - ../../sources/repos/topreward.md
  - ../../sources/repos/lerobot.md
summary: "TOPReward（arXiv:2602.19313）：用视频 VLM 对「任务已完成」token 的 log-likelihood 作零样本稠密进度奖励；OXE/ManiRewardBench 上显著优于 GVL，并支撑成功检测与 TOP-AWR；代码与基准已开源。"
---

# TOPReward：Token 概率作零样本机器人进度奖励

**TOPReward**（*Token Probabilities as Hidden Zero-Shot Rewards for Robotics*，[arXiv:2602.19313](https://arxiv.org/abs/2602.19313)，[项目页](https://topreward.github.io/webpage/)，[代码](https://github.com/TOPReward/TOPReward)）由 **华盛顿大学（UW）** / **艾伦人工智能研究所（AI2）** / **亚马逊（Amazon）** / **北卡罗来纳大学教堂山分校（UNC–Chapel Hill）** 提出：不训练专用 reward 模型，直接读预训练**视频 Vision-Language Model（VLM）**内部对肯定回答 token 的概率，估计多样真机操作任务的进度。

## 一句话定义

**把「指令 + 轨迹前缀」喂给视频 VLM，用肯定完成 token（如 `True`）的 log-likelihood 当作训练无关的稠密进度奖励。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TOPReward | Token Probabilities as Hidden Zero-Shot Rewards | 本文方法：token 似然作零样本进度 |
| VLM | Vision-Language Model | 提供视频–语言先验的骨干 |
| GVL | Generative Value Learning | 让 VLM 生成完成百分比的对照基线 |
| VOC | Value-Order Correlation | 进度曲线与真值完成序的秩相关指标 |
| AWR / TOP-AWR | Advantage-Weighted Regression | 用 TOPReward 作 advantage 权重的 BC |
| OXE | Open X-Embodiment | 大规模跨本体操作数据评测轴之一 |
| ManiRewardBench | Manipulation Reward Benchmark | 作者自建真机进度/奖励基准 |

## 为什么重要

- **过程奖励零样本落地：** 属于 [过程奖励建模](../concepts/progress-reward-modeling.md) 中「冻结基础模型打分」范式，但刻意避开 VLM 数值生成弱点。
- **跨平台证据：** OXE + ManiRewardBench（Franka·YAM·SO-100/101 等；数据集个数与任务数以论文 v2 为准，见「论文 v2 核对」）上，开源骨干（尤其 Qwen3-VL）相对 GVL 提升巨大。
- **可复现工程入口：** MIT 代码仓 + HF 上的 ManiRewardBench 子集，适合做数据过滤、成功检测与离线加权 BC。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 华盛顿大学（UW）；艾伦人工智能研究所（AI2）；亚马逊（Amazon）；北卡罗来纳大学教堂山分校（UNC–Chapel Hill） |
| **问题** | 通用、指令条件、跨域可迁移的过程奖励；避免任务特定 reward 训练 |
| **核心机制** | 轨迹前缀上查询「是否完成指令」→ 取 `True` 等 token 的 log-likelihood → 时序对齐成进度 |
| **评测** | **v1 / 项目页口径：** OXE（39 数据集）Mean VOC（Qwen3-VL-8B）0.857；ManiRewardBench ≈0.94–0.95。**v2 论文（2026-07-22）口径见下文「论文 v2 核对」**：OXE 过滤后 33 个数据集 0.874，ManiRewardBench 进度子集 0.942 |
| **下游** | 成功检测（ROC-AUC）；SO-100 上奖励加权 BC（v2 论文称 TOP-RWBC，项目页/图注仍写 TOP-AWR）相对 BC 提升成功次数 |
| **开源** | **已开源（MIT）**：<https://github.com/TOPReward/TOPReward>；基准数据见 HF `ajyanggg/manirewardbench_*` |

## 流程总览

```mermaid
flowchart TB
  vid["视频轨迹前缀\n+ 语言指令"]
  prompt["提示 VLM：是否完成指令？"]
  tok["提取肯定 token\nlog-likelihood"]
  prog["沿时间对齐\n稠密进度 / 奖励"]
  down["下游：成功检测\n· TOP-AWR · 过滤"]
  vid --> prompt --> tok --> prog --> down
```

## 核心原理

1. **Prompted video–language inference** — 不问「完成了百分之几」，而问「当前前缀是否已完成该指令」。
2. **Token-probability reward** — 用肯定回答 token 的似然，绕过数值生成与弱指令跟随。
3. **Prefix alignment** — 对越来越长的前缀重复打分，得到单调性更好的进度曲线；实验中常做 **per-episode min-max** 归一（跨轨迹绝对可比性受限）。
4. **对照 GVL** — GVL 让模型在打乱帧上输出 0–100% 完成度；TOPReward 在开源 VLM 上通常更稳、更单调。

## 源码运行时序图

官方仓库 [TOPReward/TOPReward](https://github.com/TOPReward/TOPReward)（归档见 [sources/repos/topreward.md](../../sources/repos/topreward.md)）以 Hydra 实验配置驱动推理，不训练专用 reward 权重：

```mermaid
sequenceDiagram
    autonumber
    actor Dev as 开发者
    participant Env as .env / API·HF token
    participant UV as uv sync
    participant Pred as topreward.scripts.predict
    participant DL as data_loaders<br/>HF / local_video
    participant Client as clients/*<br/>Qwen·Gemini·Molmo…
    participant Out as outputs/DATE_TIME
    Dev->>Env: 填写 OPENAI/GOOGLE/HF 等密钥
    Dev->>UV: 安装依赖与 ffmpeg
    Dev->>Pred: --config-name predict_topreward<br/>model=qwen dataset=…
    Pred->>DL: 按 configs/dataset 拉帧/视频
    DL-->>Pred: 轨迹前缀 + instruction
    Pred->>Client: 组装 text/image events
    Client-->>Pred: 生成文本 / token 似然
    Pred->>Pred: mapper + metrics（VOC 等）
    Pred->>Out: 写出 predictions / raw / metrics
```

- **最短复现路径：** `uv sync` → `.env` → `predict_topreward`（或 `run_predict.sh`）→ 查看 `outputs/`。
- **对照基线：** 同一入口换 `predict_gvl`；本地单视频可用 `dataset=local_video` + `data_loader=local`。

### LeRobot 端口（固定提交 `9e5ee98`）

LeRobot 把 TOPReward 做成 `reward_model.type=topreward` 的零样本奖励模型（`src/lerobot/rewards/topreward/`，目前只支持 Qwen3-VL 家族）。下图是 `compute_reward` 与离线标注脚本的真实调用链（路径相对 LeRobot 仓库根）：

```mermaid
sequenceDiagram
    autonumber
    actor Dev as 开发者
    participant Fac as rewards/factory.py
    participant Pre as TOPRewardEncoderProcessorStep
    participant AP as transformers AutoProcessor
    participant Model as TOPRewardModel.compute_reward
    participant VLM as Qwen3VLForConditionalGeneration
    participant CRW as topreward/compute_rabc_weights.py
    participant PQ as topreward_progress.parquet
    Dev->>Fac: make_reward_model_config("topreward", vlm_name="Qwen/Qwen3-VL-8B-Instruct")
    Fac->>Model: make_reward_model → TOPRewardModel(config) 加载 VLM (无自有权重)
    Fac->>Pre: make_reward_pre_post_processors → AddBatchDimension, Encoder, Device
    Dev->>Pre: __call__(observation.images.top 末 max_frames=16 帧, task)
    Pre->>Pre: _prepare_video_batch → uint8 (B,T,C,H,W)
    Pre->>Pre: _encode_batch: prompt_prefix + "{instruction} Decide whether ... The answer is: True"
    Pre->>AP: processor(text, videos, video_metadata, padding_side=left)
    AP-->>Pre: input_ids pixel_values_videos video_grid_thw ...
    Pre->>Pre: labels 全为 -100，仅最后一个 token (True) 保留
    Pre-->>Dev: observation.topreward.* (含 labels)
    Dev->>Model: compute_reward(batch)
    Model->>VLM: forward(logits_to_keep=2)
    VLM-->>Model: logits
    Model->>Model: reward = -cross_entropy(logits[:, -2], labels[:, -1]) 即 log P(True)
    Model-->>Dev: 每样本一个 log-prob (可选 success_threshold 二值化)
    Dev->>CRW: python -m lerobot.rewards.topreward.compute_rabc_weights --num-samples 15
    loop 每个 episode 的每个前缀长度
        CRW->>Pre: encoder(前缀帧, task)
        CRW->>Model: compute_reward → log-prob
    end
    CRW->>CRW: normalize_rewards 按 episode 做 min-max，再线性插值到全部帧
    CRW->>PQ: index / episode_index / frame_index / progress_sparse
```

- **与论文式 (1) 对应：** 奖励 = 末尾 `True` token 的 log 概率；`logits_to_keep=2` 取倒数第二个位置的 logits 去预测最后一个 token。
- **与论文式 (2) 对应：** 脚本对 `--num-samples`（文档示例 15）个均匀前缀逐个前向，再做 episode 内 min-max 并插值回逐帧曲线；注意论文强调这个归一化是**非因果**的（需要看完整个 episode），只用于评测/离线加权，不是在线奖励。
- **限制：** `forward` 不覆盖，`is_trainable=False`；`save_pretrained` 只写 `config.json`，VLM 每次从 Hub 重新加载；首次需下载约 16 GB 权重（文档数字，Qwen3-VL-8B-Instruct）；LeRobot 目前只移植 Qwen3-VL 客户端，不含 Gemini / Molmo 等官方客户端。
- **产物可喂给 SARM 的 RA-BC：** parquet 表头与 [SARM](./paper-sarm.md) 的进度文件一致，但默认文件名是 `topreward_progress.parquet`，使用 `--sample_weighting.progress_path` 指过去（推断自代码，未实跑）。

## 工程实践

| 项 | 建议 |
|----|------|
| 骨干选择 | 优先试 **Qwen3-VL-8B**（项目页开源骨干上优势最大）；Molmo-2 亦可用 |
| 提示与模板 | **勘误（对照 v2 论文附录 E.2）：** 旧版此处写「Qwen 等可能需 `add_chat_template=true`」；v2 论文的消融显示加聊天模板会让 Qwen3-VL-8B 的 VOC 从 0.945 降到 0.500（Molmo2-8B 从 0.652 降到 0.523），主实验对开源模型直接打分、不加聊天模板。LeRobot 端口的 `add_chat_template` 也默认 `False` |
| 标定 | 先做 episode 内归一；跨轨迹比较前先想清楚 VOC vs 绝对进度 |
| 成功检测 | 不要只用 VOC（失败轨迹也可能高 VOC）；v2 论文用「前缀完成/未完成对数概率差的首尾变化」加「整段视频 Yes/No 查询」两项标准化后相加（附录 C 式 7–9），不是单纯末帧平均 |
| 离线 IL | 用相邻前缀得分之差做样本权重：v2 论文式 (3) Δ=min(exp(β·(r_t−r_{t−1})), δmax)，β=0.2、δmax=2.0，始终为正，加到 π0 的 flow-matching 损失上。它与 [AWR](../methods/awr.md) 同属「按优势指数加权的回归」思路，但并不是回归形式的 AWR（项目页命名 TOP-AWR） |
| 依赖 | `uv`、ffmpeg、可选 HF token；商用模型需对应 API key |
| License | 仓 MIT；注意 NOTICE 中 OpenGVL / LeRobot 衍生文件 |

## 实验与评测

### 进度估计（Mean VOC，越高越好；项目页 / 早期口径，v2 数字见下）

| 设定 | 骨干 | GVL | TOPReward |
|------|------|-----|-----------|
| OXE（39 datasets） | Qwen3-VL-8B | 0.194 | **0.857** |
| OXE | Molmo-2-8B | -0.016 | **0.417** |
| ManiRewardBench（汇总） | Qwen3-VL-8B | ~0.2–0.5 | **≈0.94–0.95** |

ManiRewardBench 分平台（LeRobot / Franka / 双臂·单臂 YAM）上，Qwen3-VL + TOPReward 均约 0.94–0.95；曲线相对 Gemini-GVL 更平滑、更贴合真值完成。

### 成功检测与真机 TOP-AWR（早期口径，见下方 v2 核对）

- ManiRewardBench 失败子集：Qwen3-VL 上 ROC-AUC TOPReward **0.654** vs GVL **0.519**（Gemini 上两者接近）。
- 单臂 SO-100、每任务约 50 条噪声演示：6 个任务上 TOP-AWR 成功次数均 ≥ 标准 BC。

### 论文 v2 核对（arXiv 2026-07-22 版，本次通读）

上文表格是按项目页 / 论文早期口径记录的；本次通读 arXiv:2602.19313v2 全文后，下列数字和说法以 v2 为准：

| 项 | 项目页 / 早期口径（上文） | v2 论文 |
|----|------|------|
| ManiRewardBench 规模 | 113 任务 | 摘要与正文：130 个任务、四个平台（Franka、SO-100/101、单臂 YAM、双臂 YAM）；其中**进度评测子集**为 113 任务、497 条成功 episode，另有 23 任务的失败子集 |
| OXE | 39 个数据集；Qwen3-VL-8B：GVL 0.194 / TOPReward 0.857 | 过滤出 33 个 LeRobot 集合内数据集，每集 20 个 episode：Qwen3-VL-8B GVL 0.218 / TOPReward **0.874**；Qwen3-VL-32B TOPReward 0.890；Molmo2-8B GVL −0.019 / TOPReward 0.525；Gemini-2.5-Pro 上 GVL 0.572 反而高于 TOPReward 0.430（论文归因于 Gemini API 强制聊天模板） |
| ManiRewardBench VOC（进度子集） | Qwen3-VL 约 0.94–0.95 | Qwen3-VL-8B TOPReward **0.942**（GVL 0.316）；**Qwen3-VL-32B 为 0.864，反而低于 8B**；Molmo2-8B 0.619；趋势上「骨干越大越好」并不成立 |
| 与训练型奖励模型对比 | 未比较 | 与 Robometer-4B 比较：OXE 0.874 对 0.838，ManiRewardBench 0.942 对 0.858，指令混淆矩阵对角质量 0.420 对 0.149（Robometer 论文自己的 OOD 基准上结论相反，见 [Robometer](./paper-robometer.md)） |
| 成功检测 ROC-AUC | TOPReward 0.654 / GVL 0.519（早期项目页） | 在 279 条 LeRobot 清洗后 episode（150 成功 / 129 失败）上：TOPReward Qwen3-VL-8B 0.939、32B 0.965；GVL（VOC 打分）0.728；Robometer-4B 成功头 0.930（成功检测方法是前缀差 + 整段 Yes/No 的组合分，见附录 C） |
| 真机下游 | TOP-AWR | 改称 TOP-RWBC：π0 先在 200 小时 SO-100 数据上预训练，每任务 50 条演示，6 个任务的 10 次试验部分成功得分均高于 BC（如 Place doll in box：BC 7 → TOP-RWBC 10；Pick up cube：7 → 10；Put pen into cup：5.67 → 6.33） |
| 指令敏感性 | 未提 | 同一视频配错指令，平均原始奖励从 −1.68 降到 −5.04（150 条成功 LeRobot episode）；论文以此说明奖励不是仅靠时间位置 |

- **未改变的结论：** 方法本身（式 1 的 `True` token 对数概率、式 2 的 episode 内 min-max）与上文一致；论文明确说 min-max 是非因果的，原始分数才是因果的。
- **需要记住的论文自述局限：** 对聊天模板和答案 token 选择敏感（部署前要验证）；`True` 是论文选出的 token，附录 A 的「整句指令似然」变体效果更差。

## 结论

**在开源视频 VLM 上，用完成 token 似然作零样本进度，往往比让模型「报百分比」更可靠；工程价值主要在过滤/成功检测/加权 BC，而非替代在线 RL 的标定奖励。**

1. **真影响指标是 VOC + 成功检测 AUC** — 不要只看曲线好看。
2. **骨干决定上限** — Qwen3-VL 上收益最大；换弱视频理解模型会掉点；但 v2 论文显示 ManiRewardBench 上 32B（0.864）低于 8B（0.942），骨干并非越大越好。
3. **VOC ≠ 成功** — 早停失败轨迹仍可能高秩相关。
4. **归一在 episode 内** — 跨轨迹绝对进度需额外标定。
5. **下游优先 TOP-AWR / 过滤** — 比直接当在线 RL 稠密奖励更稳妥。
6. **复现从官方 `predict_topreward` 起步** — 对照跑 `predict_gvl`。

## 局限与风险

- 细粒度空间推理不足时，进度信号噪声大（VLM 感知上限）。
- Per-episode min-max 限制跨轨迹绝对比较。
- 逐步查询大 VLM 成本高，难直接作高频在线 reward。
- 性能绑定底层视频理解；模型升级会带动方法，但也会换 API/部署成本。

## 与其他工作对比

| 工作 | 关系 |
|------|------|
| [过程奖励建模](../concepts/progress-reward-modeling.md) / [综述](./paper-progress-reward-modeling-survey.md) | TOPReward = 冻结 VLM 打分范式的具体实现 |
| GVL / OpenGVL | 生成式完成百分比；本仓内建对照，开源骨干上通常弱于 TOPReward |
| [AWR](../methods/awr.md) | 项目页称 TOP-AWR；v2 论文改称 TOP-RWBC，把前缀得分增量经 exp 变成正权重加在 flow-matching 损失上，与 AWR 思路相近但损失是 π0 的 flow-matching |
| [Open X-Embodiment](../concepts/open-x-embodiment.md) | 大规模进度估计评测床之一 |
| 需训 reward 的 PRM | TOPReward **零训练**；换来的是对骨干与归一的依赖 |

## 关联页面

- [过程奖励建模](../concepts/progress-reward-modeling.md) — 接口×范式定位
- [Progress Reward Survey](./paper-progress-reward-modeling-survey.md) — 领域地图
- [AWR](../methods/awr.md) — TOP-AWR 下游
- [Imitation Learning](../methods/imitation-learning.md) — 加权 BC 语境
- [VLA](../methods/vla.md) — 指令条件操作策略侧
- [PRM-as-a-Judge](./paper-prm-as-a-judge.md) — 同类「冻结进度模型」，但输出 OPD 过程评测而非 token 似然奖励
- [Open X-Embodiment](../concepts/open-x-embodiment.md) — OXE 评测轴
- [LeRobot](./lerobot.md) — 零样本奖励模型端口所在框架
- [SARM](./paper-sarm.md) — LeRobot 内可训练的阶段感知奖励模型
- [Robometer](./paper-robometer.md) — LeRobot 内的偏好 + 进度奖励模型，与本文互为对照基线

## 参考来源

- [sources/papers/topreward_arxiv_2602_19313.md](../../sources/papers/topreward_arxiv_2602_19313.md)
- [sources/sites/topreward-github-io.md](../../sources/sites/topreward-github-io.md)
- [sources/repos/topreward.md](../../sources/repos/topreward.md)
- [sources/repos/lerobot.md](../../sources/repos/lerobot.md)
- LeRobot 文档 [`topreward.mdx`](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/topreward.mdx) 与 `src/lerobot/rewards/topreward/`（固定提交 `9e5ee98`，本次通读）
- arXiv:2602.19313v2 全文（2026-07-22，本次通读；上文早期数字来自项目页）

## 推荐继续阅读

- [项目页](https://topreward.github.io/webpage/) — 交互式进度演示与真机视频
- [arXiv:2602.19313](https://arxiv.org/abs/2602.19313) — 全文
- [GitHub: TOPReward/TOPReward](https://github.com/TOPReward/TOPReward) — 推理与评测代码
