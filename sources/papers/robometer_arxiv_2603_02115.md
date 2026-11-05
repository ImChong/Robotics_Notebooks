# Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons

> 来源归档（ingest）

- **标题：** Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons
- **类型：** paper
- **来源：** arXiv abs / PDF（读的是 v2，2026-05-13）；官方 GitHub README；LeRobot 固定提交文档与源码
- **原始链接：**
  - <https://arxiv.org/abs/2603.02115>
  - <https://arxiv.org/pdf/2603.02115>
  - 项目页：<https://robometer.github.io/>
  - 官方代码：<https://github.com/robometer/robometer>（README 标注 MIT）
  - 检查点：<https://huggingface.co/robometer/Robometer-4B>（官方）；<https://huggingface.co/lerobot/Robometer-4B>（LeRobot 文档使用）
  - LeRobot 文档：<https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/robometer.mdx>
- **作者：** Anthony Liang, Yigit Korkmaz, Jiahui Zhang, Minyoung Hwang, Abrar Anwar, Sidhant Kaushik, Aditya Shah, Alex S. Huang, Luke Zettlemoyer, Dieter Fox, Yu Xiang, Anqi Li, Andreea Bobu, Abhishek Gupta, Stephen Tu, Erdem Biyik, Jesse Zhang（前两位共同一作；最后三位共同指导）
- **机构（PDF 页首）：** USC；UT Dallas；MIT；独立研究者；UW；Ai2；NVIDIA
- **首次提交：** 2026-03-02（arXiv API `published`）；LeRobot 文档标注为 RSS 2026
- **入库日期：** 2026-11-05
- **一句话说明：** 在 Qwen3-VL-4B 上同时学「逐帧进度 + 逐帧成功 + 轨迹对偏好」，并用 100 万轨迹（含大量失败/次优数据）的 RBM-1M 训练，得到跨本体的通用视觉奖励模型。

## 开源核查（2026-11-05）

| 项 | 状态 |
|----|------|
| 官方代码 | **已开源**：README 给出 `train.py`（Hydra + accelerate/FSDP 训练）、`robometer/evals/eval_server.py`（HTTP 推理服务）、`scripts/example_inference_local.py`（本地推理）、`robometer/evals/run_baseline_eval.py`（奖励对齐/策略排序评测） |
| 数据 | RBM-1M 与处理脚本在 README 中有说明；本次未下载核对数据规模 |
| 权重 | Robometer-4B（官方 HF）；另有 LIBERO 微调检查点（README） |
| LeRobot 实现 | **推理限定**：`src/lerobot/rewards/robometer/`，文档明确不含训练 / 偏好对训练 |
| 结论 | 论文代码已开源；LeRobot 内为零样本推理端口 |

## 核心论文摘录（MVP）

### 1) 问题：只靠专家示范的绝对进度标签，难以用上失败/次优数据

- **位置：** 摘要、§I
- **摘录要点：** 现有通用奖励模型依赖专家/带奖励标注示范的逐帧绝对进度，失败轨迹上进度含糊且标注成本高；作者引入轨迹对比较（偏好）作为全局序约束，使无进度标签的失败轨迹可用。
- **对 wiki 的映射：** [Robometer 论文实体](../../wiki/entities/paper-robometer.md)

### 2) 架构与 token 设计

- **位置：** §III-B，式 (1)，Fig. 2、附录 Fig. 11
- **摘录要点：** 基于因果掩码的 Qwen3-VL-4B-Instruct；序列为 `Tok(l) <|video_start|> {Tok(o1_t) <|prog_token|>}_t <|split_token|> {Tok(o2_t)}_t <|pref_token|>`；`<|prog_token|>` 只插入第一条轨迹每帧之后，因果掩码保证只看当前/之前帧；`<|pref_token|>` 在末尾同时看两条轨迹；两条轨迹固定长度 T=8，避免靠轨迹长度判断优劣。推理可输入一条（逐帧进度 + 成功）或两条（另得偏好）。
- **对 wiki 的映射：** [Robometer 论文实体](../../wiki/entities/paper-robometer.md)

### 3) 训练目标 L = L_pref + L_prog + L_succ

- **位置：** §III-C，式 (2)
- **摘录要点：** 偏好用 `<|pref_token|>` 隐状态上的二分类 MLP（BCE）；进度把 p_t=t/T 线性插值投影到 N=10 个均匀 bin 上，用交叉熵训练，推理取 bin 中心期望；成功头 t<T 为 0、t=T 为 1，用按批平衡的 BCE。
- **对 wiki 的映射：** [Robometer 论文实体](../../wiki/entities/paper-robometer.md)

### 4) 数据与采样（RBM-1M，三类构造）

- **位置：** §III-A、§III-D，附录 A
- **摘录要点：** RBM-1M 汇集 21 种机器人本体、超过 100 万条轨迹（Open-X 等专家数据、人类视频如 Epic-Kitchens / RH20T、仿真如 LIBERO、策略 rollout 与失败检测数据中的失败轨迹）。训练样本动态构造：(1) 不同专业程度的进度对比（专家 vs 未标注失败，或进度不同）；(2) 指令负例（换另一条任务指令，被换掉的轨迹进度目标设为 0）；(3) 视频倒带（同一专家轨迹的前进段 vs 倒带段）。固定长度子采样，且总包含首帧与当前帧。
- **对 wiki 的映射：** [Robometer 论文实体](../../wiki/entities/paper-robometer.md)

### 5) 评测与下游

- **位置：** §IV，Table I–IV，Fig. 5–8
- **摘录要点：**
  - Table I（RBM-EVAL-OOD，976 条轨迹、6 个本体、3 个机构采集）：VOC Pearson r，Robometer（RBM-1M）0.94、RoboReward-8B 0.88、TOPReward 0.40、GVL 0.21；Kendall τa，Robometer 0.64（表）而正文写作 0.66（**论文内表文不一致**）、RoboReward-4B 0.50。
  - 消融（Table III）：加偏好损失、再加失败数据，Kendall τ 逐步升高；换成 ReWiND 架构（无预训练 VLM）全面崩溃。
  - 下游：DROID 上 DSRL + π0 在线 RL，单阶段任务 20%→85%（约 10k 步 / ≤45 分钟），两阶段 20%→70%；SO-101 混合质量数据离线 IQL，相对最好基线 2.4×；数据检索 + 微调 π0.5，相对最好基线 4.5×；DROID 失败检测平均 F1 0.81（RoboReward-4B 为 0.74）。
  - 训练配置（附录）：Qwen3-VL-4B-Instruct，8 帧，每卡 batch 16 × 4 卡，lr 2e-5，权重衰减 0.01，6500 步，cosine，warmup 0.1，4 张 H200 约 2 天。
- **对 wiki 的映射：** [Robometer 论文实体](../../wiki/entities/paper-robometer.md)；[TOPReward](../../wiki/entities/paper-topreward.md)（对照基线）

### 6) 局限（作者自述）

- 对子采样视频逐帧打分，难记忆很长时程里的关键事件；真实失败模式多样，训练数据可能未覆盖；视觉语言模型无法直接感知接触力、抓取稳定性等隐状态；未对安全关键状态预训练。

## 关联归档

- [LeRobot 仓库归档](../repos/lerobot.md)
- [TOPReward 论文归档](topreward_arxiv_2602_19313.md)
- [SARM 论文归档](sarm_arxiv_2509_25358.md)
