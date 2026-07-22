# Learning to Fold: prizewinning solution at LeHome Challenge 2026（arXiv:2606.27163）

> 来源归档（ingest）

- **标题：** Learning to Fold: prizewinning solution at LeHome Challenge 2026 (1st place online, 2nd offline)
- **类型：** paper（竞赛技术方案 / tech report）
- **来源：** arXiv abs / PDF；项目博客与 GitHub / HF 交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2606.27163>（v2，2026-07-18 更新；2026-06-25 首发）
  - PDF：<https://arxiv.org/pdf/2606.27163>
  - 项目博客：<https://ilialarchenko.com/projects/lehome2026>
  - 代码：<https://github.com/IliaLarchenko/lehome_solution>
  - 仿真权重：<https://huggingface.co/IliaLarchenko/lehome_sim>
  - 真机权重：<https://huggingface.co/IliaLarchenko/lehome_real>
- **作者：** Ilia Larchenko（个人参赛）
- **机构：** 独立研究者（Ilia Larchenko）；竞赛硬件栈对齐 Hugging Face LeRobot SO-ARM101；策略骨干依赖 Physical Intelligence openpi（π₀.₅）；仿真为 NVIDIA Isaac Sim
- **入库日期：** 2026-07-22
- **一句话说明：** ICRA 2026 **LeHome Challenge** 双臂叠衣竞赛技术方案：π₀.₅ VLA + **AWR+RECAP** 异步分布式 RL（HF Hub 作消息总线）夺 **仿真赛道 1/62（79.63%）**；真机侧剥掉特权头、三桶 BC+DAgger+增强仿真回放夺 **线下实体 2nd（865/1080）**；全链路代码与权重已开源。

## 开源状态（项目页核查 2026-07-22）

- **已开源（完整工程）：** 博客 Paper/Code 区与 README 一致指向 [`IliaLarchenko/lehome_solution`](https://github.com/IliaLarchenko/lehome_solution)（Apache-2.0）——含 **BC/RL 训练、异步 rollout、DAgger/遥操作采集、Isaac Sim 评测、真机 `record_real_dagger` / `serve` 推理**。
- **权重已发布：** [`lehome_sim`](https://huggingface.co/IliaLarchenko/lehome_sim)（线上仿真第 1 名提交策略）；[`lehome_real`](https://huggingface.co/IliaLarchenko/lehome_real)（线下实体第 2 名策略）。
- **依赖边界：** 需作者 fork 的 [`lehome-challenge`](https://github.com/IliaLarchenko/lehome-challenge)（特权仿真数据收集）与 [openpi](https://github.com/Physical-Intelligence/openpi) submodule；正式提交跑官方 [`lehome-official/lehome-challenge`](https://github.com/lehome-official/lehome-challenge)。作者注明代码在竞赛压力下写成、宜作参考而非生产级。
- **互指：** [`sources/sites/ilialarchenko-lehome2026.md`](../sites/ilialarchenko-lehome2026.md) · [`sources/repos/lehome_solution.md`](../repos/lehome_solution.md) · [`sources/sites/huggingface-lehome-sim.md`](../sites/huggingface-lehome-sim.md) · [`sources/sites/huggingface-lehome-real.md`](../sites/huggingface-lehome-real.md)

## 核心论文摘录（MVP）

### 1) 任务：廉价双臂 + 可变形衣物 + 品类隐藏

- **链接：** <https://arxiv.org/abs/2606.27163>；博客 § The challenge
- **摘录要点：** 折叠四类衣物（长袖/短袖上衣、长裤、短裤）；硬件为廉价 **双臂 SO-ARM101**（各 6-DoF）、三路 RGB、12 维关节动作；仿真成功由关键点距离自动判定，真机由评委打分且允许部分成功；评测时衣物品类隐藏；真机赛还要求 **未见评测机** 的 sim→自家机→官方机泛化。
- **对 wiki 的映射：**
  - [Learning to Fold / LeHome 方案](../../wiki/entities/paper-lehome-learning-to-fold.md)
  - [Manipulation](../../wiki/tasks/manipulation.md)
  - [LeRobot](../../wiki/entities/lerobot.md)

### 2) 策略：π₀.₅ + 辅助头 + advantage / garment 条件化

- **链接：** 博客 § Model architecture；arXiv summary
- **摘录要点：** 单策略联合训四类衣物，骨干为竞赛团队在 BEHAVIOR-1K 夺冠改版之上的 **π₀.₅**；同一前向额外预测成功概率、完成度、衣物类型、约 30 帧后的关键点距离与 Q 残差；衣物类型 token（回合初推断）；**RECAP 式 advantage 条件化** + multi-signal AdaRMS；VLM 与 action expert 间 exclusive self-attention；关键点/未来量预测作廉价世界模型替代。
- **对 wiki 的映射：**
  - [Learning to Fold](../../wiki/entities/paper-lehome-learning-to-fold.md)
  - [VLA](../../wiki/methods/vla.md)
  - [STEAM / RECAP 对照](../../wiki/entities/paper-steam-advantage-modeling.md)

### 3) 仿真 RL：AWR + RECAP 异步飞轮（HF Hub 总线）

- **链接：** 博客 § A reinforcement learning loop；README `run_rl_pipeline.py`
- **摘录要点：** 纯 BC 不够稳健；用 **AWR**（按 advantage 重采样）+ **RECAP**（advantage 条件化 + 推理 CFG）把策略推向高优势动作流形。Trainer（约 1×H200、~40 min/ckpt）与任意数量 rollout worker 仅经 **Hugging Face Hub** 异步同步，无同步屏障；rollout 策略含 random / curriculum / success-replay / hard-mining；人可介入遥操作或失败态 DAgger。奖励由关键点进度、成功/完成度预测与跨衣相对成功等经 **GAE** 聚合成帧级 advantage（失败则撤回进度奖励，回合回报保持二元）。
- **对 wiki 的映射：**
  - [Learning to Fold](../../wiki/entities/paper-lehome-learning-to-fold.md)
  - [AWR](../../wiki/methods/awr.md)
  - [DAgger](../../wiki/methods/dagger.md)
  - [VLA 开源复现景观](../../wiki/overview/vla-open-source-repro-landscape-2025.md)

### 4) 推理时优化：Thompson sampling 调参 + Best-of-N

- **链接：** 博客 § Inference-time optimization
- **摘录要点：** 可调执行长度、playback speed、inpainting overlap、CFG guidance（收敛到 **7–9**）、flow-matching noise temperature、Best-of-N（Q 头选 chunk）；用 **Thompson sampling bandit** 在 rollout 中在线调参，后验衰减以跟踪漂移策略；结论：短 horizon 重规划（约每 5 步）优于整段 30-step chunk；N=2–3 足够。
- **对 wiki 的映射：**
  - [Learning to Fold](../../wiki/entities/paper-lehome-learning-to-fold.md)
  - [DreamSteer](../../wiki/entities/paper-dreamsteer-vla-deployment-steering.md) — 另一类推理时 chunk 选择对照

### 5) Sim→Real：剥特权头 + 三桶数据 + 相机叠加对齐

- **链接：** 博客 § Sim to real；README `train_real_bc.yaml` / `record_real_dagger.py`
- **摘录要点：** 从「非最新、较少过拟合」的 sim checkpoint 起步；只保留 action / garment / completion 等可迁移头，去掉 advantage、CFG、Best-of-N、关键点特权头；微调混合 **官方真机 BC 60% + 自家 teleop/DAgger 30% + 强增强 sim replay 10%**；相机 overlay 对齐工具；重增强与故意改机位/光照/标定；运动强度重采样对齐步长；实体决赛 **865/1080，第 2 名**。
- **对 wiki 的映射：**
  - [Learning to Fold](../../wiki/entities/paper-lehome-learning-to-fold.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [DAgger](../../wiki/methods/dagger.md)
  - [NVIDIA SO-101 Sim2Real workflow](../../wiki/entities/nvidia-so101-sim2real-lab-workflow.md)

## 对 wiki 的映射（汇总）

- [`wiki/entities/paper-lehome-learning-to-fold.md`](../../wiki/entities/paper-lehome-learning-to-fold.md) — 主实体页
- [`wiki/methods/vla.md`](../../wiki/methods/vla.md) — π₀.₅ + RL/DAgger 全链路竞赛配方
- [`wiki/methods/awr.md`](../../wiki/methods/awr.md) — AWR 采样加权
- [`wiki/methods/dagger.md`](../../wiki/methods/dagger.md) — 真机/仿真 DAgger
- [`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md) — 廉价臂叠衣 sim→real
- [`wiki/entities/lerobot.md`](../../wiki/entities/lerobot.md) — SO-ARM101 / LeRobot 数据格式
- [`wiki/overview/vla-open-source-repro-landscape-2025.md`](../../wiki/overview/vla-open-source-repro-landscape-2025.md) — 可复现全链路入口
- [`wiki/tasks/manipulation.md`](../../wiki/tasks/manipulation.md) — 可变形衣物操作
- [`wiki/entities/paper-steam-advantage-modeling.md`](../../wiki/entities/paper-steam-advantage-modeling.md) — RECAP / CFGRL 对照
