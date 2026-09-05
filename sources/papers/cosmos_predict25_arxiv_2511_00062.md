# World Simulation with Video Foundation Models for Physical AI（arXiv:2511.00062）

> 来源归档（一手论文）

- **标题：** World Simulation with Video Foundation Models for Physical AI
- **类型：** paper / Physical AI / 视频世界基础模型
- **arXiv：** <https://arxiv.org/abs/2511.00062>
- **机构：** NVIDIA
- **代码：** <https://github.com/nvidia-cosmos/cosmos-predict2.5>、<https://github.com/nvidia-cosmos/cosmos-transfer2.5>
- **入库日期：** 2026-09-05
- **一句话说明：** Cosmos-Predict2.5（flow matching，统一 T2W/I2W/V2W，Reason1 文本编码）与 Cosmos-Transfer2.5（ControlNet 式 Sim2Real / Real2Real）；200M 策展 clip + RL 后训练。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md`](../../wiki/entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md)

## 开源边界（步骤 2.5）

论文释放源码、预训练权重与策展基准，许可 **NVIDIA Open Model License**。项目页级核查：Predict2.5 / Transfer2.5 GitHub + HF 权重可下载。2026-09-05 Predict2.5 README 声明仓进入有限维护，新工作转向 Cosmos 3。→ **已开源（2.5 代可跑；主线已交接）**。

## 核心论文摘录

### 1) 相对 Predict1 的三点升级

- **摘录：** (i) 更强过滤 + 人工策展 Physical AI 后训练数据；(ii) 架构简化，T2W/I2W/V2W 收进单网；(iii) model merging + 新 RL 后训练，T5 换成 **Cosmos-Reason1** VLM 文本编码。
- **对 wiki 的映射：** 论文实体「核心原理」

### 2) 数据规模

- **摘录：** 处理超过 **200M** 原始视频、**3500 万小时**（Predict1 为 2000 万小时）；切出 60 亿+ clip，多级过滤后约 **4%** 存活 → **2 亿** 可训练 clip（Predict1 存活约 30%）。领域：机器人、自动驾驶、智慧空间、人体动力学、物理。
- **对 wiki 的映射：** [`nvidia-cosmos.md`](../../wiki/entities/nvidia-cosmos.md)

### 3) PAI-Bench（Predict）

- **摘录：** Overall = (Domain + Quality)/2。T2W post-train 2B/14B Overall **0.768**，接近 Wan2.2-27B-A14B **0.769**。I2W post-train Overall **0.810**，高于 Wan2.2-27B-A14B **0.806**。人类偏好：14B post-train vs Wan2.1-14B 为 **48.6% vs 31.8%**。
- **对 wiki 的映射：** 论文实体「评测」

### 4) Transfer2.5 与下游

- **摘录：** Transfer2.5-2B 比 Transfer1-7B **小 3.5×** 且在 PAIBench-Transfer 的控制遵循与画质上更高。驾驶多视角 FVD/FID 最高约 **2.3×** 提升。DreamGen / GR1 后训练 14B 在未见物体 / 环境指令跟随上超过 Hunyuan、CogVideoX、WAN 2.1。
- **对 wiki 的映射：** [Sim2Real](../../wiki/concepts/sim2real.md)、[mimic-video](../../wiki/methods/mimic-video.md)

### 5) 训练效率

- **摘录：** 4096×H100，720p / 93 帧：2B MFU **36.49%**（CP=2）；14B MFU **33.08%**（CP=8）。

## 对 wiki 的映射

- Canonical 实体：[`wiki/entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md`](../../wiki/entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md)
- 仓库：[`sources/repos/nvidia_cosmos_predict25.md`](../repos/nvidia_cosmos_predict25.md)
- 下一代：[`wiki/entities/cosmos-3.md`](../../wiki/entities/cosmos-3.md)
- Transfer 族：[`wiki/entities/cosmos-transfer.md`](../../wiki/entities/cosmos-transfer.md)
