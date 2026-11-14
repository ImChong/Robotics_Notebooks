# LingBot-VLA：A Pragmatic VLA Foundation Model

> 来源归档

- **标题：** A Pragmatic VLA Foundation Model（LingBot-VLA 1.0）
- **类型：** paper
- **版本：** arXiv:2601.18692v4（2026-06-15）
- **arXiv：** <https://arxiv.org/abs/2601.18692>
- **PDF（仓库内链）：** <https://github.com/robbyant/lingbot-vla/blob/main/assets/LingBot-VLA.pdf>
- **项目页：** <https://technology.robbyant.com/lingbot-vla>
- **代码：** <https://github.com/robbyant/lingbot-vla>
- **机构：** Robbyant（蚂蚁灵波科技 / 蚂蚁集团具身智能）；核心作者 Wei Wu*、Fan Lu*、Yunnan Wang*、Shuai Yang*、Shi Liu*、Fangjing Wang* 等，Kecheng Zheng† 领衔；数据/硬件协作致谢 Galaxea、AgileX、乐聚（Leju）
- **入库日期：** 2026-07-17
- **通读：** 已完成
- **核对范围：** 论文全文 `2601.18692.txt`（6178 行，arXiv:2601.18692v4 完整正文 §§1–6、References 与 Appendix A Tables S1–S9，涵盖 4 平台 GM-100 分任务与 RoboTwin 2.0 50 任务全表）及官方仓 `robbyant/lingbot-vla` 源码实现。
- **一句话说明：** Robbyant **务实 VLA 基础模型 1.0**：**2 万小时**、**9 类双臂机器人** 真机遥操作预训练 + **Qwen2.5-VL-3B MoT + Flow Matching 动作头** + 可选 **LingBot-Depth 深度查询蒸馏**；FSDP2/HSDP + FlexAttention + `torch.compile` 实测 8 卡 **261 samples/s**（1.5–2.8× 提速），GM-100 四平台与 RoboTwin 2.0 显著超越 π₀.₅、GR00T N1.6 与 WALL-OSS。

## 核心摘录（面向 wiki 编译）

### 1) 大规模双臂真机预训练与自动化标注（§3）

- **数据规模与 9 类构型（§3.1 & Fig. 2）：** 约 **20,000 小时** 真机遥操作数据，覆盖 **9 种主流双臂构型**：AgiBot G1、AgileX、Galaxea R1Lite、Galaxea R1Pro、Realman RS-2（16 维）、Leju KUAVO 4 Pro（双足人形）、Qinglong（双足人形）、ARX Lift2、Bimanual Franka（16 维）。
- **两段式标注（§3.2）：** 先由人工按原子动作切分多视角视频并剔除首尾静态冗余帧，再用 **Qwen3-VL-235B-A22B** 生成任务与子任务语言指令。

### 2) MoT Flow-Matching 架构、LingBot-Depth 蒸馏与系统优化（§4）

- **MoT 骨干（§4.1 & Eqs. 1–4）：** 借鉴 BAGEL 的 Mixture-of-Transformers 结构，将 **Qwen2.5-VL-3B-Instruct**（Understanding Expert）与 **Action Expert**（随机初始化）通过 **共享自注意力** 逐层耦合；按 π₀ 式分块因果掩码划分为 `[I1_t, I2_t, I3_t, T_t]`、`[s_t]`、`[a_t, ..., a_{t+T-1}]`（预训练 `T=50`），目标为条件流匹配速度场 MSE 损失 \(\mathcal{L}_{FM}\)。
- **空间深度蒸馏（§4.1 Eq. 5）：** 引入三视角可学习查询 `[Q1_t, Q2_t, Q3_t]`，经交叉注意力投影后与 **LingBot-Depth** 教师 token `[D1_t, D2_t, D3_t]` 做 L1 + 相似度对齐（\(\mathcal{L}_{distill}\)），部署时不依赖深度输入或教师网络。
- **训练效率工程（§4.2 & Fig. 4）：** 基于 VeOmni 思想采用 **FSDP2 / HSDP**（仅为 Action Expert 建独立 shard group）、fp32 规约 + bf16 存储通信、**FlexAttention** 稀疏融合与 **`torch.compile`**；在 `Qwen2.5-VL-3B-π`（local bs=32）上达到 **261 samples/s**（8 GPU）至 **7356 samples/s**（256 GPU），在 `PaliGemma-3B-pt-224-π` 上达到 **229 → 6411 samples/s**，相对 StarVLA / Dexbotic / OpenPI 提速 **1.5–2.8×**。

### 3) 评测与 Scaling Laws（§5 & Tables 1–2, S1–S9, Figs. 4–6）

- **GM-100 真机四平台对比（Table 1，每任务 130 集微调、每对 15 次试验共 22,500 trials，3 分钟上限）：** v4 版覆盖 Agibot G1、AgileX、Galaxea R1Pro 与 Leju KUAVO 4 Pro，四平台平均：**WALL-OSS 3.85% SR / 10.70% PS**、**GR00T N1.6 7.31% SR / 16.66% PS**、**π₀.₅ 13.00% SR / 27.32% PS**、**LingBot-VLA w/o depth 16.20% SR / 34.32% PS**、**LingBot-VLA w/ depth 16.87% SR / 35.16% PS**（较 π₀.₅ 提高 +3.87 pt SR / +7.84 pt PS；注：仅算前 3 轮式平台时为 README 中的 15.74% / 17.30% SR）。
- **RoboTwin 2.0 50 任务（Table 2 & Table S9）：** Clean（50 demo/task）与 Randomized（500 demo/task）：π₀.₅ **82.74% / 76.76%** vs w/o depth **86.50% / 85.34%** vs w/ depth **88.56% / 86.68%**。
- **Scaling & 数据效率（§5.5 Figs. 5–6）：** 预训练数据从 **3,000 h** 扩至 **20,000 h** 时 25 项代表性任务 PS/SR 持续单调上升、尚未饱和；AgiBot G1 8 项任务仅用 **80 条演示**微调即超越用满 **130 条**的 π₀.₅。

## 对 wiki 的映射

- [x] `wiki/entities/lingbot-vla.md` 新建
- [x] 与 `wiki/entities/lingbot-vla-v2.md` 区分代际
- [x] 修正 [loco_manip_161_survey_152_lingbot-vla.md](loco_manip_161_survey_152_lingbot-vla.md) 误链至 2.0 页
