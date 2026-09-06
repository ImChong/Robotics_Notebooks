# PAI-Bench: A Comprehensive Benchmark For Physical AI（arXiv:2512.01989）

> 来源归档（一手论文）

- **标题：** PAI-Bench: A Comprehensive Benchmark For Physical AI
- **类型：** paper / benchmark / Physical AI / 视频生成 / 视频理解
- **arXiv：** <https://arxiv.org/abs/2512.01989>
- **PDF：** <https://arxiv.org/pdf/2512.01989>
- **机构：** 佐治亚理工学院（Georgia Tech）、卡内基梅隆大学（CMU）
- **作者：** Fengzhe Zhou, Jiannan Huang, Jialuo Li, Deva Ramanan, Humphrey Shi
- **会议：** CVPR 2026 **Oral**
- **代码：** <https://github.com/SHI-Labs/physical-ai-bench>
- **Leaderboard：** <https://huggingface.co/spaces/shi-labs/physical-ai-bench-leaderboard>
- **入库日期：** 2026-09-06
- **一句话说明：** 首个统一评测 Physical AI **感知（MLLM 视频理解）** 与 **预测（VGM 生成/条件生成）** 的基准：2,808 真实场景案例，三轨 G/C/U + Domain/Quality 或控制保真指标；大规模实验显示 VGM 画质高但物理一致性弱，MLLM 远落后人类。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-sa-2512-01989-pai-bench-a-comprehensive-benchmark-for-physical-ai.md`](../../wiki/entities/paper-sa-2512-01989-pai-bench-a-comprehensive-benchmark-for-physical-ai.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（MIT；GitHub + HF 三数据集 + Leaderboard Space） |
| **评测代码** | `generation/`、`conditional_generation/`、`understanding/` 三轨独立 `uv` 环境 |
| **数据** | HF：`physical-ai-bench-generation` / `-conditional-generation` / `-understanding` |
| **榜单** | HF Space：`shi-labs/physical-ai-bench-leaderboard` |
| **致谢** | 论文致谢 NVIDIA Research / Cosmos team 对 PAI-Bench 创建的支持 |

## 核心论文摘录

### 1) 三轨统一设计

- **摘录：** **PAI-Bench-G**（Generation）：1,044 视频–提示对 + 5,636 QA，评 VGM 的 **Quality Score**（VBench 八指标）与 **Domain Score**（Qwen3-VL-235B MLLM-as-Judge 答物理 QA）。**PAI-Bench-C**（Conditional）：600 视频（AgiBot / OpenDV / Ego-Exo-4D 各 200），Blur/Edge/Depth/Seg 控制保真 + DOVER 画质 + LPIPS 多样性。**PAI-Bench-U**（Understanding）：1,214 QA / 1,027 视频，物理常识（Space/Time/Physics）+ 具身推理（BridgeData、RoboVQA、RoboFail、AgiBot、HoloAssist、AV）。
- **对 wiki 的映射：** 论文实体「核心原理」「流程总览」

### 2) PAI-Bench-G 主结果（Overall = Domain+Quality 平均）

- **摘录：** 真源视频 Overall **83.9**；闭源 Veo3 **82.2**；开源最佳 Wan2.2-I2V-A14B **82.3**、Cosmos-Predict2.5-2B **81.4**。Quality 接近真源（~78），Domain 普遍低于真源（真源 Avg. **89.8**，Wan2.2 **87.1**，Veo3 **86.8**）——**画质与物理合理性脱节**。
- **对 wiki 的映射：** 论文实体「评测」；交叉 [`cosmos_predict25`](../../wiki/entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md)

### 3) 人类偏好对齐

- **摘录：** Arena 人类 pairwise 与自动指标 ELO 的 Pearson **r=0.918**（Quality + Domain）。
- **对 wiki 的映射：** 论文实体「评测」

### 4) PAI-Bench-C 条件生成

- **摘录：** Cosmos-Transfer **All** 多信号条件 Quality **9.24** 高于任一单控制；Seg 控制 Mask mIoU 反而最低（噪声监督假说）。Transfer2.5-2B 全面优于 Transfer1-7B。
- **对 wiki 的映射：** [`cosmos-transfer.md`](../../wiki/entities/cosmos-transfer.md)

### 5) PAI-Bench-U 理解

- **摘录：** 人类 Overall **93.2**；GPT-5 **61.8**；开源最佳 Qwen3-VL-235B **64.7**（超 GPT-5）。零帧输入降至随机猜水平；1 帧 vs 32 帧差距大——需时序而非语言先验。GPT-5 thinking（medium）+8.0；Qwen3 系列 thinking 略降。
- **对 wiki 的映射：** 论文实体「评测」；[`generative-world-models.md`](../../wiki/methods/generative-world-models.md)

### 6) 与现有榜对比

- **摘录：** Table 1：相对 VBench / EvalCrafter / Physics-IQ / VideoMME / PhyGenBench 等，PAI-Bench 唯一同时覆盖 **生成 + 条件生成 + 理解**，且 span AV / 工业 / 具身 / 自我中心 / 物理常识。
- **对 wiki 的映射：** 论文实体「与其他工作对比」

## 对 wiki 的映射

- Canonical 实体：[`wiki/entities/paper-sa-2512-01989-pai-bench-a-comprehensive-benchmark-for-physical-ai.md`](../../wiki/entities/paper-sa-2512-01989-pai-bench-a-comprehensive-benchmark-for-physical-ai.md)
- 仓库：[`sources/repos/physical_ai_bench.md`](../repos/physical_ai_bench.md)
- HF 入口：[`sources/sites/hf-physical-ai-bench.md`](../sites/hf-physical-ai-bench.md)
- 被引模型页：Cosmos Predict2.5、Cosmos Transfer、Kairos、PhysisForcing 等已在各自页引用 PAI-Bench 分数——本页为 **榜的定义与复现入口**
