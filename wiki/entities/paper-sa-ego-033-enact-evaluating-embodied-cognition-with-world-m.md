---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-11-26
venue: "ICLR 2026"
summary: "ICLR 2026 提出基于 POMDP 场景图状态转移的第一视角具身认知评测基准 ENACT（8,972 条仿真长程交互 QA + 960 条真机/实景 QA），将具身世界建模形式化为前向观测重排与逆向动作重排双任务；评测 30 个前沿 VLM 发现模型普遍“逆向强于前向”、随步长急剧崩塌，且存在显著右利手与类人视角偏置。"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_noarxiv_033_enact-evaluating-embodied-cognition-with.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
  - ../../sources/papers/sun_awesome_ego_noarxiv_104_enact-evaluating-embodied-cognition-with.md
---

# ENACT

**ENACT: Evaluating Embodied Cognition with World Modeling of Egocentric Interaction**（ICLR 2026，西北大学、斯坦福大学 & UCLA）收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 033/249** 篇，分组 **24 Long-form Video Understanding & Structured Reasoning**。本页基于论文全文完成结构化精读与核实。

## 一句话定义

**ENACT** 将视觉–语言模型（VLM）的具身认知评测统一形式化为部分可观测马尔可夫决策过程（POMDP）下的第一视角交互世界建模，构造前向观测重排与逆向动作重排共 **8,972** 道长程 VQA 题（3–10 步），揭示了现有前沿 VLM 随交互步长急剧退化、逆向优于前向、且带有明显类人视角与右利手偏置的认知边界。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| POMDP | Partially Observable Markov Decision Process | 部分可观测马尔可夫决策过程，本文用于形式化第一视角长程物理交互与状态记忆 |
| KFTS | Key-Frame Trajectory Synthesis | 本文基于有向无环图与动态规划的可扩展关键帧轨迹采样算法 |
| TA / PA | Task Accuracy / Pairwise Accuracy | 全序列精确/语义重排通过率（TA）与相邻状态–动作对一致率（PA） |
| VLM | Vision-Language Model | 视觉–语言模型，本文在零样本重排设定下评测了 7 个闭源与 23 个开源 VLM |
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |

## 为什么重要

- **剥离像素生成噪声，直击具身因果与物理状态记忆**：传统视频世界模型评测常受限于画质生成保真度，而静态空间 QA 又缺乏动作干预；ENACT 通过“已知动作重排观测（Forward）”与“已知观测重排动作（Inverse）”精准检验可供性识别、动作–效果因果与部分可观测长期空间记忆。
- **可规模化自动生成且 Sim-to-Real 趋势一致**：基于 BEHAVIOR（Isaac Sim）物理状态自动抽取符号场景图变化，单条演示即可组合生成海量轨迹；同时在 960 道真机/实景操作 QA 上验证了 `InternVL3.5-241B` 的精度与衰减曲线与仿真高度吻合（渲染写实度敏感性 $p \ge 0.2$）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 033/249（Awesome Egocentric Vision） |
| 分组 | 24 Long-form Video Understanding & Structured Reasoning |
| 出处 | ICLR 2026 |
| 作者与机构 | Qineng Wang¹*, Wenlong Huang²*, Yu Zhou³, Hang Yin², Tianwei Bao¹, Jianwen Lyu¹, Weiyu Liu², Ruohan Zhang²†, Jiajun Wu²†, Li Fei-Fei²†, Manling Li¹†（¹西北大学，²斯坦福大学，³UCLA） |
| 数据规模 | 源自 BEHAVIOR 29 项长程居家移动操作任务，共 **8,972** 道仿真 QA（步长 $L \in \{3,\dots,10\}$，每步长约 560 题/任务）+ **960** 道真实日常场景（厨房/餐桌/工作台）QA |
| 开源状态 | 论文声明代码与完整数据集已公开发布（见 Reproducibility Statement 与项目页） |
| 论文与项目页 | <https://openreview.net/forum?id=Patx6MRipw> · <https://enact-embodied-cognition.github.io/> |

## 核心机制

1. **POMDP 符号场景图形式化与双任务构造（§2.1）**
   - 状态 $s_t \in S$ 为机器人第一视角下的符号场景图，覆盖 **11 类物理交互谓词**（`RightGrasping`, `LeftGrasping`, `OnTop`, `Inside`, `Under`, `Contains`, `Covered`, `Open`, `Cooked`, `Transition`, `ToggledOn`）；观测 $o_t$ 为统一缩放至 $512 \times 512$ 的第一视角 RGB 图像；动作 $a_k := \Delta_{\text{Vis}}(s_{i_{k+1}}, s_{i_k})$ 为相邻关键帧之间在画面中**视觉可验证**的场景图变化。
   - **Forward World Modeling（前向世界建模）**：给定初始观测 $o_0$、有序动作序列 $(a_0,\dots,a_{L-2})$ 与打乱后的未来观测集 $O'$，要求输出观测重排索引 $\sigma$。
   - **Inverse World Modeling（逆向世界建模）**：给定有序观测序列 $(o_0,\dots,o_{L-1})$ 与打乱后的动作集 $A'$，要求输出动作重排索引 $\tau$。
2. **可扩展关键帧轨迹合成管线 KFTS（§2.2 & Appendix A.2）**
   - **状态稳定性过滤**：从 30Hz 轨迹中筛出持续 $\ge 40$ 帧（约 $1.3\text{s}$）的抽象状态变化以抑制高频抖动噪声，且仅当新变化签名的余弦相似度 $< 0.97$ 时保留关键帧，得到 $M$ 个去重关键帧。
   - **DAG 构造 + 动态规划回溯采样（Algorithm 1）**：将支持跳帧的合法可见转移建为有向无环图（DAG），通过动态规划计算各长度路径数 $DP[\ell, i]$ 并加权回溯采样，以多项式复杂度替代 $\binom{M}{L}$ 暴力枚举。
3. **语义在线验证器与结构化错误拆解（§2.3 & §3.5）**
   - **在线验证器**：针对多解合理性，前向任务放宽为“参考可见变化 $\subseteq$ 预测前后帧真实变化”（$C_i \subseteq \tilde{C}_i$），逆向任务放宽为“预测动作变化 $\subseteq$ 参考全量变化”（$\tilde{C}_i \subseteq F_i$），并统计全对率 **Task Accuracy (TA)** 与步级局部一致率 **Pairwise Accuracy (PA)**。
   - **五类结构化错误归因**：将模型输出序列还原为原子谓词变化集合并与真值比对，拆分为 **Hallucination（幻觉预测）**、**Omission（漏检状态变化）**、**Polarity Inversion（极性反转）**、**Entity Substitution（实体错配）** 与 **Predicate Substitution（谓词错配）**。

### 方法流程图

```mermaid
flowchart LR
    A["BEHAVIOR 30Hz 移动操作轨迹 (RGB + 物理仿真状态)"] --> B["抽取符号场景图 (11 类谓词) + 稳定性/余弦阈值 (<0.97) 滤波"]
    B --> C["KFTS 算法: 构建可见变化 DAG + DP 路径计数加权回溯采样"]
    C --> D["构建 8,972 条长度 L=3..10 的双向重排 QA"]
    D --> E["Forward 任务: 给定 o_0 与有序动作, 重排打乱图像"]
    D --> F["Inverse 任务: 给定有序图像, 重排打乱动作"]
    E --> G["在线语义子集验证器输出 TA / PA 与 5 类错误归因"]
    F --> G
```

## 评测与指标

### 1. 主基准 Pairwise Accuracy (%) 对比（原文 Table 9，随机基线外人类 Krippendorff's $\alpha = 0.8320$）

| 模型 / 基线 | 类别 | Forward $L=3$ | Forward $L=5$ | Forward $L=7$ | Forward $L=10$ | Inverse $L=3$ | Inverse $L=5$ | Inverse $L=7$ | Inverse $L=10$ |
|-------------|------|---------------|---------------|---------------|----------------|---------------|---------------|---------------|----------------|
| **Human Performance** | 人类上界 | 93.62 | 95.04 | 95.43 | **95.13** | 92.05 | 94.35 | 95.96 | **96.29** |
| `GPT-5` | 闭源 | 84.62 | 69.96 | 57.48 | **46.93** | 86.28 | 76.09 | 65.71 | 55.33 |
| `GPT-5 mini` | 闭源 | **87.50** | **70.65** | **58.14** | 44.11 | 85.05 | 75.43 | 63.79 | 50.02 |
| `Gemini 2.5 Pro` | 闭源 | 86.10 | 69.83 | 53.26 | 36.98 | **87.94** | 75.39 | **66.03** | **56.62** |
| `Claude Sonnet 4` | 闭源 | 65.65 | 36.65 | 26.61 | 20.16 | 73.25 | 48.87 | 37.00 | 28.49 |
| `InternVL3.5-241B-A28B` | 开源 | 75.79 | **50.83** | **37.84** | **25.24** | **82.26** | **60.61** | **45.90** | 30.56 |
| `Qwen2.5-VL-72B-Ins` | 开源 | **78.15** | 49.87 | 36.77 | 25.07 | 77.80 | 53.30 | 44.07 | 36.27 |
| `Qwen2.5-VL-32B-Ins` | 开源 | 67.83 | 44.35 | 27.52 | 18.07 | 63.55 | 54.57 | **49.36** | **40.16** |

### 2. 长程崩塌、人类中心偏置与错误归因关键数值（Table 8, 13 & Fig. 4, 5）

- **全序列 Task Accuracy 随步长剧烈崩塌（Table 8）**：在 $L=10$ 时，人类仍维持 Forward **84.00%** / Inverse **85.00%**；而 `GPT-5` 跌至 **5.00% / 13.00%**，`Gemini 2.5 Pro` 跌至 **3.60% / 14.40%**，绝大多数开源模型（含 Qwen2.5-VL-72B）跌至 **0.00%–0.40% / 0.00%–2.00%**（`InternVL3.5-241B` 最高也仅 2.00%）。
- **Sim-to-Real 一致性与渲染无关性（Table 13 & §3.2）**：`InternVL3.5-241B` 在 960 道真机/实景 QA 上呈现极相似绝对水平（Forward PA 从 $L=3$ 的 80.00% 降至 $L=10$ 的 25.74%；Inverse PA 从 90.00% 降至 26.85%）；Path Tracing、Ray Tracing Only 与 GPT-image-1 写实迁移之间无统计显著差异（$p \ge 0.2$，$|\Delta| < 0.05$）。
- **人类中心视觉与右利手偏置（§3.3–3.4 & Fig. 4）**：
  - 相机视场/高度：偏离人眼 FOV 的 Aperture 60 / 80 / Fisheye 显著掉点（$p \le 0.01$）；机位升高 `+0.5 m` 导致 `GPT-5 mini` 前向 PA 显著下降 $\Delta = -0.13$。
  - 外观鲁棒但右利手偏置显著：更换机器人颜色（白/随机/肤色）无显著影响（$p > 0.10$），但左手操作识别全面落后右手（`GPT-5 mini` 前向右手 Prec/Rec 0.4976/0.4958 vs. 左手 0.4087/0.4040；且 **9.38%** 的真值左手变化被错归为右手，仅 **4.67%** 右手变化被误判为左手）。
- **结构化错误主导项（§3.5 & Fig. 5，GPT-5）**：**Hallucination（43.9% Forward / 41.8% Inverse）与 Omission（37.1% Forward / 41.8% Inverse）** 合计占全部推理失败的 **81%（Forward）和 ~84%（Inverse）**；极性反转占 12.4% / 9.2%，实体替换 6.3% / 5.4%，谓词替换仅 0.3% / 1.9%。

## 与其他工作对比

- **对比静态空间基准与非交互视频预测（ECBench、WorldPrediction、GVL）**：ENACT 显式定义了由 11 类底层物理谓词构成的符号动作空间，既评测长程多步干预因果（区别于纯感知问答 ECBench），又作为条件转移模型而非进度价值估计器（区别于 GVL）或纯片段规划（区别于 WorldPrediction）。
- **对比具身训练模型 `Cosmos-Reason1-7B`（§3.1 & Fig. 18）**：在同量级（~7B–12B）模型中，经过具身数据训练的 `Cosmos-Reason1-7B` 在交互步长 $L > 5$ 时衰减明显更慢、整体领先同类通用开源模型。更多对比见 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md)。

## 常见误区或局限

1. **消融实验仅覆盖代表性模型子集（Appendix F）**：受限于闭源 API 与大规模推理成本，渲染写实度、相机内参、机器人外观与误差归因主要围绕 `GPT-5`、`GPT-5 mini` 与 `InternVL3.5-241B-A28B` 展开。
2. **仅做零样本诊断，尚未开展微调实验（Appendix F）**：尽管 `KFTS` 管线可为单条机器人轨迹生成百万级交互 QA，本文尚未将合成数据用于 VLM 具身世界建模微调训练。
3. **未评测纯视频生成式世界模型（Appendix F）**：由于视频生成模型常出现物理不一致且缺乏自动化忠实评测指标，除统一架构 `BAGEL` 外，未将 Sora/Genie 类视频生成器纳入横向评测。
4. **离散场景图抽象舍弃了连续动力学细节（Appendix A.2.2）**：11 类布尔/关系谓词聚焦语义状态跃迁，不能替代连续接触力、关节力矩或毫米级位姿控制精度的评估。

## 结论

**ENACT 揭示了“离线图文预训练赋予了 VLM 较强的逆向语言回溯能力，但离长程具身前向世界建模与非人类构型泛化仍有巨大鸿沟”。**

1. **不能用逆向动作识别代替前向物理推演来评估 VLA/VLM** — 几乎所有模型在 Inverse 上均高于 Forward（如 $L=10$ 时 Gemini 2.5 Pro 56.62% vs. 36.98%），前向状态预测才是暴露空间记忆短板的试金石。
2. **长程交互（$L \ge 8$）的真正瓶颈是物体持久性漏检（Omission）与先验幻觉（Hallucination）** — 二者合计占 GPT-5 错误的 81%–84%，工程上需引入显式场景图/物体状态记忆而非依赖盲目上下文自回归。
3. **部署在非人眼相机（鱼眼、高广角、非人眼高度）或双臂/左利手任务前须做数据增广** — VLM 对广角/Fisheye（$p \le 0.01$）及左手抓取（混错率翻倍）存在强烈类人先验偏置。
4. **仿真物理引擎（如 Isaac Sim/BEHAVIOR）完全可以作为具身世界建模的可靠评测代理** — 写实度变化对推理得分影响不显著（$p \ge 0.2$），且真机评测曲线与仿真高度同步。

## 关联页面

- 列表实体：[Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md)
- 技术地图：[Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md)
- 方法/任务：[vla.md](../methods/vla.md)、[imitation-learning.md](../methods/imitation-learning.md)、[manipulation.md](../tasks/manipulation.md)、[teleoperation.md](../tasks/teleoperation.md)

## 参考来源

- [`sources/papers/sun_awesome_ego_noarxiv_033_enact-evaluating-embodied-cognition-with.md`](../../sources/papers/sun_awesome_ego_noarxiv_033_enact-evaluating-embodied-cognition-with.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://openreview.net/forum?id=Patx6MRipw>
- [`sources/papers/sun_awesome_ego_noarxiv_104_enact-evaluating-embodied-cognition-with.md`](../../sources/papers/sun_awesome_ego_noarxiv_104_enact-evaluating-embodied-cognition-with.md) — 本条目策展摘录

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [ENACT 项目主页](https://enact-embodied-cognition.github.io/) · [OpenReview 原文](https://openreview.net/forum?id=Patx6MRipw)
