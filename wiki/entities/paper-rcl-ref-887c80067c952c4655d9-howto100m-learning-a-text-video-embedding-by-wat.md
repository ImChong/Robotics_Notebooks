---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
venue: "ICCV 2019"
summary: "RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_ref_887c80067c952c4655d9_howto100m-learning-a-text-video-embeddin.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# HowTo100M

**HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 111/564** 篇，分组 **Datasets**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

完全无需人工字幕标注的超大规模教学视频-文本对比预训练数据集与联合嵌入方法（HowTo100M, ICCV 2019）：利用 WikiHow 任务检索与 YouTube 自动语音识别（ASR）字幕收集 **122.1 万段视频（134,472 小时）与 1.366 亿个片段-旁白对**，通过**上下文门控映射 + Max-Margin 排序损失 + 同视频内负采样**学得强大跨模态视频表征。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ASR | Automatic Speech Recognition | 自动语音识别（生成教学视频的弱监督旁白字幕） |
| R@K | Recall at K | 文本检索视频时前 K 位返回结果命中真值片段的召回率 |
| Median R | Median Rank | 正确视频片段在检索排序中的中位排名（越低越好） |
| PT / FT | Pre-trained / Fine-tuned | 在 HowTo100M 上预训练及在下游数据集上微调 |
| 1D-CNN | One-Dimensional Convolutional Neural Network | 本文用于从 word2vec 序列提取句级文本特征的网络 |

## 为什么重要

- 将视频-文本数据规模一举扩大 **3 个数量级**（`136.6M` 片段 vs 同期最大 `How2 185k`），且零人工标注成本，证明了互联网噪声旁白（尽管仅 51% 片段直接可视对应语义）在大规模预训练下能涌现超越小规模全监督模型的视频理解能力。
- 开创了大规模教学视频作为机器人操作/具身视频表征预训练底座的路线（广泛支撑后续视频-语言模型、动作步骤定位与任务规划研究）；可顺着 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 对照阅读。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 111/564 |
| 分组 | Datasets |
| 出处 | ICCV 2019 (pp. 2630–2640) |
| 作者与机构 | Antoine Miech*, Dimitri Zhukov*, Jean-Baptiste Alayrac, Ivan Laptev, Josef Sivic et al. (ENS / Inria / CIIRC CTU) |
| 论文 | <https://openaccess.thecvf.com/content_ICCV_2019/papers/Miech_HowTo100M_Learning_a_Text-Video_Embedding_by_Watching_Hundred_Million_Narrated_ICCV_2019_paper.pdf> |
| 项目主页 | <https://www.di.ens.fr/willow/research/howto100m>（含数据特征与预训练模型） |
| 子类 / 象限 | 视频语言预训练数据 · 语言标注与再标注 · 视频动作理解数据 · 弱监督对比学习 |

## 核心机制（归纳）

> 注：以下内容基于通读 ICCV 2019 开放获取原文全文核对补充。

### 1. 免人工标注的数据收集管线（136.6M 片段-字幕对）

1. **WikiHow 物理视觉任务筛选**：从 WikiHow 12 万条目中限定 12 个含物理世界交互的类目并过滤非物理动词，获得 **23,611 项视觉任务**（`Food & Entertaining` 11,504 项/54.4M 片段、`Home & Garden` 5,068 项/29.5M、`Hobbies & Crafts` 4,273 项/29.8M 为主力）。
2. **YouTube 检索与字幕对齐**：用 `how to <task>` 抓取排名前 `200`、带英文字幕（人工或 ASR 转录/翻译）、播放量 `≥ 100`、文本 `≥ 100` 词且时长 `≤ 2,000 s` 的视频并按 YouTube ID 去重，共获 **1.221M 视频**（平均 6.5 分钟，总计 **134,472 小时 ≈ 15 年**），按字幕起止时间切分为 **136.6M 片段-字幕对**（每片段均值 4 s、去停用词后 4 词）。随机抽检显示 `71%` 为教学视频、`12%` vlog、`7%` 评测广告；约 **51%** 片段-字幕对在视觉上出现对应物体或动作。

### 2. 上下文门控联合嵌入与同视频内负采样

1. **双模态特征提取（$d_v=d_c=d=4096$，总参数 67M）**：
   - 视频特征 $v$：拼接 ImageNet 预训练 `ResNet-152`（1 fps 2D 帧特征）与 Kinetics 预训练 `ResNeXt-101 16-frames`（1.5 特征/s 3D 时空特征）并时序最大池化为 `4096` 维。
   - 文本特征 $c$：在 GoogleNews `word2vec` 词向量上接浅层 `1D-CNN` 输出 `4096` 维。
2. **上下文门控非线性映射**：通过带门控调制的双线性投影映射至共享空间并计算余弦相似度 $s(V, C)$：
   $$f(v) = (W_1^v v + b_1^v) \circ \sigma\!\left(W_2^v (W_1^v v + b_1^v) + b_2^v\right),\quad g(c) = (W_1^c c + b_1^c) \circ \sigma\!\left(W_2^c (W_1^c c + b_1^c) + b_2^c\right)$$
3. **Max-Margin 损失与 Intra-negative 策略**：使用 margin $\delta = 0.1$ 的双向 max-margin 排序损失（Adam，`lr = 10⁻⁴`，单张 Tesla P100 训练全量数据耗时 `< 3 天`）。关键设计是 **50% 负样本采自同一原视频的不同时间片段（intra-negative）**、50% 采自其他视频，强迫模型区分动作与操作对象本身（如“切纸” vs “切木头”）而非靠背景（如厨房场景）作弊。

### 方法流程图

```mermaid
flowchart LR
    WH["WikiHow 物理动词过滤 (12 类 / 23,611 视觉任务)"] --> YT["YouTube 检索过滤 (Top200 / ≥100播放 / ≥100词 / ≤2000s)"]
    YT --> SEG["ASR/人工字幕时间戳切分 (1.221M 视频 / 136.6M 片段-字幕对)"]
    SEG --> VF["视频特征: ResNet-152 (2D) + ResNeXt-101 (3D) -> 4096d"]
    SEG --> TF["文本特征: word2vec + 浅层 1D-CNN -> 4096d"]
    VF --> CG["上下文门控投影 f(v), g(c) -> 4096d 共享空间"]
    TF --> CG
    CG --> LOSS["双向 Max-Margin 损失 (δ=0.1) + 50% 同视频内负采样"]
```

## 评测与指标

- **同视频负采样消融（Table 3）**：启用 intra-negative 后，`YouCook2 R@10` 从 `18.1` 提升至 **`24.8`**，`CrossTask AVG Recall` 从 `25.7` 提升至 **`33.6`**，`LSMDC R@10` 从 `12.3` 提升至 **`14.0`**（`MSR-VTT R@10` 为 `30.1 -> 29.6`）。
- **规模扩展律（Fig. 3）**：随着训练视频按搜索排名阈值从 `15k`（Top 2）增至 `1.22M`（Top 200），CrossTask、YouCook2、MSR-VTT 与 LSMDC 四项评测指标持续单调上升且**尚未饱和**。
- **CrossTask 动作步骤定位（Table 4, 18 任务 2.7k 教学视频，平均召回率 %）**：
  - 仅在 HowTo100M 上零样本训练的开箱模型达到 **`33.6%`**，大幅超越弱监督方法 `Alayrac et al. (13.3%)` 与 `Zhukov et al. (22.4%)`，甚至超越了使用真值片段监督训练的**全监督上界（`31.6%`）**（如 *Add Oil to Car* 达 `30.7%` vs `6.4%`）。
- **YouCook2 烹饪视频片段检索（Table 5, 3.5k 验证片段）**：
  - `HGLMM FV CCA`（YouCook2 训）：`R@1=4.6, R@5=14.3, R@10=21.6, Median R=75`；
  - `Ours`（仅 HowTo100M 预训练零样本）：`R@1=6.1, R@5=17.3, R@10=24.8, Median R=46`（直接击败域内训练模型）；
  - `Ours`（**PT: HowTo100M + FT: YouCook2**）：**`R@1=8.2, R@5=24.5, R@10=35.3, Median R=24`**（`R@10` 提升 `13.7` 个百分点）。
- **MSR-VTT 通用视频检索（Table 6 & Fig. 4）**：
  - 零样本 `HowTo100M`：`R@1=7.5, R@5=21.2, R@10=29.6, Median R=38`；
  - **`PT: HowTo100M + FT: MSR-VTT`**：**`R@1=14.9, R@5=40.2, R@10=52.8, Median R=9`**（全面超越域内 SOTA `JSFusion` 的 `10.2 / 31.2 / 43.2 / 13`），且仅用 **20% MSR-VTT 标注样本**微调即可达到原 SOTA 水平。
- **LSMDC 电影片段检索（Table 7）**：零样本 `R@10=14.0`；仅 LSMDC 训练 `R@1=7.2, R@5=18.3, R@10=25.0, Median R=44`；**`PT: HowTo100M + FT: LSMDC`** 升至 `R@1=7.1, R@5=19.6, R@10=27.9, Median R=40`。

## 与其他工作对比

- 相较于 **MSR-VTT / YouCook2 / ActivityNet Captions / How2** 等人工或半自动收集的 `10k–185k` 级数据集，HowTo100M 在片段数（`136.6M`）和总时长（`134,472 h`）上高出 **3 个数量级**。
- 相较于后续端到端双塔视频对比学习（如 MIL-NCE、VideoCLIP），本文受限于 2019 年算力采用**冻结预提取 2D/3D 特征 + 浅层门控头（单卡 3 天可训完）**，未直接解决长旁白与画面间的时间偏移对齐问题。

## 结论

**HowTo100M 证明了“大规模弱对齐互联网教学视频-ASR 旁白对”完全可以替代昂贵人工标注来学习强泛化跨模态视频-文本表征，且在 122 万视频规模下仍未见性能饱和。**

1. **零样本与少样本杠杆显著**：在 CrossTask 上零样本超过全监督上界（`33.6% vs 31.6%`），在通用视频 MSR-VTT 上仅用 `20%` 微调标注即可追平全量训练 SOTA。
2. **同视频负采样至关重要**：引入 50% 同视频内部负样本抑制了环境背景捷径，使细粒度教学任务（YouCook2/CrossTask）指标提升约 7–8 个百分点。
3. **特征复用性价比高**：基于预提取 ResNet-152 + ResNeXt-101 特征在单张 P100 上不到 3 天即可完成亿级片段对齐训练。
4. **领域与特征边界**：数据集高度偏向生活手工与烹饪类教学（三类合计占 >80% 片段），迁移至电影等远域视频仍需下游微调；且受限于 51% 视觉弱对齐率，需大批量与强排序/对比损失配合。

## 常见误区

1. **误以为 1.36 亿字幕与视频画面强对齐**：人工抽检表明仅 **51%** 片段-字幕对在视觉上出现对应动作或物体，其余常存在旁白提前/滞后或口语寒暄噪声，不能直接当作强监督定位标签使用。
2. **忽略模型骨干本身是冻结预提取特征**：论文中 67M 参数的训练仅优化投影门控层与 1D-CNN 文本分支，视觉特征来自固定的 ImageNet ResNet-152 与 Kinetics ResNeXt-101，端到端微调需另行加载原视频。
3. **跨远域直接零样本使用**：HowTo100M 以教学与 vlog 为主，在通用 YouTube（MSR-VTT）与电影（LSMDC）上零样本表现低于域内监督训练，需经下游微调才能发挥最大预训练增益。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_ref_887c80067c952c4655d9_howto100m-learning-a-text-video-embeddin.md`](../../sources/papers/rcl_awesome_wam_ref_887c80067c952c4655d9_howto100m-learning-a-text-video-embeddin.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://openaccess.thecvf.com/content_ICCV_2019/papers/Miech_HowTo100M_Learning_a_Text-Video_Embedding_by_Watching_Hundred_Million_Narrated_ICCV_2019_paper.pdf>

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://openaccess.thecvf.com/content_ICCV_2019/papers/Miech_HowTo100M_Learning_a_Text-Video_Embedding_by_Watching_Hundred_Million_Narrated_ICCV_2019_paper.pdf)
