---
type: entity
tags: [paper, humanoid, benchmark, dataset, motion-evaluation, human-likeness, smpl-x, xmu, oppo, shanghaitech, humanoid-paper-notebooks]
status: complete
updated: 2026-07-28
arxiv: "2603.06181"
venue: "CVPR 2026"
related:
  - ./paper-notebook-robostriker.md
  - ../concepts/whole-body-tracking-pipeline.md
  - ../concepts/motion-retargeting.md
  - ../methods/motion-retargeting-gmr.md
  - ./dataset-bfm-motion-xpp.md
  - ./dataset-bfm-phuma.md
  - ../overview/paper-notebook-category-11-simulation-benchmark.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_towards-motion-turing-test.md
  - ../../sources/sites/motion-turing-test.md
summary: "Motion Turing Test（CVPR 2026）用 1,000 个统一 SMPL-X 的人/机器人动作与 0–5 人工类人度评分建立 HHMotion，并以 PTR-Net 回归人类判断；跳跃、拳击、跑步差距最大，代码和数据截至 2026-07-28 尚无下载入口。"
---

# Towards Motion Turing Test：量化人形动作类人度

**Towards Motion Turing Test: Evaluating Human-Likeness in Humanoid Robots**（[arXiv:2603.06181](https://arxiv.org/abs/2603.06181)，CVPR 2026）由厦门大学、OPPO 研究院与上海科技大学提出。

## 一句话定义

**把 500 段人形机器人动作与 500 段人类动作统一成去外观的 SMPL-X 序列，让人类按 0–5 分评价“像不像人”，再训练 PTR-Net 近似这把类人度标尺。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HHMotion | Human-Humanoid Motion Dataset | 1,000 段人类与人形动作类人度数据集 |
| PTR-Net | Pose-Temporal Regression Network | 从姿态时序回归类人度的基线 |
| SMPL-X | Skinned Multi-Person Linear Model eXpressive | 消除机器人外观差异的统一人体表示 |
| ST-GCN | Spatial-Temporal Graph Convolutional Network | 提取跨关节与跨帧协调模式 |
| IAC | Inter-Annotator Consistency | 剔除不一致评分者的质量控制 |
| MAE | Mean Absolute Error | 类人度预测与人工分数的平均绝对误差 |

## 为什么重要

- **把“像人”变成可回归指标：** 任务成功和不摔倒无法描述节奏、姿态与全身协调的细微机械感。
- **隔离外观偏差：** 所有视频重建为同一种 SMPL-X 形体，评分者主要看运动学而非机器人造型。
- **揭示任务优先级：** 动态行为的差距显著大于站立、步行，可直接指导拳击、跑酷和动作跟踪的评测设计。
- **潜在 reward model：** PTR-Net 可作为动作生成或 RL 的外部类人度评分，但须防止 reward hacking。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 厦门大学；OPPO 研究院；上海科技大学 |
| **数据** | 1,000 个 5 秒片段；500 human + 500 humanoid；15 类动作、11 种机器人、10 名人类受试者 |
| **原始规模** | 来自真实赛事、仿真、志愿者、模仿机器人动作和 YouTube 的 21.7 h 原视频 |
| **标注** | 30 人、每人 1,000 段、累计 500+ 小时；IAC 后保留 25 人 |
| **开源** | **待发布**（截至 2026-07-28）：[项目页](http://www.lidarhumanmotion.net/mtt/)未列代码、数据或 benchmark 下载链接 |

## 数据与评测流程

```mermaid
flowchart LR
  raw["人类 / 真机 / 仿真视频"] --> clip["统一 5 秒片段"]
  clip --> gvh["GVHMR 姿态估计"]
  gvh --> smpl["统一 SMPL-X<br/>去外观线索"]
  smpl --> qc["人工重建质检"]
  qc --> score["30 人 0–5 Likert 评分"]
  score --> iac["IAC 剔除 5 位不一致者"]
  iac --> labels["HHMotion 平均分标签"]
  labels --> ptr["PTR-Net 训练与 benchmark"]
```

## 核心机制（方法栈）

### 1）Motion Turing Test 协议

评分者只看 SMPL-X pose sequence，按姿态、节奏和协调性判断：0 表示完全机械，5 表示与人类动作不可区分。人类和机器人片段随机混排，降低先验标签偏差。

### 2）HHMotion 数据构建

机器人动作来自 WRC、WAIC、WHRG 与 LAFAN1 retarget 仿真；人类侧含 10 名受试者、网络视频和专门模仿机器人动作的 hard subset。通用 HPE 方法 GVHMR 同时重建人和机器人，失败/噪声样本经人工交叉检查。

### 3）PTR-Net

两层双向 LSTM 编码长时依赖，ST-GCN 建模关节图的空间—时间协调，attention pooling 聚焦关键片段，MLP 输出 \([0,5]\) 标量。训练目标为 L2 分数回归加平滑正则。

## 源码运行时序图

**不适用。** 论文和项目页宣称将发布 dataset、code 与 benchmark，但截至 2026-07-28 没有公开仓库或下载入口，无法验证数据预处理与训练命令。

## 工程实践

| 环节 | 建议 | 风险检查 |
|------|------|----------|
| 视频采样 | 每动作平衡人/机器人来源，保留真机/仿真标签 | 来源版权、机器人型号泄漏 |
| 姿态统一 | 固定 SMPL-X 骨架、root frame 与帧率 | 重建错误被误评为机器人不自然 |
| 人工评分 | 随机混排、盲测、IAC 与标注者校准 | 文化偏差、疲劳、同一人重复性 |
| 模型评测 | 同时报 MAE、RMSE、Spearman \(\rho\) | 只优化均值却丢失排序 |
| RL 使用 | 先离线验证再小权重并入 reward | 对 PTR-Net 漏洞做对抗性动作 |

## 与其他工作对比

| 评测 | 输入 | 监督 | 回答的问题 |
|------|------|------|------------|
| Motion Turing Test | 统一 SMPL-X 时序 | 人类 0–5 类人度 | 动作看起来像不像人 |
| 任务成功率 | 环境状态 / 接触 | 成功条件 | 是否完成任务 |
| 轨迹跟踪误差 | robot vs reference | 参考动作 | 是否贴近指定示范 |
| VLM 视频打分 | RGB 视频 | 通用预训练 | 语义解释强，但细粒度运动判断弱 |

## 实验与评测

- PTR-Net 达到 **MAE 0.5813、RMSE 0.7926、Spearman \(\rho=0.6841\)**，优于 Gemini 2.5 Pro、Qwen3-VL-Plus、MotionBERT 与轻量 Transformer 基线。
- 去 Temporal Encoder 后 MAE / \(\rho\) 退化到 **0.7631 / 0.3610**；去 attention 后为 **0.6185 / 0.6255**，说明时序建模是主要增益。
- 最大人—机分差来自 jump **3.23**、boxing **2.53**、run **2.26**；walk 分差 **1.31**，说明“会走”不等于动态动作已类人。
- 未见机器人 XPeng IRON 上 PTR-Net 预测 **4.25**，人工均分 **4.36**，但单个 OOD 案例不能替代跨平台统计。

## 结论

**Motion Turing Test 的价值是把类人度从演示观感拆成数据、标注协议和可训练基线，但它仍是主观运动学指标，不是安全或任务能力总分。**

1. **动态任务应优先补课** — 跳跃、拳击和跑步的人机差距最大。
2. **时序协调比单帧姿态更关键** — PTR-Net 的 temporal encoder 消融跌幅最大。
3. **SMPL-X 降低外观偏见但引入重建偏差** — benchmark 必须保留 pose QC。
4. **应与任务指标联合使用** — 高类人度不保证成功率、能耗或接触安全。
5. **公开状态仍是复现阻塞点** — 没有下载入口前，数字只能由论文核读，不能跑通。

## 局限与风险

- 评分来自有限标注群体；“类人”的文化与经验偏好未必跨群体稳定。
- 5 秒片段不覆盖长程策略、恢复与动作转换；SMPL-X 也忽略接触力、力矩与真实机体形态。
- 网络视频和赛事视频可能带来选择偏差；同一机器人不同控制器未必能被公平归因。
- 作为 RL reward 时可能奖励“骗过回归器”而非真正自然、安全的运动。

## 与其他页面的关系

- 应用锚点：[RoboStriker](./paper-notebook-robostriker.md) — 拳击是人机类人度差距最大的动作之一
- 表示前置：[Motion Retargeting](../concepts/motion-retargeting.md)、[GMR](../methods/motion-retargeting-gmr.md)
- 数据对照：[Motion-X++](./dataset-bfm-motion-xpp.md)、[PHUMA](./dataset-bfm-phuma.md)
- 评测选型：[具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 本文属于人体判断对齐的动作质量评测层
- 路线入口：[人形拳击纵深](../../roadmap/depth-humanoid-boxing.md)

## 参考来源

- [论文与深读笔记归档](../../sources/papers/humanoid_pnb_towards-motion-turing-test.md)
- [项目页与开放状态核查](../../sources/sites/motion-turing-test.md)
- 论文：<https://arxiv.org/abs/2603.06181>

## 推荐继续阅读

- [Motion Turing Test 官方项目页](http://www.lidarhumanmotion.net/mtt/)
- [机器人论文阅读笔记：Towards Motion Turing Test](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/Towards_Motion_Turing_Test__Evaluating_Human-Likeness_in_Humanoid_Robots/Towards_Motion_Turing_Test__Evaluating_Human-Likeness_in_Humanoid_Robots.html)
