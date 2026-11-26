---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-11-26
project_id: aff-bootstrapping-relational-affordances-of-object-pairs-usi
doi: "10.1109/TCDS.2016.2616496"
venue: "TCDS 2016"
summary: "IEEE TCDS 发育机器人双物体关系 affordance 迁移学习论文：基于 150 万次动作仿真，通过直接自举（DB）与相关性剪枝的类别自举（CB）扩展随机森林状态空间，在 PCA 空间以 25 个样本达到无自举 1000 样本精度（CBn 平均自举因子 2.895）并验证 Sim-to-Real 迁移。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_bootstrapping-relational-affordances-of-object-pairs-usi.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Bootstrapping Relational Affordances of Object Pairs Using Transfer

**Bootstrapping Relational Affordances of Object Pairs Using Transfer**（IEEE TCDS 2016/2018，DOI 10.1109/TCDS.2016.2616496，University of Southern Denmark & University of Aberdeen）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套文献。本页基于论文全文完成结构化精读与核实。

## 一句话定义

一项面向发育机器人的双物体关系型 affordance 迁移学习研究，通过将已学动作 affordance 预测器（**Direct Bootstrapping, DB**）或基于预测相关性自动生成的空间关系类别分类器（**Category-based Bootstrapping, CB**）的输出扩展进随机森林状态空间，实现少样本条件下手段–目的（means-end）操作关系的快速自举学习。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| NB / DB / CB | No / Direct / Category-based Bootstrapping | 无自举基线 / 直接复用已有预测器 / 复用抽象关系类别预测器 |
| PCA | Principal Component Analysis | 主成分分析，本文中构建 18 维双物体质心、姿态角与伸长率基础状态空间 |
| RHF | Relational Histogram Features | 关系直方图特征，300 维 1D 点对距离角度特征（与 PCA 合并为 318 维） |
| RF / CART | Random Forest / Classification & Regression Trees | 随机森林与基分类回归树，负责特征选择与多模态数值/类别输入分类 |
| $bf$ | Bootstrap Factor | 自举加速因子 $\frac{P_{bl}-P_c}{P_{sl}-P_c}$，量化相对无自举基线超越随机猜测的倍数 |

## 为什么重要

- 发育机器人通过自身交互试错学习空间关系（如 `on top`、`inside`、`beside`）及其对应的手段–目的操作（如拉动托盘取其上杯子）成本极高，而手工为每项新技能设计高维特化几何特征（如 RHF）又违背终身自主学习初衷。
- 本文基于约 **150 万次**动作仿真系统验证了正迁移与负迁移边界，证明在简洁基础的 18 维 PCA 空间中引入自举，仅凭 **25 个样本**即可达到无自举 **1000 个样本**才能达到的精度，且无需人工预设即可涌现出“在上方/在旁边”等空间概念类别。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | IEEE Transactions on Cognitive and Developmental Systems（Vol. 10, No. 1, pp. 56–71, 2018；Early Access 2016） |
| 作者与机构 | Severin Fichtl, Dirk Kraft, Norbert Krüger, Frank Guerin（SDU, University of Aberdeen） |
| 清单分组 1 | Reasoning · Reasoning Subcategory: Relation-Based Affordance Reasoning |
| 论文 | <https://doi.org/10.1109/TCDS.2016.2616496> |

## 核心机制

1. **双物体点云感知与两套状态空间**
   - 从模拟 Kinect（640×480）采样点云经 0.0125 m 体素下采样至 40k–50k 点并按颜色分割双物体：
     - **PCA 状态空间（18 维）**：每个物体提取 9 个特征（质心 XYZ、特征向量 Roll/Pitch/Yaw、特征值伸长率），双物体共 18 维；
     - **RHF 状态空间（318 维）**：在 18 维 PCA 基础上叠加 300 维 1D 点对距离与法线角度关系直方图（RHF）。
2. **九类手段–目的动作与大规模仿真采样**
   - 在 6-DoF 机械臂（带二指夹爪）与 29 个物体（5 Toys、15 Bases、5 Obstacles、5 Rakes）上运行约 **150 万次**动作仿真（约 900 小时集群算力），为 `Lift`、`Move`、`Pull`、`Push`、`Rake`、`Take`、`Pour`、`Slide`、`Unobstruct` 共 9 个动作各收集约 10,000 个正负均衡样本（负样本匹配质心距离分布，避免模型靠质心距作弊）。
3. **直接自举（DB）、类别自举（CB）与启发式剪枝**
   - **随机森林基学习器**：树数 $T=95+N/20$（上限 115 棵），最大深度 200；为确保随机森林分裂节点有足够概率选中二进制自举特征，将每个自举源输出在 PCA 空间复制 6 次（至 24 维）、在 RHF 空间复制 30 次（至 348 维）。
   - **Direct Bootstrapping（DB1 / DBn）**：直接拼接 1 个或全部 8 个已学动作预测器的输出。
   - **Category-based Bootstrapping（CB1 / CBn）**：两两合并已有动作训练集训练“类别预测器”（$\binom{9}{2}=36$ 个候选），自发抽象出对应 `on top/inside`（`Lift/Pull/Push` 强相关簇）与 `beside`（`Move/Take/UnOb/Rake` 弱相关簇）的二值类别。通过**预测相关性阈值 $T$ 剪枝**：当 $T=0.5$ 时剔除约 **3/4** 低相关候选而几乎无损性能；另提出**前 10 样本并行筛选（best after ten）**自动选定最优单一自举源。

### 方法流程图

```mermaid
flowchart LR
    A["RGB-D 场景点云 (体素下采样 + 颜色分割)"] --> B["特征提取: 18维 PCA 或 318维 RHF"]
    C["已有动作预测器库 (9 类动作)"] --> D["相关性统计与剪枝 (阈值 T=0.5 保留约 1/4)"]
    D --> E["训练二元类别预测器 (如 on top / beside 抽象)"]
    B --> F["状态空间扩展 (DB: 拼接动作输出 / CB: 拼接类别输出)"]
    C --> F
    E --> F
    F --> G["随机森林分类器并行训练 (T<=115 棵 CART 树)"]
    G --> H["10 样本在线择优 (best after ten) 或全量 CBn 预测"]
    H --> I["预测目标手段-目的动作能否成功"]
```

## 评测与指标

在 PCA 状态空间（4–25 样本早期区间，50 次重复、每次 4000 测试样本）上的平均 **Bootstrap Factor（$bf$ ↑，Table II）** 及真机迁移结果：

| 学习策略 | 平均 $bf$ ↑ | 最小 $bf$ | 最大 $bf$ | 说明 |
|------|---|---|---|---|
| `NB`（无自举基线） | 1.000 | 1.000 | 1.000 | 仅使用 18 维 PCA 基础特征 |
| `DB1 average` | 1.356 | 0.8055 | 1.582 | 随机选 1 个已有动作预测器（部分场景负迁移） |
| `DB1 best` | 2.819 | 0.9760 | 3.657 | 事后最优单动作自举（25 样本匹敌无自举 1000 样本） |
| `DB1 best after ten` | 2.751 | 0.8756 | 3.657 | 仅前 10 样本并行择优即接近理论最优 |
| `DBn` | 2.357 | 1.064 | 3.090 | 拼接全部 8 个动作预测器（受维度灾难拖累低于 DB1 best） |
| `CB1 average` | 1.548 | 0.7860 | 1.913 | 随机选 1 个双动作抽象类别预测器 |
| `CB1 best` | 2.889 | 0.8503 | 3.701 | 最优单类别自举 |
| `CB1 best after ten` | 2.802 | 0.7702 | 3.752 | 10 样本自动择优类别 |
| **`CBn`（$T=0.5$ 剪枝后全类别）** | **2.895** | 0.9072 | **3.888** | 抽象类别泛化性优于直接拼接全动作的 DBn |

- **特征空间与 Sim-to-Real 对比**：在已高度工程化的 318 维 RHF 空间中自举收益接近 1.0（已接近饱和，甚至因输入冗余拖慢 `Take` 学习）；真机使用 **Carmine 1.09** 传感器采集 160 样本（100 训 / 60 测），`SIM_full`（6500 样本训练）直接测试真机可稳定区分 `beside` 与 `on top`，将其作为真机自举源（`REAL<-SIM`）在少样本阶段同样带来显著正加速。

## 与其他工作对比

- **对比直接设计强特征（RHF）**：手写 318 维 RHF 虽在非自举情况下远胜 18 维 PCA，但计算与存储开销高且可能因维度膨胀在 `Take` 上产生负效应；PCA + 自举在 25 样本内即可追平 RHF。
- **对比早期单任务自举工作（如 Do et al.、Ugur et al.）**：前者多局限于单对动作（如 `mixing→wiping` 或 `rollable→stackable`），本文在 9 类动作、36 个候选类别上系统揭示了正/负迁移规律与相关性剪枝策略。同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。

## 常见误区或局限

1. **视觉分割与场景假设简化**：实验依赖各物体拥有独立已知纯色（真机通过 RANSAC 去桌面 + k-means 聚三类），尚未处理真实杂乱纹理与多物体遮挡分割问题。
2. **存在负迁移与运动程序噪声**：并非任意已有 affordance 都能加速新任务（`DB1 average` 最低跌至 0.8055），如 `Take` 与 `inside` 相关性弱且无路径规划逆运动学本身抓取失败率高，噪声特征或无关特征会拉低随机森林性能。
3. **DB 与 CB 在底层学习速度上差异不显著**：实验表明 DB 与 CB 加速效果相近，CB 的核心长期价值在于无需人工设计即可沉淀出可供高层符号规划器调用的关系谓词（如 `on top`、`beside`）。
4. **物理深度噪声限制**：仿真深度模型未模拟真实红外相机的强环境光与镜面反射失效场景；论文未提供公开源码仓库链接。

## 结论

**本文证明在简单几何状态空间中，复用已有动作或相关性剪枝后的抽象空间类别预测器，可将双物体关系 affordance 学习样本需求从千级压缩至数十级，并能从仿真正向自举真机感知。**

- 若特征空间本身尚未高度特化，优先采用 **10 样本并行择优（best after ten）** 的单源直接自举（DB1）或相关性阈值 $T=0.5$ 剪枝后的类别集合（CBn），避免全量特征拼接（DBn）引发维度灾难。
- 在开发发育机器人或分层规划架构时，可利用两两动作预测相关性自动发现物理因果簇（`on top` vs `beside`），作为无监督符号抽象起点。
- 跨仿真到真实传感器迁移时，即使存在领域分布落差，将仿真训练分类器作为特征自举先验（`REAL<-SIM`）仍比真机从零学习更具样本效率。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/affordance_awesome_bootstrapping-relational-affordances-of-object-pairs-usi.md`](../../sources/papers/affordance_awesome_bootstrapping-relational-affordances-of-object-pairs-usi.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://doi.org/10.1109/TCDS.2016.2616496>

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://doi.org/10.1109/TCDS.2016.2616496)
