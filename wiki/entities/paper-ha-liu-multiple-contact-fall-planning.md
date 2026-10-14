---
type: entity
project_id: ha-liu-multiple-contact-fall-planning-2015
tags: [paper, humanoid, fall-safety, contact-planning, gatech]
status: complete
updated: 2026-10-14
doi: "10.1109/IROS.2015.7353756"
venue: "IEEE/RSJ IROS 2015"
project: https://faculty.cc.gatech.edu/~sha9/projects/ha2015mcs/index.html
related:
  - ../tasks/balance-recovery.md
  - ../concepts/contact-dynamics.md
  - ./paper-notebook-unified-humanoid-fall-safety-policy-from-a-few-d.md
sources:
  - ../../sources/papers/ha_liu_2015_multiple_contact_humanoid_fall_planning.md
  - ../../sources/sites/ha2015mcs-gatech.md
summary: "把人形跌倒规划成倒立摆子问题链，用动态规划选择多接触序列，在仿真与 BioloidGP 上验证。"
---

# Multiple Contact Planning for Minimizing Damage of Humanoid Falls

**一句话定义：** Ha 与 Liu 的 IROS 2015 工作把失稳后的人形跌倒建模为一串倒立摆问题，再用动态规划寻找多接触响应序列，让初始动量通过后续接触逐步耗散、减少冲击。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IROS | IEEE/RSJ International Conference on Intelligent Robots and Systems | 本文发表的机器人学会议 |
| DP | Dynamic Programming | 在分阶段状态/决策结构上求解最优策略的动态规划方法 |
| CoM | Center of Mass | 质心；机器人整体平衡与倒立摆近似的关键状态 |
| DoF | Degrees of Freedom | 自由度；描述机器人可独立运动的关节或广义坐标数 |

## 为什么重要

平衡恢复方法通常试图让机器人重新站稳，但高速失稳时直立恢复未必可行。本文把目标转到**安全地接受不可避免的跌倒**：主动安排接触位置和先后顺序，以渐进方式卸掉动量。它让多接触从“临近倒地时临时扶墙/扶地”的概念，变成一个可规划的响应序列。

论文摘要给出的范围从轻微推扰后的单步恢复延伸到高速跌倒时的滚动动作。项目页称方法通过物理仿真验证，并在 BioloidGP 人形机器人上实验。

## 方法栈与流程总览

```mermaid
flowchart LR
  A["失稳初始状态"] --> B["连续倒立摆子问题"]
  B --> C["动态规划选择接触序列"]
  C --> D["接触逐步耗散动量"]
  D --> E["仿真与 BioloidGP 验证"]
```

图只表示官方项目摘要确认的高层流程；具体模型方程和控制执行细节需查完整论文。

## 方法

### 多接触规划

输入是机器人当前的失稳状态。算法把跌倒过程抽象为一串倒立摆问题，并用动态规划选择后续接触点的序列。不同的初始动量和失稳程度可以导向不同响应：轻微扰动可通过一步恢复，高速跌倒则可用滚动逐次接触来降低冲击。

核心视角是把接触时序作为规划变量。机器人不必在一组固定的“恢复策略”之间切换，而是根据状态生成适当的接触响应。项目页摘要没有公开状态离散化、接触集合、代价函数权重或完整动作控制器，因此不能仅凭摘要复现算法。

### 优化目标的证据边界

官方摘要描述的是以最小冲击耗散初始动量。用户提供的截图将该工作概括为**最小化单次最大冲击力**，并称多次接触可以分散受力；这是与摘要相容的二手解释，但当前可访问的官方项目页没有展示目标函数公式。精确复现目标时应核对完整论文，而不要把截图文案当成已确认的数学定义。

## 实验与评测

- **项目页报告：** 物理仿真验证，并在 BioloidGP 人形机器人上进行了实验测试。
- **可核实的结果粒度：** 当前项目页摘要没有给出样本数、冲击力降幅、对照组或统计显著性，因此不补造定量结论。
- **完整论文入口：** 官方页所列 PDF 直链在 2026-10-14 返回 404；书目信息可由 [IEEE DOI](https://doi.org/10.1109/IROS.2015.7353756) 交叉确认。

## 结论

**一句话总判：** 这项工作的关键是把“摔倒时怎么接触”变成可规划的序列问题；它证明了多接触规划可以从仿真走向小型人形平台，但公开摘要不足以支撑精确复现或量化性能比较。

1. **区分防摔与减损。** 当恢复直立已不可行时，规划安全的接触序列可能比继续追求平衡更合适。
2. **接触顺序是控制自由度。** 规划目标不是单纯选一个落地点，而是让后续接触共同耗散初始动量。
3. **将“峰值冲击最小化”视为待核查细节。** 用户截图这样概括；项目页摘要只确认减少冲击，没有可访问的代价函数公式。
4. **仿真 + 实机验证不等于广泛安全保证。** 摘要只确认物理仿真与 BioloidGP 实验，未给出跨机器人、跨跌倒方向或耐久性证据。
5. **复现前先获取全文。** 目前官方 PDF 链接失效，状态空间、接触模型、动态规划复杂度和低层控制器仍需核对。

## 源码运行时序图

**不适用：** 官方项目页未列可运行代码仓库或训练 / 推理入口，因此没有可对齐的源码运行时序。页面虽提供 PDF 和视频入口，但这不是可执行实现；当前直链也返回 404。

## 工程实践

- **状态估计：** 部署前需检查失稳状态、角动量和接触状态是否能以足够频率、精度估计。
- **模型边界：** 倒立摆链是用于规划的简化模型。接触顺序需要经过包含实际连杆几何、摩擦、结构柔顺与电机限幅的全身仿真验证。
- **安全验证：** 分别评估接触点可达性、单次峰值冲击、总能量耗散、执行器/传感器脆弱部件受力、恢复可行性与硬件耐久；不能只看是否最终停住。
- **源码状态：** 官方项目页没有公开运行仓库或数据集链接；官方 PDF 和 MP4 链接当前失效。此状态记录的是可见资源，不推断作者没有其他代码。

## 局限与风险

- **公开证据有限：** 官方摘要确认方法骨架与仿真/实机验证，但全文资源当前不可访问，无法确认公式、参数、基线和量化提升。
- **简化模型迁移：** 多阶段倒立摆规划的有效性取决于接触模型与真实全身动力学之间的误差；不同机器人质量分布、关节限位和外壳可能改变可行动作。
- **接触与硬件风险：** 低峰值不必然意味着低总损伤；腕部、摄像头、连接件等部位的局部脆弱性和重复跌倒寿命仍需单独测试。
- **适用范围待核查：** 摘要提到多种跌倒响应，但没有给出方向覆盖、障碍环境能力或在线规划时延的细节。

## 与其他工作对比

| 工作路线 | 决策对象 | 主要证据/侧重 | 阅读时的边界 |
|---|---|---|---|
| Ha & Liu（本文） | 失稳后接触点及接触顺序 | 倒立摆链 + DP；摘要报告物理仿真和 BioloidGP | 精确代价函数与性能数字需全文核验 |
| SafeFall | 跌倒预警后的保护控制动作 | 学习式保护策略，见 [SafeFall 实体页](./paper-hrl-stack-41-safefall.md) | 与解析/模型驱动接触序列规划不同 |
| Unified Humanoid Fall-Safety Policy | 防摔、减损、起身的统一策略 | 少量示范 + RL + 自适应扩散记忆；见[实体页](./paper-notebook-unified-humanoid-fall-safety-policy-from-a-few-d.md) | 目标覆盖全跌倒过程，方法栈和评测对象更广 |

## 关联页面

- [Balance Recovery（平衡恢复）](../tasks/balance-recovery.md) — 含多接触恢复策略分类；本文补充不可避免跌倒时的接触序列规划。
- [接触动力学](../concepts/contact-dynamics.md) — 接触冲击、摩擦和多接触建模背景。
- [Unified Humanoid Fall-Safety Policy](./paper-notebook-unified-humanoid-fall-safety-policy-from-a-few-d.md) — 学习式防摔、减损与起身统一策略。
- [SafeFall](./paper-hrl-stack-41-safefall.md) — 学习保护控制的对照路线。

## 参考来源

- [论文摘录与证据整理](../../sources/papers/ha_liu_2015_multiple_contact_humanoid_fall_planning.md)
- [Georgia Tech 官方项目页](../../sources/sites/ha2015mcs-gatech.md)
- [IEEE DOI 10.1109/IROS.2015.7353756](https://doi.org/10.1109/IROS.2015.7353756)
- [IROS 2015 会议文摘](https://www.iros2015.org/docs/IROS_Digest_WWW.pdf)

## 推荐继续阅读

- [Georgia Tech 项目页](https://faculty.cc.gatech.edu/~sha9/projects/ha2015mcs/index.html) — 摘要与资源入口。
- [IEEE 论文记录](https://doi.org/10.1109/IROS.2015.7353756) — 获取完整论文或核对出版信息。
- [IEEE Spectrum 对演示与方法边界的报道](https://spectrum.ieee.org/robots-judo-techniques-to-fall-down-without-exploding) — 二手背景报道，阅读时与论文证据区分。
