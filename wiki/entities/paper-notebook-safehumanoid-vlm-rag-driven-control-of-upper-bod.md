---
type: entity
tags: [paper, humanoid-paper-notebooks, humanoid, manipulation, safety, impedance-control, vlm, rag, hri, unitree]
status: complete
updated: 2026-09-28
arxiv: "2511.23300"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../concepts/impedance-control.md
  - ../concepts/safety-filter.md
  - ../concepts/retrieval-augmented-generation.md
  - ./paper-notebook-humanoidvlm-vision-language-guided-impedance-con.md
  - ../concepts/llm-robotics-control-interfaces.md
sources:
  - ../../sources/papers/humanoid_pnb_safehumanoid.md
summary: "SafeHumanoid 把「怎么调阻抗」这个低层控制问题，交给一个第一视角 VLM + RAG 检索库来回答：头部相机画面 → VLM 抽成结构化场景语义（任务、物体易碎性、是否有人、障碍等）→ 在 16 条经安全标准验证的模板库里做最近邻检索 → 取回每关节的刚度 Kp / 阻尼 Kd / 速度，下发给 50 Hz 的板载阻抗控制器。一旦画面里出现人手，机器人自动降刚度、升阻尼、减速，在不丢任务的前提下提升人机协作安全性。"
---

# SafeHumanoid

**SafeHumanoid: VLM-RAG-driven Control of Upper Body Impedance for Humanoid Robot** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页编译自深读笔记与 arXiv 论文正文（实验数字、开源状态于 2026-09-28 核对），细节以论文 PDF 为准。

## 一句话定义

SafeHumanoid 把「怎么调阻抗」这个低层控制问题，交给一个第一视角 VLM + RAG 检索库来回答：头部相机画面 → VLM 抽成结构化场景语义（任务、物体易碎性、是否有人、障碍等）→ 在 16 条经安全标准验证的模板库里做最近邻检索 → 取回每关节的刚度 Kp / 阻尼 Kd / 速度，下发给 50 Hz 的板载阻抗控制器。一旦画面里出现人手，机器人自动降刚度、升阻尼、减速，在不丢任务的前提下提升人机协作安全性。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| VLM | Vision-Language Model | 视觉语言模型，本文用 Molmo-7B 把画面转成结构化语义 |
| RAG | Retrieval-Augmented Generation | 检索增强生成，这里是「查模板库取参数」 |
| Impedance | Impedance Control | 阻抗控制，用刚度 Kp / 阻尼 Kd 调节交互柔顺度 |
| IK | Inverse Kinematics | 逆运动学，把 6-DoF 目标位姿解成关节参考角 |
| FAISS | Facebook AI Similarity Search | 向量近邻检索库，做语义模板匹配 |
| HRI | Human-Robot Interaction | 人机交互 |
| ISO/TS 15066 | — | 协作机器人人机协作安全技术规范 |

## 为什么重要

- **人机协作安全**：提供一条「语义 → 阻抗」的可落地路径，把安全标准嵌进控制参数选择
- **VLM 接低层控制**：不让 LLM 直接吐扭矩，而是经检索库做中介，工程上更稳、更可控
- **模板库范式**：用少量人工验证模板 + 近邻检索，权衡了「可解释/合规」与「自动化」
- **延迟瓶颈**：也暴露了离板大模型 + 低频感知在动态 HRI 下的实时性天花板

## 解决什么问题

人形机器人和人**共享同一工作空间**做桌面操作（擦桌、递物、倒液体）时，安全的核心矛盾是：

1. **刚度高 = 精度好但危险**：硬碰到人手会产生大的接触力。 2. **刚度低 = 安全但任务做不好**：太软抓不稳、对不准。 3. 传统做法的阻抗参数是**固定 / 手调**的，无法随「现在面前是不是有人、物体易不易碎」自动变化。

## 核心机制

1. **语义驱动阻抗的闭环系统**：第一个把「VLM 看场景 → RAG 取参数 → 关节阻抗」串成在线闭环、并跑在真 G1 上的工作。
2. **安全标准入库**：模板库里的每条阻抗配置都按 ISO/TS 15066、ISO 13855 实测验证，把「合规」前移到检索库设计阶段。
3. **结构化检索而非自由生成**：用固定提示词 + FAISS 最近邻，换来确定、可复现、可兜底的参数输出，规避 LLM 端到端生成控制量的不稳定。
4. **一定泛化能力**：库里没有的物体（如 pin 销钉）也能被 VLM-RAG 归到一个合适的柔顺配置。

方法拆解（深读笔记小节）：第一视角感知（1–2 Hz）；VLM 抽语义（Molmo-7B）；RAG 两阶段检索；关节级阻抗执行（50 Hz，板载 Jetson Orin NX）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/SafeHumanoid__VLM-RAG-driven_Control_of_Upper_Body_Impedance/SafeHumanoid__VLM-RAG-driven_Control_of_Upper_Body_Impedance.html> |
| arXiv | <https://arxiv.org/abs/2511.23300> |
| 发表 | 2025-11-28 (arXiv) |
| 源码 | **未开源**：论文未公开代码 / 项目页（正文链接的 socraticmodels.github.io 为引用的相关工作），截至 2026-09-28 未见仓库 |
| 笔记阅读日期 | 2026-06-22 |

## 实验与评测


**设置**：Unitree G1，头部 RealSense；机载 Jetson Orin NX 以 50 Hz 执行 IK 与关节阻抗控制，VLM–RAG 在外部 RTX 4090 服务器经 TCP/IP 通信。Molmo-7B 以固定 JSON 模式输出任务、主物体、易碎性、是否有人、障碍类型、工作区状态与置信度；FAISS 从场景库中检索整行参数（14 个关节的 Kp / Kd 共 28 个增益 + 名义速度）。

**场景库**：先在 G1 上做试点实验（取方块、倒瓶、擦桌、递物，分有人 / 无人），记录第一视角视频、关节状态、力矩与 PD 增益；逐条做稳定性筛查，并对照 ISO/TS 15066、ISO 13855 限制易碎物刚度与有人时的速度，最终只保留 **16 个**验证过的场景（CSV，16 行 × 34 列）。

**实验**：6 个桌面操作任务，每个任务先自主运行、再在工作区放入人手重复一次。

- 所有任务中系统都按任务与人的在场情况调整了 Kp、Kd 与速度，任务均完成：擦桌时人手进入 → 立即降 Kp、升 Kd、减速，离开后恢复基线。
- **库外物体**：「销钉」不在场景库中，递交时仍被归到合适的柔顺配置；「方块」在库中，有人接触时正确切到低刚度并在人离开后恢复。
- **液体处理**：递交并倒酱油瓶时，检测到液体操作后把名义速度降到慢速，放下后恢复。
- **延迟**：离板 VLM–RAG 回路延迟最高 **1.4 s**，作者认为不满足动态人机交互。
- 论文 Table 1 只以箭头表示相对基线的升降，未给出力或碰撞等量化安全指标。

## 与其他工作对比

| 做法 | 安全参数的来源 | 与 SafeHumanoid 的差异 |
|------|------|------|
| 固定增益基线 | 手调一组参数 | 不随人的在场与物体易碎性变化 |
| [HumanoidVLM](./paper-notebook-humanoidvlm-vision-language-guided-impedance-con.md) | 同组，检索任务空间阻抗 + 夹爪角 | 关注任务适配的柔顺；SafeHumanoid 关注人在场时的安全调度，并对照 ISO 标准筛选参数 |
| [安全过滤器 / CBF](../concepts/safety-filter.md) | 在线约束求解 | 有形式化保证、实时性高；SafeHumanoid 的安全性取决于场景库覆盖 |
| LLM 直接输出控制量 | 生成式 | 输出不确定；SafeHumanoid 只检索已验证参数 |

## 结论

**SafeHumanoid 的核心取舍是「让 VLM 只做语义判断、不做控制量生成」：语义 → 检索 → 阻抗参数这条中介链，用确定性和合规性换掉了端到端生成的自由度。**

- 真正起作用的是检索而不是生成：固定提示词 + FAISS 最近邻，从 16 条模板里取回每关节的 Kp / Kd / 速度，输出确定、可复现、可兜底，规避了 LLM 直接吐控制量的不稳定。
- 合规被前移到库的设计阶段：每条模板按 ISO/TS 15066、ISO 13855 实测验证，因此安全性上限取决于模板库的覆盖度——库外物体（如 pin 销钉）只能被归到「一个合适的柔顺配置」，属于泛化而非保证。
- 最明确的瓶颈是时间尺度错配：第一视角感知只有 1–2 Hz，而阻抗执行在板载 Jetson Orin NX 上跑 50 Hz；离板大模型 + 低频感知在动态 HRI 下存在实时性天花板。
- 适用边界是人机共享工作空间的桌面级上半身操作（擦桌、递物、倒液体）中的阻抗调节，它回应的是「刚度高则危险、刚度低则做不好任务、固定手调无法随场景变化」这一矛盾，并不解决接触规划或全身安全。
- 落地状态：系统跑在真 Unitree G1 上的 6 个桌面任务中，阻抗与速度调节方向全部正确，但论文只给定性结果、没有力或碰撞测量，且未公开代码 / 项目页。

## 局限与风险

- **延迟**：离板推理 + 图像传输导致最高 1.4 s 延迟，不适合高动态人机交互。
- **场景库人工整理**：16 个场景来自试点实验，需人工验证任务与参数，规模与多样性受限。
- **动作非自主**：实验中 IK 输入是预先指定的末端位姿，系统只调度阻抗与速度，不生成动作。
- **缺少量化安全指标**：结果以「调节方向是否正确」描述，没有接触力、碰撞次数等测量。
- **开源边界**：未见代码；源码运行时序图 **不适用**。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 阻抗控制：[impedance-control](../concepts/impedance-control.md)
- 安全过滤器：[safety-filter](../concepts/safety-filter.md)
- RAG 检索增强：[retrieval-augmented-generation](../concepts/retrieval-augmented-generation.md)
- 同组 VLM + RAG 阻抗检索：[paper-notebook-humanoidvlm-vision-language-guided-impedance-con](./paper-notebook-humanoidvlm-vision-language-guided-impedance-con.md)
- LLM / VLM 与控制器的接口方式：[llm-robotics-control-interfaces](../concepts/llm-robotics-control-interfaces.md)

## 参考来源

- [humanoid_pnb_safehumanoid.md](../../sources/papers/humanoid_pnb_safehumanoid.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/SafeHumanoid__VLM-RAG-driven_Control_of_Upper_Body_Impedance/SafeHumanoid__VLM-RAG-driven_Control_of_Upper_Body_Impedance.html>
- 论文：<https://arxiv.org/abs/2511.23300>
- 论文正文（场景库构建、实验与局限节）：<https://arxiv.org/html/2511.23300>

## 推荐继续阅读

- [机器人论文阅读笔记：SafeHumanoid](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/SafeHumanoid__VLM-RAG-driven_Control_of_Upper_Body_Impedance/SafeHumanoid__VLM-RAG-driven_Control_of_Upper_Body_Impedance.html)
