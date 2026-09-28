---
type: entity
tags: [paper, humanoid-paper-notebooks, humanoid, manipulation, impedance-control, vlm, rag, contact-rich, unitree]
status: complete
updated: 2026-09-28
arxiv: "2601.14874"
venue: "HRI 2026 (Late-Breaking Report)"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../concepts/impedance-control.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/llm-robotics-control-interfaces.md
  - ./paper-notebook-safehumanoid-vlm-rag-driven-control-of-upper-bod.md
  - ../methods/tactile-impedance-control.md
sources:
  - ../../sources/papers/humanoid_pnb_humanoidvlm-vision-language-guided-impedance-con.md
summary: "HumanoidVLM 把\"挑阻抗参数 + 选抓取角\"这件老靠手调的事，外包给一个轻量管线：VLM 看一眼第一视角图把任务和物体说出来 → FAISS-RAG 从两个小数据库（9 个任务 + 9 个物体）里查出实验验证过的 stiffness/damping 与手指角→ 直接喂给 G1 的任务空间阻抗控制器，让接触富集的人形操作\"软硬合适\"。14 个测试场景命中率 93%。"
---

# HumanoidVLM

**HumanoidVLM: Vision-Language-Guided Impedance Control for Contact-Rich Humanoid Manipulation** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation）。本页编译自深读笔记与 arXiv 论文正文（实验数字、开源状态于 2026-09-28 核对），细节以论文 PDF 为准。

## 一句话定义

HumanoidVLM 把"挑阻抗参数 + 选抓取角"这件老靠手调的事，外包给一个轻量管线：VLM 看一眼第一视角图把任务和物体说出来 → FAISS-RAG 从两个小数据库（9 个任务 + 9 个物体）里查出实验验证过的 stiffness/damping 与手指角→ 直接喂给 G1 的任务空间阻抗控制器，让接触富集的人形操作"软硬合适"。14 个测试场景命中率 93%。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 本文用 Molmo-7B 从第一视角图推断任务 |
| RAG | Retrieval-Augmented Generation | 用向量检索从数据库取出已验证参数 |
| FAISS | Facebook AI Similarity Search | 向量相似度检索库 |
| K / D | Stiffness / Damping | 任务空间阻抗的刚度与阻尼 |
| HRI | Human-Robot Interaction | 论文发表的会议方向（HRI 2026 LBR） |

## 为什么重要

- **替代手调阻抗参数**：多数人形控制器用固定、手调的阻抗增益与夹爪设置，换任务就要重调。
- **可解释**：参数来自人工验证过的数据库条目，出问题可以直接追溯到哪一条。
- **无需力传感**：在没有腕部力传感的 G1 上实现柔顺接触。
- **免训练**：只需构建两个小数据库，不训练控制策略。

## 核心机制

1. **任务推断**：头部第一视角 RGB → Molmo-7B（4-bit）通过层级是 / 否问答得到任务标签。
2. **两级检索**：标签经 MiniLM 嵌入，FAISS 先检索阻抗条目（K、D），再以「场景描述 + 标签」检索夹爪闭合角。
3. **执行**：任务空间笛卡尔质量–弹簧–阻尼模型生成柔顺的末端参考轨迹，经 IK 转为关节目标，由 G1 自带位置控制器执行；虚拟力 F = K·e + D·ė 作为接触力代理。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation.html> |
| arXiv | <https://arxiv.org/abs/2601.14874> |
| 源码 | **未开源**：论文与 arXiv 页未给出代码链接（正文链接的 Hugging Face 页面为所用的 Molmo-7B 与 MiniLM 公共模型），截至 2026-09-28 未见项目页 |

## 实验与评测


**设置**：Unitree G1（双 1-DoF 夹爪、头部 RealSense）；任务空间阻抗控制器在机载 PC 上 50 Hz 运行，VLM–RAG 管线在外部 RTX 4090 工作站。VLM 为 Molmo-7B-O（4-bit），通过一串是 / 否视觉问答推断任务；标签经 all-MiniLM-L6-v2 嵌入后，用 FAISS 分两级检索：先检索阻抗条目，再把场景描述与任务标签拼接检索夹爪角。数据库为两个 JSON：9 个任务的阻抗参数（每任务试多组、取最稳定柔顺的一组）+ 9 类物体的夹爪闭合角。

- **检索准确率**：14 张第一视角测试图（覆盖 9 类任务，视角、摆放、手臂姿态不同）中 13 张检索正确，**93%**；唯一失败是主要物体被部分遮挡。
- **阻抗执行**（Table 1，z 向）：

| 任务 | K_z [N/m] | D_z [Ns/m] | 平均 / 最大 \|e_z\| [m] | 最大虚拟力 [arb.] |
|------|---:|---:|------|---:|
| 沿曲面跟随（右） | 3.0 | 2.0 | 0.016 / 0.024 | 0.103 |
| 按压按摩球（右） | 5.0 | 3.0 | 0.017 / 0.035 | 0.334 |
| 双手放置·鸡蛋（右） | 2.0 | 1.0 | 0.013 / 0.034 | 0.089 |
| 双手放置·酱瓶（左） | 6.0 | 1.5 | 0.009 / 0.013 | 0.176 |
| 叉子戳果蔬（右） | 2.0 | 1.5 | 0.006 / 0.016 | 0.183 |
| 抓取提起（右） | 4.0 | 1.5 | 0.013 / 0.022 | 0.153 |

- z 向跟踪误差大多在 1–3.5 cm；虚拟力随所选刚度 / 阻尼变化一致。双手放置时易碎的鸡蛋用软参数、酱瓶用硬参数，体现了按物体分配阻抗。

## 与其他工作对比

| 做法 | 阻抗参数如何确定 | 与 HumanoidVLM 的差异 |
|------|------|------|
| 固定手调增益（常见做法） | 人工调一组参数通用 | 不随任务 / 物体变化 |
| [SafeHumanoid](./paper-notebook-safehumanoid-vlm-rag-driven-control-of-upper-bod.md) | 同组 VLM + RAG 驱动上身控制 | 同样检索式思路，侧重安全相关的上身行为 |
| [触觉阻抗控制](../methods/tactile-impedance-control.md) | 触觉反馈在线调节 | 闭环依赖触觉；HumanoidVLM 无力 / 触觉传感，只在开始时按图像选参数 |
| 端到端学习阻抗（如可变阻抗 RL） | 策略直接输出增益 | 需训练数据；HumanoidVLM 免训练但只能覆盖数据库内任务 |

## 结论

**HumanoidVLM 没有去学一个新的阻抗控制器，而是把「阻抗参数与抓取角怎么定」改写成一次检索问题：VLM 认场景、RAG 查已验证的参数，控制器本身保持不变。**

- 真正起作用的是两个很小的数据库（**9 个任务 + 9 个物体**）里**实验验证过的 stiffness/damping 与手指角**，由 FAISS-RAG 检索取出，直接喂给 G1 的任务空间阻抗控制器。
- 报告指标是 14 个测试场景 **93% 命中率**——衡量的是"参数查得对不对"，不是长时程操作的成功率，读数时别放大。
- 适用边界由数据库覆盖决定：任务或物体落在 9+9 之外时管线没有外推机制，这是最直接的失败模式。
- 定位是接触富集人形操作里替代手调参数的**轻量管线**，价值在工程可用而非方法新颖；执行侧 z 向误差 1–3.5 cm、虚拟力随增益一致变化，但没有实测接触力验证。

## 局限与风险

- **评估规模小**：14 张测试图、9 类任务，论文结论节也提醒结果应在有限规模下解读。
- **无外推机制**：任务或物体不在 9 + 9 的数据库中时无法给出参数；数据库条目靠人工实验逐一标定。
- **没有力传感**：G1 手腕无六维力传感、夹爪无反作用力反馈，「虚拟力」只是由位姿误差计算的代理量，不是实测接触力。
- **仅 z 向、桌面场景**：评估聚焦法向交互；只调平动刚度 / 阻尼，虚拟质量与转动阻抗固定。
- **遮挡敏感**：唯一的检索失败来自目标被部分遮挡。
- **开源边界**：未见代码；源码运行时序图 **不适用**。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 阻抗控制：[impedance-control](../concepts/impedance-control.md)
- 接触丰富型操作：[contact-rich-manipulation](../concepts/contact-rich-manipulation.md)
- LLM / VLM 与控制器的接口方式：[llm-robotics-control-interfaces](../concepts/llm-robotics-control-interfaces.md)
- 同组 VLM + RAG 控制路线：[paper-notebook-safehumanoid-vlm-rag-driven-control-of-upper-bod](./paper-notebook-safehumanoid-vlm-rag-driven-control-of-upper-bod.md)
- 基于触觉反馈的阻抗控制对照：[tactile-impedance-control](../methods/tactile-impedance-control.md)

## 参考来源

- [humanoid_pnb_humanoidvlm-vision-language-guided-impedance-con.md](../../sources/papers/humanoid_pnb_humanoidvlm-vision-language-guided-impedance-con.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation.html>
- 论文：<https://arxiv.org/abs/2601.14874>
- 论文正文（系统架构、Table 1）：<https://arxiv.org/html/2601.14874>

## 推荐继续阅读

- [机器人论文阅读笔记：HumanoidVLM](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation.html)
