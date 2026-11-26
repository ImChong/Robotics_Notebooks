---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-11-26
project_id: aff-affordgrasp-in-context-affordance-reasoning-for-open-voc
doi: "10.1109/IROS60139.2025.11245995"
venue: "IROS 2025"
summary: "IROS 2025 论文（arXiv:2503.00778v1），提出无需额外训练的开放词汇杂乱场景任务导向抓取框架 AffordGrasp：通过 GPT-4o 上下文三步推理隐式指令得到任务-物体-部件-affordance，再由 VLPart 两阶段视觉接地并结合 AnyGrasp/GraspNet 筛选质心对齐最优 6D 抓取。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_affordgrasp-in-context-affordance-reasoning-for-open-voc.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AffordGrasp

**AffordGrasp: In-Context Affordance Reasoning for Open-Vocabulary Task-Oriented Grasping in Clutter**（IROS 2025，arXiv:2503.00778v1，中科院自动化所 / 哈工大深圳 / 智源研究院 / 北京大学）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套文献。本页基于论文全文完成结构化精读与核实。

## 一句话定义

**AffordGrasp** 是一个**无需额外微调训练（training-free）**的开放词汇杂乱场景任务导向抓取框架：利用 **GPT-4o** 对隐式语言指令和视觉场景做三步上下文 affordance 推理，经 **VLPart** 两阶段开放词汇部件接地后，从 **AnyGrasp / GraspNet** 候选中挑选靠近 affordance 质心的高分 6D 抓取。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 视觉–语言模型，本文采用 GPT-4o 进行上下文多模态意图与部件推理 |
| VLPart | Open-Vocabulary Part Segmentation | 开放词汇部件分割模型，完成目标物体框定位与部件 affordance 掩码接地 |
| GSR | Grasp Success Rate | 抓取成功率，评估单物体及杂乱场景仿真与真机任务导向抓取的核心指标 |
| RGB-D | Red-Green-Blue + Depth | 彩色与深度图像输入，分别用于视觉推理/分割与 3D 局部点云抓取生成 |

## 为什么重要

- 既有任务导向抓取方法（如 GraspGPT、FoundationGrasp）普遍依赖显式物体名称并需大量标注数据训练抓取评估器，或仅聚焦单物体桌面场景（如 LERF-TOGO、ShapeGrasp），难以应对“我渴了（I am thirsty）”这类隐式指令和杂乱堆叠桌面。
- AffordGrasp 通过“VLM 上下文推理 → 两阶段目标隔离部件分割 → 掩码质心对齐 6D 抓取筛选”全免训练管线，在杂乱真机环境中达到 **83%** 的任务导向抓取成功率。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | IROS 2025（arXiv:2503.00778v1） |
| 作者与机构 | Yingbo Tang, Shuaike Zhang, Xiaoshuai Hao, Pengwei Wang, Jianlong Wu, Zhongyuan Wang, Shanghang Zhang（CASIA / HIT-SZ / BAAI / PKU） |
| 方法名（清单标注） | AffordGrasp |
| 清单分组 1 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 论文与项目页 | <https://doi.org/10.1109/IROS60139.2025.11245995> · <https://arxiv.org/abs/2503.00778> · <https://eqcy.github.io/affordgrasp/> |

## 核心机制

1. **上下文 Affordance 推理（In-Context Affordance Reasoning）**
   - 给定隐式自然语言指令 $L$ 与场景 RGB 图像 $I$，通过结构化上下文提示引导 **GPT-4o** 完成三步推理并输出标准四元组 $(T, O, p^*, a^*) = \text{GPT-4o}(L, I)$：
     - **Step 1（任务分析）**：从隐式需求（如“I want to scoop something”）抽取显式任务 $T$ 及所需工具功能属性；
     - **Step 2（相关物体识别）**：结合场景视觉观察定位最适配任务的目标物体 $O$（如木勺），排除杂乱桌面上的无关干扰物；
     - **Step 3（部件分解与 Affordance 推理）**：将 $O$ 拆解为功能部件集合，依据物理特性选择最佳抓握部件 $p^*$（如手柄）及其 affordance $a^*$（如 grasp / hold / grip）。
2. **两阶段视觉 Affordance 接地（Visual Affordance Grounding）**
   - 直接在杂乱全图检索细粒度部件易受相邻物体干扰，因此利用开放词汇部件分割模型 **VLPart** 分两步执行（推理输入缩放为 224×224）：
     - **物体定位与掩蔽**：先提取目标物体边界框 $B_O = \text{VLPart}(I, O)$，将框外像素置零生成隔离图像 $M_{B_O}$；
     - **部件级掩码预测**：在 $M_{B_O}$ 上基于 $(p^*, a^*)$ 预测像素级可交互部件掩码 $M_{p^*} = \text{VLPart}(M_{B_O}, p^*, a^*)$。
3. **几何感知 6D 抓取生成与质心筛选**
   - 利用相机内参将深度图反投影为局部视角点云，并用 $M_{p^*}$ 过滤掉背景与非目标部件点云；
   - 在仿真中调用 **GraspNet**、在真机中调用 **AnyGrasp** 生成 6D 抓取候选 $g = [R, t, w]$，随后依据以下目标筛选最优抓取：
     $$g^* = \arg\max_{g \in G} \frac{\text{score}(g)}{\|t(g) - c\|_2}$$
     其中 $c = [x,y,z]$ 为 affordance 掩码点云的几何质心，兼顾抓取置信度与几何中心对齐度（避免如 ThinkGrasp 依赖 3×3 网格划分导致在马克杯弧形杯沿等部位抓取失稳）。

### 方法流程图

```mermaid
flowchart LR
    A["隐式指令 L + 场景 RGB 图像 I"] --> B["GPT-4o 上下文三步推理"]
    B --> C["输出四元组 (Task T, Object O, Part p*, Affordance a*)"]
    A --> D["VLPart 阶段1: 目标物体框 B_O 定位与背景掩蔽 M_BO"]
    C --> D
    D --> E["VLPart 阶段2: 部件 Affordance 掩码 M_p* 接地"]
    F["深度图 D + 相机内参"] --> G["局部点云反投影与 M_p* 掩码过滤"]
    E --> G
    G --> H["AnyGrasp / GraspNet 生成 6D 抓取候选 g=[R, t, w]"]
    H --> I["质心对齐评分筛选 g* = argmax score(g)/||t(g)-c||_2"]
    I --> J["UR5 机械臂执行任务导向抓取"]
```

## 评测与指标

所有仿真（PyBullet + UR5 + ROBOTIQ-85 + L515，单场景 50 次试验）与真机实验（UR5 + RS-485 夹爪 + eye-to-hand L515 @1280×720，8 类物体各 10 次）均以 **Grasp Success Rate（GSR ↑）** 为核心指标：

| 实验场景 | 对比方法 | 平均 GSR ↑ | 典型分项表现 |
|------|------|------------|--------------|
| 仿真 · 单物体（7 类×50 次，Table I） | GraspNet | 0.57 | Scissors 0.14 · Wine Glass 0.44 · Screwdriver 0.84 |
| 仿真 · 单物体（Table I） | ThinkGrasp | 0.75 | Scissors 0.48 · Wine Glass 0.46 · Screwdriver 0.94 |
| 仿真 · 单物体（Table I） | **AffordGrasp（Ours）** | **0.85** | Scissors **0.62** · Wine Glass **0.66** · Screwdriver **1.00** · Spoon **0.98** |
| 仿真 · 杂乱场景（7 类×50 次，Table II） | ThinkGrasp | 0.54 | Screwdriver 0.16 · Wine Glass **0.00** · Scissors 0.30 |
| 仿真 · 杂乱场景（Table II） | **AffordGrasp（Ours）** | **0.77** | Screwdriver **0.76** · Wine Glass **0.52** · Hammer **0.90** · Spoon **0.92** |
| 真机 · 杂乱场景（8 类×10 次，Table III） | RAM | 0.14 | Bottle 6/10 · Mug 1/10 · 其余 6 类均 0/10 |
| 真机 · 杂乱场景（Table III） | AnyGrasp | 0.61 | Mug 3/10 · Kettle 2/10 · Knife 10/10 |
| 真机 · 杂乱场景（Table III） | ThinkGrasp | 0.73 | Spatula 5/10 · Mug 6/10 · Kettle **7/10** · Screwdriver **9/10** |
| 真机 · 杂乱场景（Table III） | **AffordGrasp（Ours）** | **0.83** | Spatula **9/10** · Pan **9/10** · Mug **8/10** · Knife **10/10** · Kettle 6/10 |

## 与其他工作对比

- **对比纯 LLM 推理方案（LAN-Grasp、OVAL-Prompt）**：前者缺乏实时视觉场景联合推理，无法根据当前桌面摆放情况从隐式指令反推正确物体；AffordGrasp 利用 VLM 图文上下文同时消歧物体与部件。
- **对比强杂乱基线（ThinkGrasp）**：ThinkGrasp 倾向选取图像最显著区域并用 3×3 网格中心选抓，导致杂乱场景下高脚杯完全失败（0.00）、螺丝刀仅 0.16；AffordGrasp 先框定物体再分割部件并做质心距离加权，在复杂几何与杂乱场景优势显著。
- **对比无任务推理 6D 抓取/迁移方案（AnyGrasp、RAM）**：AnyGrasp 常抓在锅身或刀刃而非手柄；RAM 受限于小规模记忆库（仅瓶子 6/10，其余多类 0/10）。同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。

## 常见误区或局限

1. **“Training-free”指系统无需额外任务微调，而非完全零学习先验**：整个管线高度依赖闭源 **GPT-4o** 的推理能力、**VLPart** 的开放词汇部件分割精度（输入缩放至 224×224 可能损失超细部件细节）以及 **AnyGrasp / GraspNet** 预训练抓取生成器的质量；三者呈串行级联，上游错检或漏分割难以在下游恢复。
2. **仅覆盖单臂抓取阶段**：实验仅评估抓起目标部件的成功率（GSR），未验证抓起后的工具后续操作（如倾倒、刮炒、拧螺丝）闭环控制与手内滑移恢复。
3. **重型/特殊曲面物体仍有改进空间**：真机上水壶（Kettle）成功率仅 6/10（略低于 ThinkGrasp 的 7/10），且尚未在严重遮挡需先移开障碍物的多步整理场景验证。
4. **开源核实提示**：论文已给出项目主页 `https://eqcy.github.io/affordgrasp/`，但正文未明确说明代码仓库具体开放状态，复现前需按项目页核查。

## 结论

**AffordGrasp 将“VLM 上下文三步推理 + VLPart 两阶段隔离接地 + 质心加权 6D 抓取筛选”串联成无需微调的开放词汇任务抓取管线，在杂乱桌面仿真（0.77 GSR）与真机（0.83 GSR）上大幅超越纯抓取与网格启发式 VLM 基线。**

- 面对隐式指令与多物体杂乱干扰时，先由 VLM 图文联合锁定物体再做框内局部掩蔽分割，可显著降低跨物体背景误分割。
- 使用点云掩码几何质心加权抓取置信度（`score(g)/||t(g)-c||_2`），比 3×3 图像网格中心法更适配杯沿、细长手柄等弯曲/细小部件。
- 实际工程落地时应留意 GPT-4o 推理延迟及 API 依赖，并针对细小工具适当提高 VLPart 输入分辨率或加入重试/遮挡重规划机制。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/affordance_awesome_affordgrasp-in-context-affordance-reasoning-for-open-voc.md`](../../sources/papers/affordance_awesome_affordgrasp-in-context-affordance-reasoning-for-open-voc.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://doi.org/10.1109/IROS60139.2025.11245995>

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://doi.org/10.1109/IROS60139.2025.11245995)
