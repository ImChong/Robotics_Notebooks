# Unlocking fast robotic locomotor propulsion through dynamic spine-leg synergy（Science Advances, 2026）

> 来源归档（ingest）

- **标题：** Unlocking fast robotic locomotor propulsion through dynamic spine-leg synergy
- **类型：** paper / quadruped / bio-inspired / spine-leg synergy / micro robot / robophysics
- **期刊：** *Science Advances*，2026-09-25（Crossref）；Issue 39
- **DOI：** <https://doi.org/10.1126/sciadv.aed5603>
- **Science 页：** <https://www.science.org/doi/10.1126/sciadv.aed5603>
- **PubMed：** <https://pubmed.ncbi.nlm.nih.gov/42789728/>（PMID 42789728）
- **机构：** 北京理工大学（BIT，第一单位）；南京大学（NJU，合作）；长三角研究院（嘉兴）
- **作者（Crossref）：** Ruochao Wang（一作）；Weitao Zhang, Xiaolong Quan, Rongjie Du, Zhenshan Bing, Gang Wang, Jian Sun, Zhiqiang Yu；Qing Shi（末位，通讯）
- **平台：** **FLEXOR**（*F*ast *Leg*ged robot with a *Flex*ible spine for *O*ptimal *R*unning）— 微小型仿生四足，**双关节耦合脊柱** + **弹性腿**；仿生原型 **象鼩（elephant shrew）**
- **代码与数据：** 截至 **2026-10-01**，[北理工新闻稿](../sites/bit-flexor-sciadv-2026.md) 与 DOI 落地页 **未见官方 GitHub / Zenodo** → **确认未开源**
- **入库日期：** 2026-10-01
- **一句话说明：** 通过 **FLEXOR** 的仿真与实机实验，识别 **动态脊-腿协同** 的最优相位（脊柱伸展与后腿后摆对齐），利用闭链脊柱 **放大力矩 → 增大 GRF 与推进**，在 **不增加额外能量输入** 下显著提升小尺度四足速度与 CoT，并用扩展动力学框架说明该规律跨形态与尺度泛化。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| DOI | [10.1126/sciadv.aed5603](https://doi.org/10.1126/sciadv.aed5603) | 期刊原文 |
| PubMed | [42789728](https://pubmed.ncbi.nlm.nih.gov/42789728/) | 书目记录 |
| 机构稿 | [bit-flexor-sciadv-2026.md](../sites/bit-flexor-sciadv-2026.md) | 开源核查 + 中文实验数字 |
| 任务 | [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md) | 四足高速运动 |
| 对照 | [`wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md`](../../wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md) | 生物启发四足步态/切换（Nature MI，不同尺度与问题） |

## 摘要级要点（Crossref abstract + 新闻稿互证）

- **问题：** 自然四足靠 **脊柱节律屈伸 + 腿协同** 实现敏捷运动（身体编码的 physical intelligence），但 **如何把脊-腿协同识别并嵌入机器人** 以提升推进仍难。
- **系统：** **FLEXOR** — 双关节 **耦合脊柱** 将作动器 **单向旋转** 转为快速周期脊柱屈伸；相对传统 **单关节脊柱**，闭链多连杆 **放大驱动扭矩** 并传到后腿。
- **机制：** 多参数动力学模型以 **后腿后摆相对脊柱伸展的相位差** 为关键协同变量；**最优** 处 GRF **前向分量** 与 **绕质心俯仰力矩** 同时更优 → 速度、稳定性、CoT 同步改善。
- **实机（新闻稿，对照同质量/尺寸/峰值功率单关节脊柱机）：** 速度 **0.784 m/s**（**>7 BL/s**）；速度 **+31.6%**，CoT **−32.2%**；峰值后腿 GRF **5.19 N vs 2.82 N**。
- **泛化：** **扩展动力学框架** + robophysical 验证 → 不同脊柱/腿构型与尺度下，最优协同仍落在「脊柱伸展 ≈ 后腿后摆」附近。
- **局限：** 微小型平台、特定脊柱机构；**无公开代码** 限制工程复现；高速仍受驱动功率与结构带宽约束。

## 核心摘录（面向 wiki 编译）

### 1) 双关节耦合脊柱 vs 单关节脊柱

| 维度 | 单关节脊柱（对照） | FLEXOR 双关节耦合脊柱 |
|------|-------------------|------------------------|
| 驱动→脊柱运动 | 常需往复控制 | 单向旋转 → 周期屈伸 |
| 力传递 | 基准 | 闭链放大扭矩 → 后腿 GRF ↑ |
| 实机速度 / CoT | 基准 | +31.6% / −32.2%（新闻稿同规格对照） |

### 2) 最优脊-腿协同（相位）

```
脊柱伸展相位  ←—接近同步—→  后腿后摆相位
        ↓
GRF 前向分量 ↑；力线近质心 → 俯仰力矩 ↓
        ↓
腾空相 ↑、支撑相 ↓ → 触地损耗 ↓
```

### 3) 开源状态（步骤 2.5）

- **项目页：** 无独立 `*.github.io`；仅有北理工新闻网稿链到 DOI。
- **结论：** **确认未开源**（2026-10-01）；后续若期刊 Data availability 或团队发布仓库，应另建 `sources/repos/` 并补 wiki 时序图。

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-flexor-dynamic-spine-leg-synergy.md`](../../wiki/entities/paper-flexor-dynamic-spine-leg-synergy.md)**
- 交叉：**[`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)**（四足 / 生物启发高速运动）
