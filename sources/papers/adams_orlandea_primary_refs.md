# ADAMS（Automatic Dynamic Analysis of Mechanical Systems）一手学术资料索引

> 来源归档（ingest）

- **标题：** Orlandea / Chace / Calahan 多体动力学与 ADAMS 程序的经典论文与学位论文
- **类型：** paper / thesis（合集）
- **入库日期：** 2026-07-28
- **一句话说明：** 汇总 ADAMS（Automatic Dynamic Analysis of Mechanical Systems）从密歇根博士论文到 1977 ASME 双篇方法论文、再到作者 2016 史述的一手学术链，作为工业多体动力学（MBD）软件谱系的原始依据。
- **沉淀到 wiki：** 是 → [adams](../../wiki/entities/adams.md)
- **产品页：** [Cadence MSC Adams](../sites/cadence-msc-adams.md)
- **学位论文馆藏页：** [U-M Deep Blue Orlandea 1973](../sites/umich-deepblue-orlandea-adams-thesis.md)

## 为什么值得保留

- ADAMS 是 **工业多体动力学仿真** 的奠基性程序名与商业谱系；机器人学习侧常用的 MuJoCo / Drake / Isaac 等引擎，在「约束多体 + DAE 数值」叙事上与之同源，但目标场景不同（工业整机验证 vs RL/控制研究）。
- 一手链可避免把 ADAMS 误写成「只是 MSC/Cadence 商业品牌」：名称与核心数值配方来自 **Orlandea 1973 论文工作 + 1977 ASME 发表**。
- 本索引只收录 **学位论文、期刊论文、作者自述史述**；商业产品页另见 `sources/sites/`。

## 核心摘录

### 1) Orlandea (1973) — 密歇根博士论文（ADAMS 程序源头）

- **来源：** Nicolae Orlandea, *Node-Analogous, Sparsity-Oriented Methods for Simulation of Mechanical Dynamic Systems*, Ph.D. thesis, University of Michigan, Ann Arbor, 1973.
- **馆藏 / DOI：** [Deep Blue 条目](https://deepblue.lib.umich.edu/items/cae8a491-4968-4da6-a1b0-090ba1d38523) · DOI [10.7302/10731](https://doi.org/10.7302/10731) · Handle [hdl.handle.net/2027.42/180342](https://hdl.handle.net/2027.42/180342) · ProQuest Dissertation No. **7415821**
- **开放状态（2026-07-28 核查）：** Deep Blue 标注 PDF **仅限 U-M 校园用户**（Access Restricted to UM users only）；元数据与题名对外可检索。
- **要点（据作者 2016 史述与后续商业化叙事交叉核对）：**
  - 采用 **节点类比（node-analogous）** 表述、**稀疏 tableau（Sparse Tableaux Formulation, STF）**、**Gear BDF** 刚性积分与 **Lagrange 方程**，实现三维机械系统数值仿真程序。
  - 作者将该程序命名为 **ADAMS（Automatic Dynamic Analysis of Mechanical Systems）**。
  - 早期应用叙事包含 Boeing 747 起落架、整车与材料点阵等（见 §4 史述）。
- **对 wiki 的映射：** [adams](../../wiki/entities/adams.md)

### 2) Maros & Orlandea (1971) — 多自由度运动方程（前序）

- **来源：** Desideriu Maros, Nicolae Orlandea, *Contributions to the Determination of the Equations of Motion for Multidegree of Freedom Systems*, ASME *Journal of Engineering for Industry*, 93(1):191–195, 1971. DOI [10.1115/1.3427874](https://doi.org/10.1115/1.3427874)
- **要点（Crossref 摘要）：**
  - 在平面多自由度机构传动函数工作基础上，建立对应的 **运动微分方程组**，目标是 **便于计算机编程**。
  - 以 Beyer 的工作为出发点，采用从一般到特殊的演绎写法；强调二、三自由度情形可作为一般结果的特例。
- **对 wiki 的映射：** [adams](../../wiki/entities/adams.md)（数值程序前的解析/编程友好方程层）

### 3) Orlandea, Chace & Calahan (1977) — Part 1：运动方程与约束

- **来源：** N. Orlandea, M. A. Chace, D. A. Calahan, *A Sparsity-Oriented Approach to the Dynamic Analysis and Design of Mechanical Systems—Part 1*, ASME *Journal of Engineering for Industry*, 99(3):773–779, 1977-08-01. DOI [10.1115/1.3439312](https://doi.org/10.1115/1.3439312)
- **要点（Crossref 摘要；Semantic Scholar 引用量约 219，截至检索日）：**
  - 将电路仿真中成熟的 **稀疏矩阵 + 刚性（stiff）积分** 算法扩展到三维机械动力学。
  - 大稀疏线性方程组可高效求解；宽谱特征值导致的数值不稳定性可被抑制；**数值方法反过来影响问题的初始列式**。
  - Part 1 聚焦 **运动方程与约束**；与 Part 2 一起构成 ADAMS 程序的方法论文发表。
- **对 wiki 的映射：** [adams](../../wiki/entities/adams.md)

### 4) Orlandea, Calahan & Chace (1977) — Part 2：力元、分析类型与程序实现

- **来源：** N. Orlandea, D. A. Calahan, M. A. Chace, *A Sparsity-Oriented Approach to the Dynamic Analysis and Design of Mechanical Systems—Part 2*, ASME *Journal of Engineering for Industry*, 99(3):780–784, 1977-08-01. DOI [10.1115/1.3439313](https://doi.org/10.1115/1.3439313)
- **要点（Crossref 摘要；Semantic Scholar 引用量约 128）：**
  - 补齐 **弹簧/阻尼等力函数**，并实现 **静力、瞬态、线性化** 分析及 **模态优化** 算法。
  - 明确写出：上述数值方法 **实现于 ADAMS（automatic dynamic analysis of mechanical systems）计算机程序**，用于三维机械系统仿真。
  - 算例：**1973 Chevrolet Malibu 前悬架**、**Boeing 747 起落架**；给出仿真效率与实验结果对照表。
- **对 wiki 的映射：** [adams](../../wiki/entities/adams.md)

### 5) Orlandea (2016) — 《Multibody Systems History of ADAMS》（作者一手史述）

- **来源：** Nicolae V. Orlandea, *Multibody Systems History of ADAMS*, ASME *Journal of Computational and Nonlinear Dynamics*, 11(6), 2016-11-01. DOI [10.1115/1.4034296](https://doi.org/10.1115/1.4034296)
- **要点（公开元数据 + 作者自述片段交叉核对；全文需 ASME 权限）：**
  - 作者第一人称回顾：1973 博士工作将 **STF + BDF + Lagrange** 写入名为 ADAMS 的程序；并称由此推开 **「Multibody System Dynamics」** 这一称谓的接受过程。
  - 记述与 Milt Chace、Mechanical Dynamics, Inc.（MDI）商业化、以及 Gear 积分 **Index** 选择等相关里程碑。
  - 引用链回指 1971 Maros–Orlandea 与 1973 学位论文。
- **对 wiki 的映射：** [adams](../../wiki/entities/adams.md)

## 商业化与产品谱系（交叉指针，非本文件主体）

| 节点 | 说明 | 归档 |
|------|------|------|
| Mechanical Dynamics, Inc.（MDI, 1976） | Chace / Korybalski / Angell 等基于 Orlandea 工作商业化 ADAMS | 见 Janevic 纪念文与 2016 史述 |
| MSC Software → Hexagon → Cadence | 当代产品页称 Adams 为 MBD「gold standard」，垂直线含 Adams Car、Adams Real Time 等 | [cadence-msc-adams](../sites/cadence-msc-adams.md) |

## 开源结论（2026-07-28）

- **学术方法已公开发表**（ASME 期刊 + 学位论文元数据）。
- **当代 Adams 产品代码未开源**（商业 CAE 许可）；无公开训练/推理式「官方 GitHub 复现仓」可比拟 MuJoCo。
- 学位论文 PDF 在 Deep Blue **校园限制**；1977 双篇与 2016 史述以 DOI 为准，全文访问取决于机构订阅。

## 推荐继续阅读（外部）

- C. W. Gear — 刚性 ODE / DAE 的 BDF 数值方法（ADAMS 积分叙事的数值底座）
- 当代工业对照：Cadence [Adams 产品页](https://www.cadence.com/en_US/home/tools/msc-software/adams.html)
- 机器人研究侧对照：[MuJoCo](../../wiki/entities/mujoco.md)、[Drake](../../wiki/entities/drake.md)

## 当前提炼状态

- [x] 五条一手学术来源摘录与 wiki 映射
- [x] Deep Blue / Cadence 站点归档交叉链接
- [x] 开源与全文访问边界写明
- [ ] 后续可补：1977 双篇公式级 STF/DAE 摘录（需全文权限）、Adams Car / Real Time 模块对照表
