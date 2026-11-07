---
type: concept
tags: [mechanical-design, fastener, thread, screw, manufacturing, standards]
status: draft
updated: 2026-11-07
original:
  - ../../sources/blogs/fastener_quanjiechu_thread_basics_2024-04-10.md
related:
  - ../entities/mechtool.md
  - ./humanoid-mechanical-layout-design.md
  - ./robot-joint-scheme-selection.md
  - ./robot-joint-bearing-selection.md
  - ./planetary-roller-screw-humanoid-leg-actuation.md
  - ./structure-tolerance-simulation-toolchain.md
sources:
  - ../../sources/blogs/fastener_quanjiechu_thread_basics_2024-04-10.md
  - ../../sources/sites/mechtool_cn_catalog_2026-11-07.md
summary: "螺纹入门：分类（普通 M、管螺纹 G/Rc/Rp/R/NPT、梯形 Tr、锯齿形 B）、普通螺纹基本尺寸公式、性能等级、攻丝底孔与套丝杆径经验式；含本库对文章公式与表值的重算结果和两处待核疑点。"
---

# 螺纹基础（分类、尺寸公式、性能等级）

## 一句话定义

**螺纹**是在圆柱或圆锥表面沿螺旋线形成的、具有相同牙型的连续凸起；按用途分连接螺纹、传动螺纹、密封（管）螺纹。本页依据微信公众号「紧固件全接触」的入门手册（2024-04-10）整理，并用 Python 重算了其中的公式和数值表（见「核心原理」）。

## 英文缩写速查

| 缩写 | 含义 | 说明 |
|------|------|------|
| M | 普通（公制）螺纹 | 牙型角 60° |
| G / Rp / Rc / R | 55° 管螺纹 | G 非密封；Rp 圆柱内、Rc 圆锥内、R 圆锥外（密封） |
| NPT | 60° 圆锥管螺纹 | 美制，锥度 1:16 |
| Tr | 梯形螺纹 | 传动用，牙型角 30° |
| B | 锯齿形螺纹 | 单向受力传动，3° / 30° 非对称牙侧 |
| P | 螺距 | 相邻牙对应点轴向距离 |
| d / d1 / d2 | 大径 / 小径 / 中径 | 外螺纹 |

## 为什么重要

- 机器人本体里螺纹连接、传动（丝杠、行星滚柱丝杠）、管路接头无处不在；选紧固件等级、定底孔直径、查标准号都靠这些基础量。
- 与本库关节、轴承、公差页互补：[人形机械布局](./humanoid-mechanical-layout-design.md)、[关节方案选型](./robot-joint-scheme-selection.md)、[关节轴承选型](./robot-joint-bearing-selection.md)、[行星滚柱丝杠](./planetary-roller-screw-humanoid-leg-actuation.md)、[公差仿真](./structure-tolerance-simulation-toolchain.md)。

## 核心原理

```mermaid
flowchart LR
  A["确定用途"] --> B{"连接 / 传动 / 密封"}
  B -->|连接| C["普通螺纹 M<br/>查 GB/T 193 直径螺距"]
  B -->|传动| D["Tr 梯形 / B 锯齿形"]
  B -->|密封| E["管螺纹 G Rc Rp R NPT"]
  C --> F["性能等级 + 底孔直径"]
```

**普通螺纹基本尺寸**（文章公式，已重算）：H = 0.866P；d2 = d − 0.6495P；d1 = d − 1.0825P。系数分别为 √3/2、3√3/8、5√3/8，重算一致。

**本库重算结果**（Python，非一手来源）：
- 55° 圆锥管螺纹 Rc 8 行：d1 = d − 2h，h = 0.640327P，P = 25.4/n，与文章一致（差 ≤ 0.001 mm）。
- 梯形螺纹 8 行（d3、d2、D4、D1，h3 = 0.5P + ac）与文章一致。
- 性能等级 4.6、5.6、6.8、8.8、9.8 与「前数 ×100 = σb，后数 ×10% = 屈强比」一致；3.6、4.8、5.8、10.9、12.9 表值略高于名义值（推测与最小值取法有关，未核标准）。

**性能等级表（文章）：** 3.6–12.9，σb 最小 330…1220 MPa，σs 最小 190…1100 MPa。

**攻丝与套丝经验式（文章）：** 底孔直径 塑性材料 D − P、脆性材料 D − (1.05~1.1)P；盲孔深度需加约 0.7D；套丝杆径 D − 0.13P。

**ISO / GB/T 对照**（Mechtool 螺纹页）：基本牙型 ISO 68-1 ↔ GB/T 192-2003；直径螺距 ISO 261 ↔ GB/T 193-2003；公差 ISO 965-1 ↔ GB/T 197-2003。

## 工程实践

- 底孔、等级、拧紧力矩可先用 [Mechtool](../entities/mechtool.md) 的「螺栓连接设计校核」或其外链紧固件小工具粗算，关键件仍以现行标准和供应商资料为准。
- 引用标准号前核对版本：文章用 GB/T 192-81、7306-87、5796.3-86 等旧号，Mechtool 列的是 -2003 版。
- 螺纹规格（粗牙 / 细牙、左右旋、公差带）要在图纸标注里写全。

## 局限

- **通读未完成：** 原文大量表格和图片在抓取文本里缺失，且第 7 章重复出现，只读到文字部分。
- **疑点：** 文章把锯齿形螺纹写成「工作面 30°、非工作面 3°」；GB/T 13576 标题为「锯齿形（3°、30°）螺纹」，第三方资料称承载牙侧 3°、非承载 30°，文章可能写反。未读到标准原文，待核。
- 重算只证明文章内部自洽，不证明与现行标准一致。

## 关联页面

- [Mechtool 在线机械设计工具](../entities/mechtool.md)
- [人形机械布局](./humanoid-mechanical-layout-design.md) · [关节方案选型](./robot-joint-scheme-selection.md) · [关节轴承选型](./robot-joint-bearing-selection.md) · [行星滚柱丝杠](./planetary-roller-screw-humanoid-leg-actuation.md) · [公差仿真工具链](./structure-tolerance-simulation-toolchain.md)

## 参考来源

- [紧固件全接触：常用螺纹基础知识](../../sources/blogs/fastener_quanjiechu_thread_basics_2024-04-10.md)
- [Mechtool 站点目录归档](../../sources/sites/mechtool_cn_catalog_2026-11-07.md)

## 继续阅读

- 现行 GB/T 192 / 193 / 197、GB/T 13576.1-2008、GB/T 3098.1（性能等级）原文（本次未读）。
