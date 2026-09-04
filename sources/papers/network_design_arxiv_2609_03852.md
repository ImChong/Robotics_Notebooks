# Design Principles for Reproducible Networks（arXiv:2609.03852）

> 来源归档（ingest）

- **标题：** Design Principles for Reproducible Networks
- **简称：** Network Design / Unigraphical Design
- **类型：** paper / network-science / reproducible-assembly / hardware
- **arXiv：** <https://arxiv.org/abs/2609.03852>
- **PDF：** <https://arxiv.org/pdf/2609.03852>
- **代码：** <https://github.com/Barabasi-Lab/NetworkDesign> — 归档见 [`sources/repos/barabasi-networkdesign.md`](../repos/barabasi-networkdesign.md)
- **机构：** 中欧大学（CEU）、美国东北大学（Northeastern University）Barabási Lab
- **入库日期：** 2026-09-04
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_9_papers_open_source_2026-09-04.md)
- **一句话说明：** 把构件局部约束编码为 design set，用 Unigraphical Design Theorem 判断何时能唯一装配；否则用时间顺序做 guided assembly。

## 开源状态（步骤 2.5，2026-09-04）

| 组件 | 状态 |
|------|------|
| GitHub | **已开源** `Barabasi-Lab/NetworkDesign`：`Data/` + `UnigraphCheck/` |
| LICENSE / 根 README | **无** |

**结论：已开源** — 可运行 `unigraphicality.py` / `numerical_unigraphicality.py`；无许可证声明，商用需自行确认。

## 核心摘录

### 摘录 1：两条可复现路径

- unigraphical assembly：约束已保证唯一结构。
- guided assembly：时间顺序把构造拆成可唯一装配的步骤。
- 应用于 3618 个可复现系统（蛋白复合体、分子、机器人），并用 3D 打印组件验证。

**对 wiki 的映射：** [paper-network-design-reproducible](../../wiki/entities/paper-network-design-reproducible.md)

## 当前提炼状态

- [x] 仓库核查（2026-09-04）
- [x] wiki 映射：`wiki/entities/paper-network-design-reproducible.md`
