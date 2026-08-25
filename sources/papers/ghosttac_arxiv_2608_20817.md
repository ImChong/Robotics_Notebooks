# GhostTac（arXiv:2608.20817）

> 来源归档（ingest）

- **标题：** GhostTac: Manipulating Tactile Sensors without Physical Contact
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2608.20817>
  - <https://ghosttac.github.io/GhostTacCCS.io/>
- **代码：** <https://github.com/GhostTac/GhostTac_CCS>
- **机构：** 浙江大学（ZJU）；香港科技大学（HKUST）
- **会议：** ACM CCS 2026
- **入库日期：** 2026-08-25
- **一句话说明：** 首次展示针对机器人触觉的非接触 EMI 攻击：利用非线性整流与有限带宽放大产生持续 DC 偏移，在 10 个传感模块、2 只灵巧手、15 种触觉传感器上验证，并演示抓取/滑移/材料分类案例影响。

## 核心摘录（MVP）

### 1) 触觉物理层安全空白

- **摘录要点：** 触觉是抓取、滑移检测与材料分类的关键，但物理层安全研究不足；软件层防御无法覆盖传感器前端被操纵。
- **对 wiki 的映射：**
  - [GhostTac](../../wiki/entities/paper-ghosttac.md) — 威胁模型。
  - [tactile-sensing](../../wiki/concepts/tactile-sensing.md) — 传感栈语境。

### 2) EMI → 持续偏移机制

- **摘录要点：** 精心构造 EMI 经耦合、整流、放大绕过板载滤波，形成可控空间分布与幅值的测量偏差，可实现正向（掉落）或负向（过握）干扰。
- **对 wiki 的映射：**
  - [GhostTac](../../wiki/entities/paper-ghosttac.md) — 电路层机理。

### 3) 跨设备与任务案例

- **摘录要点：** 覆盖 15 种触觉传感器类型；案例含医疗瓶抓取（过握/掉落）、滑移误报/抑制、材料分类测量扰动；支持预置与路过攻击场景。
- **对 wiki 的映射：**
  - [GhostTac](../../wiki/entities/paper-ghosttac.md) — 案例研究。

### 4) 开源状态（截至 2026-08-25）

- **摘录要点：** **已开源** — `GhostTac/GhostTac_CCS` 含 Franka + Inspire 灵巧手上的闭环抓取、滑移检测、材料分类演示代码与 `demo/` 视频。
- **对 wiki 的映射：**
  - [ghosttac-ccs](../../sources/repos/ghosttac-ccs.md) — 仓库布局。

## 当前提炼状态

- [x] arXiv + 项目页 + GitHub README 已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-ghosttac.md` 新建
