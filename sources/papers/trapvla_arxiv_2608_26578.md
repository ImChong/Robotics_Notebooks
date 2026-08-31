# TrapVLA（配置化 VLA 后门攻击）

> 来源归档（ingest）

- **标题：** TrapVLA: Trapping Vision-Language-Action Models in Configured Failure Modes
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.26578>
- **机构：** 中山大学（SYSU）、鹏城实验室、香港大学（HKU）等
- **项目页：** <https://john-liua.github.io/TrapVLA/>
- **入库日期：** 2026-08-31
- **一句话说明：** Configured Failure Trapping：隐蔽文本触发器诱导 **指定** 失败模式（如抓取偏移）；Trap-LIBERO / Trap-RoboTwin 四类失败；TrapVLA 学习触发器诱导的动作残差。

## 核心摘录（MVP）

### 1) 攻击任务升级

- **摘录要点：** 传统后门把任意失败当成功；本文要求控制 **如何失败**（Early Close / Grasp Deviation / Early Open / Release Deviation）。
- **对 wiki 的映射：**
  - [TrapVLA](../../wiki/entities/paper-trapvla.md) — 问题设定。

### 2) TrapEngine + TrapEval

- **摘录要点：** 合成目标轨迹引擎 + 自动化失败忠实度评测；Trap-LIBERO 与 Trap-RoboTwin 基准。
- **对 wiki 的映射：**
  - [TrapVLA](../../wiki/entities/paper-trapvla.md) — 数据与评测。

### 3) 开源状态（截至 2026-08-31）

- **摘录要点：** **未开源**训练/攻击代码。`John-liua/TrapVLA` 仓仅为项目页静态文件（`index.html` / `media`），无可运行实现。
- **对 wiki 的映射：**
  - [TrapVLA 项目页](../sites/john-liua-trapvla.md)
  - [TrapVLA GitHub Pages 仓](../repos/john-liua-trapvla.md)

## 当前提炼状态

- [x] arXiv 摘要与项目页已对齐摘录
- [x] 步骤 2.5：仓内无训练入口
- [x] wiki 映射：`wiki/entities/paper-trapvla.md` 新建
