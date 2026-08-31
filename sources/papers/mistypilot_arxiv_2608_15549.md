# MistyPilot（社交机器人多智能体 LLM 技能编排）

> 来源归档（ingest）

- **标题：** MistyPilot: Enabling Social-Robot Control through Multi-Agent LLM Skill Orchestration
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.15549>
- **机构：** 纽约州立大学布法罗分校（University at Buffalo / SUNY Buffalo）
- **项目页：** <https://wangxiaoshawn.github.io/MistyPilot.html>
- **代码：** <https://github.com/WangXiaoShawn/MistyPilot>
- **入库日期：** 2026-08-31
- **一句话说明：** Task Router 分派物理交互智能体（PIA）与社交对话智能体（SIA）；Misty 真机五组组件测试 + 12 人初步用户研究；可扩展至 100 项技能。

## 核心摘录（MVP）

### 1) 双智能体分工

- **摘录要点：** PIA 处理传感器触发与技能调用；SIA 管理对话状态、多模态响应与结果复用。
- **对 wiki 的映射：**
  - [MistyPilot](../../wiki/entities/paper-mistypilot.md) — 架构。

### 2) 组件级评测

- **摘录要点：** 路由、传感器—技能绑定、任务状态解析、结果复用、技能扩展五套件；真机执行；单智能体基线方差更高。
- **对 wiki 的映射：**
  - [MistyPilot](../../wiki/entities/paper-mistypilot.md) — 评测。

### 3) 开源状态（截至 2026-08-31）

- **摘录要点：** **已开源**。`WangXiaoShawn/MistyPilot` 含 `MistyPilot.py`、`PIA/`、`SIA/`、`requirements.txt` 等可运行入口。
- **对 wiki 的映射：**
  - [MistyPilot 仓库](../repos/wangxiaoshawn-mistypilot.md)
  - [MistyPilot 项目页](../sites/wangxiaoshawn-mistypilot.md)

## 当前提炼状态

- [x] arXiv 摘要与仓库已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-mistypilot.md` 新建
