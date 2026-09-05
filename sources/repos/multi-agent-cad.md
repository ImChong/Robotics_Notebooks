# Multi-Agent-CAD（MAC / Pan-Chera/Multi-Agent-CAD）

> 来源归档

- **标题：** MAC (Multi-Agent CAD) — A decoupled multi-agent framework for text-to-CAD generation via constrained test-time compute
- **类型：** repo / text-to-cad / llm-agent / build123d
- **作者：** Guanxing Qu、Xueyan Zou
- **机构：** 清华大学（Tsinghua）· Interactive Embodied Intelligence Lab（IEI Lab）
- **链接：** <https://github.com/Pan-Chera/Multi-Agent-CAD>
- **克隆：** `https://github.com/Pan-Chera/Multi-Agent-CAD.git`
- **Stars / Forks：** 930★ / 84（2026-09-05 GitHub API）
- **许可：** **MIT**（根目录 `LICENSE`；vendored `packages/cadpy` 亦 MIT，源自 CAD Skills）
- **项目页：** 无独立 `*.github.io` 项目页；GitHub README + `multi_agent_cad/WORKFLOW.md` 为入口。实验室页 [IEI Lab](https://maureenzou.github.io/lab.html) 未单列本仓下载链。
- **论文：** 无 arXiv；README 给 `@misc{mac2026}` GitHub 引用
- **入库日期：** 2026-09-05
- **一句话说明：** 四段 LangGraph 多智能体把自然语言压成 `CADBrief` / `ArchitectPlan` JSON，再用确定性翻译器 + Aider 修成 **build123d → STEP**；在 CAD Skills 的 10 prompt / 141 特征基准上自报 **116× 更少 token、13× 更低费用、特征通过率 99.3%**。
- **为什么值得保留：** 直接对照本库已有 [CAD Skills](../../wiki/entities/cad-skills.md)（同一套 P1–P10 prompt）。核心判断不是「又一个写 build123d 的 agent」，而是 **结构化状态传递 + 确定性翻译器** 把测试时算力从对话重放里抠出来。机器人夹具草稿可抄这条「少 token、可审计中间件」的组织方式。
- **开源状态（2026-09-05 核查）：** **已开源** — 训练不涉及；推理/编排代码完整（`python -m multi_agent_cad.graph` / Web UI）。需自备 OpenAI 兼容 API Key（默认阿里云 DashScope `qwen3.7-max`）。无权重发布。
- **沉淀到 wiki：** 是 → [`wiki/entities/multi-agent-cad.md`](../../wiki/entities/multi-agent-cad.md)

## 仓库概况（2026-09-05 GitHub API / README）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub（`Pan-Chera/Multi-Agent-CAD`） |
| 默认分支 | `main` |
| 创建 / 最近推送 | 2026-07-30 / 2026-08-13 |
| Topics | `text-to-cad`, `langgraph`, `build123d`, `multi-agent`, `aider`, `opencascade`, `qwen`, `step` |
| 语言 | Python 3.11 |
| Homepage | 空 |
| 讨论 | Discussions 已开；Issues 1 |

## README 入口（归纳）

| 组件 | 路径 / 命令 |
|------|-------------|
| 环境 | `conda env create -f environment.yml`；`pip install --no-deps aider-chat==0.82.3`（避开 numpy 1.x pin） |
| 配置 | `multi_agent_cad/config.py`；`DASHSCOPE_API_KEY` 优先于文件内密钥 |
| 终端 | `python -m multi_agent_cad.graph`；改已有设计 `graph_aider` |
| Web | `pip install -e ".[web]"` → `python -m multi_agent_cad.web`（`:8000`；生成 `.py` **在服务端执行**） |
| 缓存 | `pipeline_cache/cad_brief.json` + `architect_plan.json`（只看文件是否存在，**不校验 prompt**） |
| 工作流 | `multi_agent_cad/WORKFLOW.md` |
| 数字 | `docs/quantified_quality.md`、`docs/qwen3.7_token.md` |
| 基准 prompt | 与 [earthtojake/text-to-cad `benchmarks/`](https://github.com/earthtojake/text-to-cad/tree/main/benchmarks) 共用 P1–P10 |

### 四段智能体

| 阶段 | 角色 | 输入 → 输出 |
|------|------|-------------|
| 1 | Spec Planner | 自然语言 → `CADBrief` JSON（三类验证目标） |
| 2 | Geometric Architect | `CADBrief` → `ArchitectPlan`（草图 / 步骤 / 选择器） |
| 3 | Python Coder | `ArchitectPlan` → `temp_design_*.py`（`_plan_to_code` 确定性翻译优先，Aider 兜底） |
| 4 | Autonomous Skill Loop | 代码 + STEP/STL → 双引擎 QA + Aider 修复（≤5 次） |

### 自报基准（Qwen 3.7-max，相对 CAD Skills 单 agent）

| 指标 | CAD Skills | MAC |
|------|------------|-----|
| Tokens | 103.9M | 896k（116×↓） |
| 费用 | ¥125.69 | ¥9.67（13×↓） |
| API 次数 | 1,307 | 50（26×↓） |
| 特征通过 | 97.9%（138/141） | 99.3%（140/141） |

墙钟「约 10×」README 标明为量级估计，非正式计时。

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 主实体 | [`wiki/entities/multi-agent-cad.md`](../../wiki/entities/multi-agent-cad.md) |
| 对照基线 | [`wiki/entities/cad-skills.md`](../../wiki/entities/cad-skills.md) |
| 概念总览 | [`wiki/concepts/text-to-cad.md`](../../wiki/concepts/text-to-cad.md) |
| 工具索引 | [`sources/sites/text-to-cad-tools.md`](../sites/text-to-cad-tools.md) |
| 桌面对照 | [`wiki/entities/freecad-mcp.md`](../../wiki/entities/freecad-mcp.md) |

## 参考链接

- 仓库：<https://github.com/Pan-Chera/Multi-Agent-CAD>
- 工作流：<https://github.com/Pan-Chera/Multi-Agent-CAD/blob/main/multi_agent_cad/WORKFLOW.md>
- 量化说明：<https://github.com/Pan-Chera/Multi-Agent-CAD/blob/main/docs/quantified_quality.md>
- IEI Lab：<https://maureenzou.github.io/lab.html>
- 对照仓：[earthtojake/text-to-cad](https://github.com/earthtojake/text-to-cad)
- 内核：[gumyr/build123d](https://github.com/gumyr/build123d)
