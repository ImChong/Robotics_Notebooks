# HUMEMBR（samirahuber/humembr）

> 来源归档（repo）

- **标题：** HUMEMBR — Learning Human Routines for Predictive Embodied Navigation
- **类型：** repo / embodied-qa / long-horizon-memory / spot / graphnav / llm-agent
- **来源：** SamiraHuber（GitHub）
- **链接：** <https://github.com/samirahuber/humembr>
- **项目页：** <https://samirahuber.github.io/humembr/>
- **论文：** [arXiv:2606.30404](https://arxiv.org/abs/2606.30404) · IROS 2026
- **Stars：** ~1（2026-08-06）
- **入库日期：** 2026-08-06
- **一句话说明：** HUMEMBR 官方实现：Spot + GraphNav 采集与导航、PostgreSQL/pgvector 记忆库、人脸/KPR ReID 聚类、Qwen/Gemini caption+agent 工具调用；README 提供 uv 安装与三进程启动；**COBD 数据集 archive 暂不公开**。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-humembr.md`](../../wiki/entities/paper-humembr.md)

---

## 开源状态（步骤 2.5）

| 项 | 状态（2026-08-06） |
|----|-------------------|
| 训练 / 部署代码 | **已开源**（`src/humembr/`，可 `uv sync` 后启动） |
| PersonEQA 评测脚本 | 有 `src/humembr/eval/question/`、`eval/` 日志产物 |
| 聚类评测 | `src/humembr/eval/person/` + `eval/clustered_results/` |
| KPR 权重 | Google Drive 外链，需放到 `src/humembr/processing/pretrained` |
| COBD 数据集 | **private**（README 明示不可公开下载） |
| 许可证 | 仓库未声明 SPDX license（截至核查日） |

**结论：** **已开源可运行代码栈**；完整论文数据复现受数据集私有限制。wiki「源码运行时序图」对齐 README 三进程入口。

---

## README 宣称的技术栈

| 组件 | 文案 / 路径 |
|------|-------------|
| 包管理 | `uv` + Python **3.10.3** |
| 记忆库 | PostgreSQL + **pgvector**（Docker `pgvector/pgvector`，端口 5434） |
| 迁移 | `dbmate` → `src/humembr/db/migrations/` |
| 低层导航 | Boston Dynamics **GraphNav**（`robot/navigation/graph_nav_interface.py`） |
| Caption | vLLM / Ollama / OpenAI 兼容（`processing/qwen_openai.py` / `qwen.py`） |
| Agent | `agent/llm_agent.py`（真机）、`agent/interview_agent.py`（问答评测）、`agent/tools.py` |
| Web UI | `server/app.py` → `http://127.0.0.1:5050/` |
| 感知 | YOLO、InsightFace、KPR ReID、ResNet 去冗余、mxbai embedding |

---

## 目录快照（关键入口）

```
humembr/
  README.md
  pyproject.toml
  uv.lock
  src/humembr/
    robot/main.py              # Terminal 1：机器人控制 / 采集
    server/app.py              # Terminal 2：Web 聊天 + agent
    processing/qwen*.py        # Terminal 3：字幕服务客户端
    agent/{llm_agent,interview_agent,tools}.py
    processing/person_processor.py
    db/migrations/             # pgvector schema
    eval/{person,question}/    # 聚类与 PersonEQA 评测
  eval/                        # 聚类指标与 live 任务 HTML 日志
```

## 对 wiki 的映射

- [`wiki/entities/paper-humembr.md`](../../wiki/entities/paper-humembr.md)
- [`sources/papers/humembr_arxiv_2606_30404.md`](../papers/humembr_arxiv_2606_30404.md)
- [`sources/sites/samirahuber-humembr-github-io.md`](../sites/samirahuber-humembr-github-io.md)
