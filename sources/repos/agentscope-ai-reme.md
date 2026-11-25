# AgentScope ReMe 官方代码仓库（论文 ReMe 的后续产品实现）

> 来源归档（repo / 官方一手 README、入口代码和运行配置）

- **仓库：** <https://github.com/agentscope-ai/ReMe>
- **官方文档：** <https://reme.agentscope.io/>
- **相关论文：** [Remember Me, Refine Me（arXiv:2512.10696）](https://arxiv.org/abs/2512.10696)，Findings of ACL 2026
- **许可：** Apache-2.0（主分支 README 和 LICENSE）
- **开放程度：** 公开完整框架源码，支持安装、服务、文件记忆、搜索及整理；**当前 main 架构已相对 2025 论文实验演进**，不可直接等同论文的 Table 1 实验代码。
- **核查日期：** 2026-11-23
- **核查范围：** 主分支 README；pyproject.toml 中的 project.scripts；reme/reme.py 的 CLI main、_start_application、call_server；reme/core/application.py 的组件/Job 装配；reme/config/default.yaml 的 auto_memory、auto_dream、索引 watcher 任务配置；历史 reme_v3 README 用于确认版本演进。

## 与原论文的关系及边界

论文描述的实验算法为：对多条执行轨迹蒸馏结构化程序性经验，经场景向量检索，动态选择性写入并依检索效用删除。主分支当前 README 则明确宣称 **Memory as File, File as Memory**，采用 session / resource → daily → digest 的 Markdown 记忆生命周期，支持 BM25、可选 embedding、auto_memory / auto_dream 等命令。二者都属于 ReMe 项目，但功能组织和复现入口已有变化；**不能据当前 main 推断论文删除阈值/任务采样代码仍按原实验实现**。

## 已核实的运行入口（当前 main）

1. Python >=3.11：安装 pip 包 `pip install "reme-ai[core]"`；若从源码安装参考 README 的可编辑安装与可选 Studio 构建。
2. pyproject.toml 声明 `reme = "reme.reme:main"`。执行 `reme start` 时，`reme/reme.py:main` 调用 `_start_application`，由 `Application` 初始化 workspace、service、components、jobs。
3. `reme auto_memory` 被 CLI `call_server` 送入已运行服务的 job；default.yaml 的 `auto_memory_step` 负责写日记忆，`auto_tag_step` 补标签；后台索引 watcher 读取 daily/digest 的文件变化更新检索索引。
4. `reme auto_dream` 的默认配置依次执行 `dream_extract_step`、`dream_integrate_step`、`dream_finish_step`、`auto_tag_step`，将日记忆整理入 digest。
5. `reme search query="..."`、`reme read path="..."` 用于按需召回，embedding 检索需在配置中显式启用；基础文本文件与 BM25 可在无 LLM 凭证下使用。

## 复现实验前需额外核对

- 找出对应论文时期的 commit/tag/benchmark 入口（论文与当前 main 未提供可直接运行的 BFCL-V3 / AppWorld 一键复现说明）；锁定模型、数据划分、embedding 与 Elasticsearch。
- 研究代码的 ReMe(fixed/dynamic) 与今天文件原生的 auto_dream 并非同一个接口，应分开评测。
- 所有 API Key、数据依赖与服务端口从官方文档配置；无实测运行记录，当前仅完成静态源码阅读。

## 关联来源

- [论文原文归档](../papers/reme_procedural_memory_arxiv_2512_10696.md)
- [论文详情](../../wiki/entities/paper-reme-procedural-memory.md)
