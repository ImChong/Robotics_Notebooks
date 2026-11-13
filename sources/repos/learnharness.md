# LearnHarness：用于深度学习的 Agent Harness（官方源码）

> 来源归档（ingest）

- **标题：** LearnHarness — Agent harness for learning difficult things
- **类型：** repo / agentic tutoring / adaptive learning / knowledge graph / desktop app
- **作者：** Amos Blomqvist（GitHub: amosblomqvist）
- **官网：** <https://learnharness.ai/>
- **代码：** <https://github.com/amosblomqvist/learnharness>（**已开源**）
- **许可：** AGPL-3.0（以仓库 LICENSE 为准）
- **固定核验版本：** [cb25c8e](https://github.com/amosblomqvist/learnharness/tree/cb25c8e49b85eed9e78315584731cad73a0118a6)
- **平台：** macOS（Apple Silicon）、Windows x64、Linux x64；Electron / Next.js
- **入库与核验日期：** 2026-11-13
- **通读：** 已完成
- **核对范围：** 已完整读取官方 README 与 `docs/getting-started.md`，并核对 `docs/architecture.md` 中的代码边界、教学、知识图谱与本地存储章节，以及 `src/agent/teacher-agent.ts`、`src/agent/tools/probe.ts`、`src/agent/tools/plan.ts`、`src/agent/tools/practice.ts` 和 `src/agent/subagents/extraction.ts` 的实际入口。尚未安装桌面包，也未用真实模型跑端到端教学。
- **一句话说明：** 把知识前置依赖诊断、生成课程依赖图、预测式讲解、独立练习与跨会话命题推理图连接起来的本地优先 AI 教学应用。

## 官方能力核对

1. **Measure / Probe：** Prober 通过问题、信心陈述与自由推理说明判断学习边界；`probe_done` 将评估交给 Teacher。
2. **Plan：** Teacher 的 `plan_init` 根据测得起点写入 Mermaid 依赖 DAG；这是教学计划，不等于永久知识图谱。
3. **Teach：** `quiz` / `explain` 等工具以 plan 的 `nodeId` 标记当前教学步骤，先让学习者预测、再解释与检查理解。
4. **Practice：** 独立的 Practice session 使用 `problem` 与 `conclude`，用于变化题面下的提取练习，而非只是教学回合里的打分。
5. **Graph / Memory：** 教师讲解文本或导入笔记经 `extractor → identity auditor → deterministic merge` 转成 proposition / reasoning chain；保存的是命题及推理关系，不是“掌握程度分数”。
6. **输入：** 桌面端可附加本地 PDF/照片/其他文件路径；另有 `Extract files` 结构化图谱导入，二者不是同一流程；语音听写是可选的 Deepgram 服务。
7. **模型：** 支持用户自备 API key、部分订阅连接、Claude Code CLI 或 OpenAI-compatible 自定义/本地 endpoint。模型和外部语音/搜索服务的调用可能产生费用或发送相关内容。
8. **存储：** 会话 `chats/*.json`、命题与推理链 `graph/*.jsonl`、设置与认证文件本地保存。**不等于离线运行或经过验证的学习成效。**

## 源码主路径

| 路径 | 作用 |
|------|------|
| `app/api/chat/route.ts` / `src/server/chat-handler.ts` | 接收回合、持久化输入、选择 Agent、流式结果 |
| `src/agent/teacher-agent.ts` | 构建 Prober / Teacher / Practice 三类 Agent 和工具集 |
| `src/agent/tools/probe.ts` | `probe_done`，生成并投递探测到教学阶段的接力 |
| `src/agent/tools/plan.ts` | `plan_init`：解析、校验、保存依赖图 |
| `src/agent/tools/practice.ts` | `problem`、`conclude` 与单题在场校验 |
| `src/agent/subagents/extraction.ts` | 从讲解/导入材料提取命题和推理链、审计同一性、合并 |
| `src/core/graph.ts`、`src/store/` | 命题推理图派生与本地数据落盘 |

## 官方开发命令（README）

```bash
npm ci
npm run plugins:setup
npm run dev
npm test
```

- 此为上游文档提供的本地开发入口，**未在本次环境执行**；不据此声明运行成功。
- 原生发布二进制、Windows SmartScreen 与密钥存储细节以官方 Releases 和 Getting Started 为准。

## 已知边界

- 自动图谱提取默认累计足够未覆盖的教师讲解文本后才触发；Practice 成绩不是图谱掌握度。
- 扫描型 PDF 的图谱文件导入**不带 OCR**；不要混淆桌面“附加路径”和“抽取进图谱”两种操作。
- 凭官方 README、架构与源码只能证实设计与实现存在，无法推导其教学效果在对照实验中更优。

## 对 wiki 的映射

- [LearnHarness 实体页](../../wiki/entities/learnharness.md)
- [LearnHarness 官方站归档](../sites/learnharness-ai.md)
- [Get It. Study Companion](../../wiki/entities/get-it-study-companion.md) 与 [DeepTutor](../../wiki/entities/deeptutor.md) 供读者做产品机制对比。

## 主要原始资料

- [README（固定版本）](https://github.com/amosblomqvist/learnharness/blob/cb25c8e49b85eed9e78315584731cad73a0118a6/README.md)
- [Getting Started（固定版本）](https://github.com/amosblomqvist/learnharness/blob/cb25c8e49b85eed9e78315584731cad73a0118a6/docs/getting-started.md)
- [Architecture（固定版本）](https://github.com/amosblomqvist/learnharness/blob/cb25c8e49b85eed9e78315584731cad73a0118a6/docs/architecture.md)
- [teacher-agent.ts](https://github.com/amosblomqvist/learnharness/blob/cb25c8e49b85eed9e78315584731cad73a0118a6/src/agent/teacher-agent.ts)
- [extraction.ts](https://github.com/amosblomqvist/learnharness/blob/cb25c8e49b85eed9e78315584731cad73a0118a6/src/agent/subagents/extraction.ts)
