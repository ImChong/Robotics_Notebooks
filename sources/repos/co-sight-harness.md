# Co-Sight Harness Studio（ZTE-AICloud）

- **标题：** Co-Sight Harness Studio — 围绕模型的 Agent 运行时控制结构
- **类型：** repo
- **机构：** 中兴通讯（ZTE Corporation / ZTE-AICloud）
- **代码：** <https://github.com/ZTE-AICloud/Co-Sight/tree/master/Co-Sight%20Harness>
- **仓库根：** <https://github.com/ZTE-AICloud/Co-Sight>
- **官方文档：** [中文 README](https://github.com/ZTE-AICloud/Co-Sight/blob/master/Co-Sight%20Harness/README.md) · [快速开始](https://github.com/ZTE-AICloud/Co-Sight/blob/master/Co-Sight%20Harness/QUICKSTART.md)
- **版本：** Harness Studio README 标注 0.1.0；Drive README 标注 2.3.4（子组件独立版本）
- **许可证：** 子项目 `LICENSE` Apache-2.0；第三方依赖与改编代码须遵从各自声明
- **入库日期：** 2026-11-19
- **核对范围：** 仓库根 README；`Co-Sight Harness/README.md`（全文）；`QUICKSTART.md`（全文）；`framework/README.md`（框架启动、配置、模块装配、安全）；`drive/README.md`（phase/DAG、HITL、状态与插件结构）；`package.json`、`framework/server/src/index.ts`、`app.ts`、`routes/module-face.ts`、`routes/modules.ts`、`drive/skills/drive/SKILL.md`、`drive/skills/state-machine/SKILL.md`、`scripts/state_cli.py`（主要流程）、`drive/hooks/hooks.json`、`state_transition_guard.py`（入口和防护机制）。非全部组件源码全覆盖。
- **通读：** 已完成
- **沉淀到 wiki：** 是 → [Co-Sight Harness 实体页](../../wiki/entities/co-sight-harness.md)

## 原文定位与独立性

用户给出的 GitHub URL 是 **Multi-Repo Workspace**，根 README 明确分成三个独立方向：

| 子目录 | 定位 | 与本次 ingest 关系 |
|---|---|---|
| `Co-Sight/` | Manus-like 的 Agent 系统及应用平台 | 不纳入本节点 |
| `Co-Knowledge/` | 私有知识管理与检索工作台 | 外部依赖，不纳入本节点 |
| `Co-Sight Harness/` | 本地优先的多运行时 Agent 宿主、控制与评测工作台 | **本节点唯一对象** |

这不是独立 GitHub 仓库，而是同仓库中的独立子项目；不要把 Co-Sight 主平台、Co-Knowledge 或运行时 DSH 混称为 Harness。

## 官方设计和实际入口

- **四个控制面（官方架构提法）：** 业务（目标/约束/验收/风险）；执行（Orchestrator/Loop/Graph/State/Checkpoint）；资产（Knowledge/Memory/Tool/Skill/Policy）；证据（Trace/Eval/Review/Version/Rollback）。原则是控制数据外置于 LLM。
- **Studio 框架：** `framework/server/src/index.ts` 执行 `loadConfig → createServices → seedPresetExperts → buildApp → app.listen`。 `app.ts` 装配 Fastify 路由；`routes/modules.ts` 根据模块清单注册和暴露模块；前端以模块 iframe 方式挂载子系统。Studio 并不是 Drive 插件的状态机实现。
- **Drive 插件：** `drive/skills/drive/SKILL.md` 从 `startup` 初始化 `.drive/<plan_dir>/`，调用 `read_workflow.py` 选择 `phase` 或 `dag`，通过子 Agent 分析、规划、执行、评测；`state-machine/scripts/state_cli.py` 进行阶段校验/状态写入；`drive/hooks/hooks.json` 注册 `PreToolUse` 与 `SubagentStop` 门禁。
- **状态与门禁：** `state_cli.py` 将 `confirmed_stages` / `confirmed_nodes` 列为受保护字段，`transition` 校验状态合法性、评测 verdict 与人工确认，且要求受信编排器设置 `DRIVE_TRUSTED=1`。默认非 `auto_mode` 需人工确认；DAG 还有节点级 HITL。
- **运行数据：** Drive 将 `executeState.json`、`intent.md`、`manifest.md`、`taskRecord.yaml`、`waves.yaml`、`artifacts/`、`evaluation/` 和 `run-log.jsonl` 放在 `.drive/<plan_dir>/`。

## 开源与复现状态（2026-11-19）

**已开源源码与运行入口。** 目录包含 `framework/`、`drive/`、`agent-genesis/`、`agent-evaluator/`、`skill-optimizer/`、`cosight-bot/`、`cosight-viewer/`、`smart-factory/`、`dsh-compat/`；根 `package.json` 提供开发、构建、打包与测试命令。并非已经独立验证所有模块可运行。

源码开发所需 Node.js `^22.19.0 || >=24.0.0`、pnpm `11.28.5`，进入 **子目录** 后：

```bash
git clone https://github.com/ZTE-AICloud/Co-Sight.git
cd "Co-Sight/Co-Sight Harness"
corepack enable
pnpm install
pnpm dev             # 后端 4300 + 前端 4310
# pnpm dev:all        # 先为 cosight-viewer / smart-factory 分别装依赖
# pnpm verify         # 官方自测：typecheck + oxlint + tests
```

分发包路径（仅在实际取得官方包后）：`node start.mjs --self-check` → `./install.sh` → `./start.sh`；源码路径与分发包路径不是一套命令。

**未在本次环境运行：** `pnpm install`、`pnpm dev`、`pnpm verify`、端到端工作流、打包 smoke test。README 的业务提升数字为**项目方自述**，不是本次独立实验。

## 安全边界及工程判断

1. `COSIGHT_HOST` 默认 `0.0.0.0`，而 `index.ts` 明示框架**没有登录鉴权**且存储模型凭据；本机使用应绑定 `127.0.0.1`，公网使用需另做鉴权、隔离和网络访问控制。
2. `routes/module-face.ts` 的 `/api/framework/v1/*` 使用回环 IP/Host、Origin 校验与模块令牌；令牌可访问敏感模型配置与工作区，且同 UID 进程可读取本地凭据，所以它**不是宿主机安全隔离边界**。
3. Harness 的 Hook / Gate 管理 Agent 工作流，不等于物理机器人的实时安全回路；不能代替 ROS 2 控制器、设备互锁、急停、现场验收。
4. 自报“完成率 50%→100%”“F1 68%→91.25%”等没有在本次核查中获得可独立复核的数据集、样本量和评测脚本结果；不得写作通用能力保证。

## 与 wiki 的映射

- [Co-Sight Harness](../../wiki/entities/co-sight-harness.md)
- [Agnes Harness](../../wiki/entities/agnes-harness.md)
- [Agent Lightning](../../wiki/entities/agent-lightning.md)
- [DeepSeek Harness](../../wiki/entities/deepseek-harness.md)
