# Security Audit Skill（cloudflare/security-audit-skill）

> 来源归档

- **标题：** Security Audit Skill
- **类型：** repo（coding-agent skill）
- **来源：** Cloudflare
- **链接：** https://github.com/cloudflare/security-audit-skill
- **博客：** https://blog.cloudflare.com/build-your-own-vulnerability-harness
- **安装：** `npx skills add https://github.com/cloudflare/security-audit-skill --skill security-audit`
- **入库日期：** 2026-09-19
- **协议：** MIT
- **Stars（入库日）：** ~15.8k（GitHub，以克隆时为准）
- **一句话说明：** Cloudflare 开源的 **六阶段安全审计 Agent Skill**：侦察 → 覆盖率驱动狩猎 → 候选对抗验证 → 结构化 `findings.json` → 独立记录复核 → 目标中立报告；产出机器可读且可增量叠加的多轮审计工件。
- **为什么值得保留：** 与本知识库 [Codex Security](../../wiki/entities/codex-security.md)、[Open Code Review](../../wiki/entities/open-code-review.md) 及 [软件安全基础](../../wiki/concepts/software-security-basics.md) 同属 **Agent 驱动 AppSec** 工具位；对遥操作网关、OTA、训练 farm API 等机器人云边代码面的 **可审计漏洞发现管线** 有直接对照价值。
- **沉淀到 wiki：** 是 → [`wiki/entities/cloudflare-security-audit-skill.md`](../wiki/entities/cloudflare-security-audit-skill.md)

## 开源状态

- **已开源** — 完整 skill 文件、`report-schema.json`、零依赖 Node 校验器与测试均在 GitHub；无独立项目页，以 README 与 Cloudflare 博客为准。

## README 要点（归纳）

- **起源：** 该 skill 是 Cloudflare 内部漏洞发现 harness 的单仓起点；博客 [Build your own vulnerability harness](https://blog.cloudflare.com/build-your-own-vulnerability-harness) 描述其演进为多阶段、舰队级系统。
- **六阶段工作流：**
  1. **Reconnaissance** — 映射架构、信任边界、输入面、先验证据与确定性覆盖率 → `architecture.md` + `coverage-ledger.json`。
  2. **Coverage-led hunting** — 按 ledger 单元分配隔离 hunter，coverage critic 找缺口。
  3. **Candidate validation** — 每个唯一候选交给 **全新 verifier** 尝试证伪（发现者与验证者分离）。
  4. **Structured output** — 写入 `findings.json`（`confirmed` / `needs_validation` / `rejected`），对照 `report-schema.json` 校验。
  5. **Independent record verification** — 独立 agent 复核最终 source 声明；实质性替换需再次独立验证。
  6. **Target-neutral reporting** — 从已验证记录与 coverage ledger 派生 `REPORT.md`、`FINDINGS-DETAIL.md`、`NEEDS-VALIDATION.md`。
- **校验门禁：** Phase 1 及每次 ledger 更新后跑 `validate-coverage-ledger.cjs`；Phase 4 与 Phase 5 每次替换后跑 `validate-findings.cjs`。
- **Verdict 语义：** `confirmed` 需完整 source trace 与有界观测结果；`needs_validation` 有精确未决事实且无严重度；`rejected` 为已证伪候选。
- **多轮叠加：** 同一仓库多次运行可增量——利用历史 ledger/findings 补缺口、重验变更源码、携带仍有效的证据。
- **攻击类 prompt 分面：** `ATTACK-CLASSES.md` 及按目标类型拆分的 `AI-AND-LLM.md`、`WEB-PROTOCOL-AND-AUTH.md`、`SUPPLY-CHAIN-AND-RELEASE.md`、`CLOUD-AND-DEPLOYMENT.md` 等。
- **设计原则：** 只确认已建立的边界失败；对抗性验证；严重度需 impact；纵深防御缺口不算漏洞；单轮约只发现重复运行总量的一半漏洞（Cloudflare 测试经验）。
- **依赖：** 支持 tool use 与并行子 agent 的 coding agent；Node.js 跑校验器；目标可控构建/测试需 OS 级沙箱（禁外网、白名单环境、资源限制、仅写 scratch 路径），否则 lead 保持 `needs_validation` 而不执行目标代码。

## 与本站 sources 的其它锚点

- 对照：[Codex Security 归档](./codex-security.md)（OpenAI agent 扫描 + SARIF）
- 对照：[Open Code Review 归档](./open-code-review.md)（确定性 code review 管线）
