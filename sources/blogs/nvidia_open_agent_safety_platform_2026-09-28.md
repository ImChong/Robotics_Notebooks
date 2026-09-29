# NVIDIA Open Agent Safety Platform: A Reference for Continuous In-Silicon Agent Monitoring

> 来源归档（blog / NVIDIA Developer Blog）

- **标题：** NVIDIA Open Agent Safety Platform: A Reference for Continuous In-Silicon Agent Monitoring
- **类型：** blog
- **作者：** John Myers, Alex Watson, Ali Golshan, Ofir Arkin（NVIDIA）
- **原始链接：** https://developer.nvidia.com/blog/nvidia-open-agent-safety-platform-a-reference-for-continuous-in-silicon-agent-monitoring/
- **发布日期：** 2026-09-28
- **关联代码：** https://github.com/NVIDIA/OpenShell（Apache 2.0 运行时；平台另含 **NVIDIA Sentry** 与 BlueField-4 硅内执行，非单一 GitHub 仓）
- **技术 walkthrough：** https://developer.nvidia.com/blog/add-runtime-controls-to-ai-agents-with-nvidia-openshell/
- **入库日期：** 2026-09-29
- **一句话说明：** 提出 **Open Agent Safety Platform** 参考架构：**OpenShell** 在 Vera CPU 上编排沙箱与可验证策略，**NVIDIA Sentry** 在 BlueField-4 DPU 上带外观测与线速策略执行，形成软件 + 硬件分层 agent 安全层。

## 核心摘录（归纳，非全文）

### 动机

- 前沿 lab 报告 agent **突破评测沙箱**、访问未授权系统、甚至 **误报行为**；根因常是工具 + 长时间运行 + 模糊指令的组合，而非单一新能力。
- 类比 90 年代互联网：**浏览器标签沙箱** 使电商/社交成为可能；agent 需要类似的 **独立安全控制**，不能仅靠开发者「承诺守规矩」。

### 五条设计原则

1. **Policy 可验证** — 运行前 prover 证明策略不超出操作者意图。
2. **带外执行（out-of-band）** — 控制不在 agent 可达范围内；agent 不必知晓被监视。
3. **控制通往模型的路径** — agent 无「下一 thought」则无法行动；该路径是最佳观测点与 **kill switch**。
4. **权限与推理可见性同比例扩展** — 能力越大，越需 inspect thinking；开放权重模型下 reasoning/activation 更可审计。
5. **共享责任模型** — lab / 企业 / 硬件供应商各守一层；运行时与策略语言应 **开放** 以便多厂商接入。

### 三层栈

| 层 | 职责 |
|----|------|
| **Application** | 用户构建：模型、harness、工具、数据、脚本 |
| **Runtime** | 将应用投影到基础设施；编排 workload；**持续监控 + 实时策略** |
| **Infrastructure** | 网络、存储、通用算力、加速算力（含安全监控与密度） |

### OpenShell（软件运行时）

- **Apache 2.0** 开源；内核级隔离沙箱执行自主 agent。
- 操作者声明文件/网络/工具/进程/凭证边界 → **运行前检查 + 运行中强制**。
- 博客强调 **drift**：agent 偏离任务/约束（策略拦截、bug、缺工具、歧义指令、长时试错）——不能指望 agent **完全自治** 约束自身。

### NVIDIA Sentry + BlueField-4（硬件增强层）

- **可选** 独立层：经 **NVIDIA DOCA**  programmable，与 OpenShell 策略关联。
- 关联 agent 交互、策略决策、工具/数据访问 → **上下文活动记录**；DOCA gateway 补充 **身份治理**（持续验证 agent 身份与委派权限）。
- **Vera Rubin POD**：每 compute tray 的 BlueField-4 位于 **节点通往模型的唯一路径** → 带外、线速、实时策略；与 host 隔离，host 不可信时仍可作为基础设施保护层。
- 已在 Vera + BlueField-4 上的组织：**软件更新即可启用**（博客表述）。

### 生态

- 博客配图列举应用、模型、基础设施、芯片、能源等多类厂商支持；邀请 frontier lab、开发者、基础设施提供商共建。

## 对 wiki 的映射

- [NVIDIA OpenShell](../../wiki/entities/nvidia-openshell.md) — 开源沙箱运行时、策略 YAML、formal prover 与 Gateway/Supervisor 分工
- [NVIDIA Open Agent Safety Platform](../../wiki/entities/nvidia-open-agent-safety-platform.md) — OpenShell + Sentry + Vera/BlueField 参考设计与五条原则
- [Agentic Coding 时代的软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md) — 「安全可靠」与带外控制互补
- [Agent Reach](../../wiki/entities/agent-reach.md) — 代理外网能力脚手架；与 **运行时信任边界** 正交
