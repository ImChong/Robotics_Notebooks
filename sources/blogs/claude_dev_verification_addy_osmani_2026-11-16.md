# How the Claude Code team thinks about verification / Claude Code 团队如何看待验证

> 一手来源归档：Claude Code 官方工程博客，结构化摘录与核对记录（非原文转载）。

- **类型：** blog
- **标题：** How the Claude Code team thinks about verification
- **中文标题：** Claude Code 团队如何看待验证
- **作者：** Addy Osmani
- **发布机构：** Claude Code / Anthropic 工程团队
- **发布日期：** 2026-11-16
- **一手链接：** https://claude.dev/blog/how-the-claude-code-team-thinks-about-verification/
- **版本：** 2026-11-16 公开博客发布版；2026-11-17 核对
- **核对范围：** 通读网页全部正文（引言；Not all tests are created equal；E2E / Property / Differential tests；How the Claude Code team thinks about verification；Good tests, good results），核对文中团队内部流程、数字、测试类型与条件限制。
- **通读：** 已完成
- **沉淀到 wiki：** [Coding Agent 验证闭环](../../wiki/methods/coding-agent-verification-loops.md)
- **代码开放状态：** 这是工程实践博文而非独立开源软件项目；正文未提供可归属于本文的可运行代码仓库或测试套件，故没有可按源码核验的运行时序图。

## 原文核心观点

**目标不是让 agent 自评“完成”，而是让它交付可复核的行为证据。** 作者称 Claude Code 团队约 90% 的代码由 Claude 编写并审阅；这是一项团队自身描述，不是公共基准。文章还引用内部结构化文件生成测试：独立 grader 按 rubric 检查，相较标准提示循环最多提高约 10 个百分点，不能外推至所有编码任务。

### 三类值得投入的验证

| 类型 | 原文用途 | 易错边界 |
| --- | --- | --- |
| E2E（End-to-End） | 启动真实应用、执行用户路径，提供截图 / trace / 失败步骤；作者建议优先投资 | 慢、易 flaky；需可靠的可重复启动方式 |
| Property-based | 用生成输入反证往返、幂等、不变量、等价等性质 | 属性本身必须真实；不能只测 agent 也熟悉的样例 |
| Differential | 同一输入分别运行旧系统和新系统，比对输出，适合行为保持型迁移 | 需归一化无关差异，并列明经批准的行为变化 |

### 团队验证机制（原文陈述）

1. **真实运行：** Web 可用浏览器/CDP；移动端用模拟器；桌面端计算机操作；后端本地运行。推荐将验证步骤放在仓库内的 .claude/skills/verify 中形成 smoke-test harness。
2. **失败升级：** Claude 尝试修复失败项；同一问题三轮仍不过，转人工处理。
3. **所有者维护门禁：** 组件 owner 自己定义检查；团队维护 39 项 reliability checklist，以 diff 触发通常 3–10 条，而非全部盲跑。
4. **评审上下文连续：** 单个主要 reviewer 结合 REVIEW.md、既往失败和多轮结果审查；不是单纯堆叠更多模型。
5. **被检查的变更不能控制检查：** REVIEW.md 从 base/default branch 读取；评审绑定确定的 commit SHA；便宜的检查和本地 build/lint 先跑，再占用共享 CI/评审资源。
6. **门禁循序上线：** off → observe → enforce；提交记录写入验证标记，保留未验证解释路径，观察误拦截后再强制。
7. **评测验证器：** 用历史事故 / revert 构造可复现实例，固定测试集并用稳定合并的变更控制误报；分别看 raw finding / post-verification / final review 的召回与重要性。
8. **失败案例沉淀：** 将至少在两个 PR 出现、且能从 diff 判断的模式提炼为候选规则，并由 agent 反证、人审定稿；不要靠无限 retry / sleep 掩盖 flaky test；定期删去失效规则。

## 工程边界

- 文中 90%、10 个百分点、39 项、3–10 项等均为作者对团队内部实践的叙述，**不代表本仓库运行结果或外部独立验证**。
- “运行应用比单测更重要”是作者对验证优先级的工程判断；不等于单元、集成、安全或形式化测试可以省略。
- 本文只涉及软件编码 agent；迁移到机器人控制系统时，还需独立验证仿真/真机安全、时延、故障模式和部署审批。
- 未在本文找到可直接复现的实验数据集、测试脚本或源码入口；不虚构运行时序、效果或 CI 通过情况。

## 与已有知识的关系

- [多 Agent 编码管理与认知把关](../../wiki/concepts/coding-agent-orchestration-and-cognitive-oversight.md)：多 agent 调度、人类责任和审阅能力。
- [Coding Agent 验证闭环](../../wiki/methods/coding-agent-verification-loops.md)：将本文的验证机制整理成可执行的方法页。
- [Agentic Coding 时代的软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)：验证与软件系统设计、生产环境约束之间的关系。

## 参考原文

- [How the Claude Code team thinks about verification — Claude Code Blog，2026-11-16](https://claude.dev/blog/how-the-claude-code-team-thinks-about-verification/)
- [How the Claude Code team does code review — Claude Code Blog，2026-10-15](https://claude.dev/blog/solving-code-review-at-anthropic/)（相关官方背景，非本文替代来源）
