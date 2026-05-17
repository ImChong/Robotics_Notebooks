# Superpowers 发布文（Jesse Vincent / fsck.com）

> 来源归档

- **标题：** Superpowers
- **类型：** blog
- **作者：** Jesse Vincent（obra）
- **链接：** https://blog.fsck.com/2025/10/09/superpowers/
- **入库日期：** 2026-05-17
- **一句话说明：** 阐述 Superpowers 的由来：将个人使用编码代理的流程产品化为 **skills + 启动 hook**；默认 **brainstorm → plan → implement**；git **worktree** 隔离并行任务；**子代理**实现与评审；强调 **skills 是可组合、可检索、可测试的规约资产**。
- **沉淀到 wiki：** 间接支撑 [`wiki/entities/superpowers-obra.md`](../wiki/entities/superpowers-obra.md) 中与「技能即能力边界」相关的归纳。

## 文中要点（归纳，非全文）

- **动机：** Anthropic 为 Claude Code 推出插件系统后，将已演进的代理流程固化为可分发包；启动时注入 hook，引导代理先读 `getting-started/SKILL.md`，并确立「若存在对应 skill 则必须使用」的约束。
- **工作流差异：** 与「人类当 PM 开双会话」相对的新路径：按任务派发 **subagent**，逐项实现并接受 **code review**；结束时可选择 PR、合并 worktree 等。
- **Skills 元叙事：** 与 Microsoft Amplifier、自改进代理写 `SKILL.md` 等实践并列讨论；作者用 **子代理压力场景** 验证技能可读性与遵从度。
- **未完成方向：** 跨用户 **sharing** 设计仍在演进；**memories**（会话导出、向量索引、Haiku 摘要等）技能已写但未完全接线。

## 对 wiki 的映射

- 作为理解 **Superpowers 设计意图与演进背景** 的一手叙述，与仓库 README 的「安装 + 技能列表」互补；不替代官方仓库正文。
