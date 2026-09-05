# DexHoldem Skills（DexHoldem/DexHoldemSKills）

> 来源归档（repo）

- **标题：** DexHoldem Skills — 编码 agent 路由灵巧手策略
- **类型：** repo / embodied-agent / skill / dexterous-manipulation
- **来源：** DexHoldem / 香港大学 IDS
- **链接：** <https://github.com/DexHoldem/DexHoldemSKills>
- **论文：** [arXiv:2605.18727](https://arxiv.org/abs/2605.18727) — [`sources/papers/dexholdem_arxiv_2605_18727.md`](../papers/dexholdem_arxiv_2605_18727.md)
- **项目页：** <https://dexholdem.github.io/Dexholdem/> — [`sources/sites/dexholdem-github-io.md`](../sites/dexholdem-github-io.md)
- **入库日期：** 2026-09-05
- **一句话说明：** 把 perceive–route–execute 写成 Claude Code / Codex / Gemini CLI 可装的 skill，而不是自建 agent loop。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-dexholdem.md`](../../wiki/entities/paper-dexholdem.md)

---

## 开源状态（步骤 2.5）

| 项 | 状态（2026-09-05 复核） |
|----|-------------------|
| Agent skill | **已开源**：`npx skills add DexHoldem/DexHoldemSKills` |
| 运行时 | 依赖宿主编码 agent + 其原生 VLM；论文 sandbox 含 `SKILL.md`、`visual_guidelines/`、`router.py`、`action_translator.py`、`executor.py` |
| 策略执行 | 另需 [Dexholdem-Policy](https://github.com/DexHoldem/Dexholdem-Policy) 的 ZeroMQ 服务 |
| 许可证 | **未附 LICENSE**（GitHub `license: null`） |
| 成熟度 | README 自称 preview（2026-03-31） |

**结论：** **已开源、可装 skill**。它管路由与感知工作流，不替代策略训练仓。

## 仓库入口（对齐时序图）

论文 Appendix B.1 给出 sandbox 角色（以论文文档为准，仓内文件名可能随 preview 变动）：

| 组件 | 角色 |
|------|------|
| `SKILL.md` | 循环、动作空间、路由规则 |
| `visual_guidelines/` | 八个感知字段怎么从牌桌图读 |
| `router.py` | 规则门：等待 / 恢复 / 续跑多原子序列 |
| `action_translator.py` | agent primitive → 14 个策略原语 |
| `executor.py` / `remote_exec.py` | 下发到灵巧手控制端 |
| `capture.py` / `state.py` | 采图与结构化状态落盘 |

安装：

```bash
npx skills add DexHoldem/DexHoldemSKills
```

然后对 Claude Code / Codex / Cursor 说：`Launch a new experiment with the game loop.`

## 关联资料

- 论文：[`sources/papers/dexholdem_arxiv_2605_18727.md`](../papers/dexholdem_arxiv_2605_18727.md)
- 项目页：[`sources/sites/dexholdem-github-io.md`](../sites/dexholdem-github-io.md)
- 策略仓：[`sources/repos/dexholdem-policy.md`](./dexholdem-policy.md)
- Wiki：[wiki/entities/paper-dexholdem.md](../../wiki/entities/paper-dexholdem.md)
