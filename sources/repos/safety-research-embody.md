# safety-research/embody（Anthropic Embody 评测仓）

> 来源归档

- **标题：** embody
- **类型：** repo（宣称公开镜像，入库日未上线）
- **组织：** [github.com/safety-research](https://github.com/safety-research)
- **链接：** https://github.com/safety-research/embody
- **关联研究：** [Claude plays robotics](../sites/anthropic-claude-plays-robotics.md)
- **入库日期：** 2026-08-28
- **再核日期：** 2026-11-19
- **一句话说明：** Anthropic Frontier Red Team 在 *Claude plays robotics* 中承诺的 **Embody** 评测套件公开镜像：按 cell 列出命令（`EXPERIMENTS.md`）与计分（`METRICS.md`）。
- **开源状态：** **宣称将开源 / 截至 2026-11-19 仍未公开** — 原文："The code, once released, will be in `github.com/safety-research/embody`"。`2026-08-28` 与 `2026-11-19` 两次对该 URL 的 HTTPS 请求均返回 **HTTP 404**；无 LICENSE、无 README 可核。
- **核对范围：** 2026-11-19 再次实测 `https://github.com/safety-research/embody` HTTP 状态（404），无可读取代码仓
- **通读：** 未完成
- **沉淀到 wiki：** [Embody](../../wiki/entities/anthropic-embody.md)

---

## 核查记录（步骤 2.5）

| 项 | 结果 |
|----|------|
| 研究页是否写将开源 | 是，给出精确 GitHub 路径 |
| 项目页是否另有链接 | 无独立项目页 |
| 2026-08-28 GitHub | **404** |
| 可运行入口 | 无（仓不存在） |
| 源码运行时序图 | **不适用**（无可运行实现） |

后续 lint：仓一旦公开，应补 stars/许可/入口脚本，并在 wiki 实体页把「不适用」改为 sequenceDiagram。

## 原文承诺的仓内文件

- `EXPERIMENTS.md` — 每个评测 cell 的命令
- `METRICS.md` — 计分
- `envapi/training_bridge.py:train_ppo_batched` — RL 接口的 PPO 路径（研究文附录）

## 对 wiki 的映射

- 评测实体 → `wiki/entities/anthropic-embody.md`
- 控制接口概念 → `wiki/concepts/llm-robotics-control-interfaces.md`

## 参考链接

- <https://github.com/safety-research/embody>（入库日 404）
- <https://www.anthropic.com/research/claude-plays-robotics>
