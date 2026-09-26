# Dream-RSI 项目页（dream-rsi.com）

> 来源归档

- **标题：** Dream-RSI — Recursive Self-Improvement through Evolving Worlds
- **类型：** site / paper-project
- **链接：** <https://dream-rsi.com/>（canonical：`https://www.dream-rsi.com/`）
- **论文 PDF（站内）：** <https://dream-rsi.com/assets/dream-rsi.pdf>
- **arXiv：** <https://arxiv.org/abs/2609.14858>
- **GitHub：** <https://github.com/zhengkid/Dream-RSI>
- **Hugging Face Papers：** <https://huggingface.co/papers/2609.14858>
- **入库日期：** 2026-09-26
- **一句话说明：** Google / DeepMind / UMD / UVA 联合技术报告页：History-as-exact-replay-simulator + dreaming meta-exploration + evolving worlds pool；含交互 demo 与结果表。
- **沉淀到 wiki：** [`wiki/entities/paper-dream-rsi.md`](../../wiki/entities/paper-dream-rsi.md)

## 开源状态（步骤 2.5）

| 资源 | 2026-09-26 核查 |
|------|-----------------|
| Paper PDF | ✅ 项目页 + GitHub `papers/Dream-RSI.pdf` |
| 项目页 & Live demo | ✅ `#demos` 交互 walkthrough |
| arXiv | ✅ abs 2609.14858（页内 CTA 曾暂链 PDF，以 arXiv 为准） |
| GitHub 完整代码 | ⏳ README Release plan：**Being prepared** |
| Reproduction scripts | ⏳ 待发布 |
| Discovered programs | ⏳ 待发布 |

- **结论：** **部分开源** — 复现入口以 PDF + demo 为主；**executable codebase 尚未发布**。

## 对本库的意义

- 为 [递归自改进](../../wiki/concepts/recursive-self-improvement.md) 提供 **meta-exploration / history replay** 路线：不训练 WM，用 **discovery tree 当 exact simulator**。
- 与机器人读者：长 horizon **Auto-Research / coding-agent discovery**（写 env、kernel、optimizer）的 **搜索策略 RSI** 可借鉴；真机策略仍须独立 verify 环境（见 [autoresearch harness 指南](../../wiki/queries/real-robot-policy-autoresearch-harness.md)）。
