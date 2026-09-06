# SDPG — UCLA 自蒸馏策略梯度项目页

- **来源：** https://lauyikfung.github.io/SDPG
- **类型：** site
- **机构：** UCLA（通讯 Quanquan Gu）；合作 Princeton
- **论文：** arXiv:2606.04036 — *Self-Distilled Policy Gradient*
- **代码：** https://github.com/lauyikfung/SDPG（**已开源**）
- **归档日期：** 2026-09-06

## 一句话说明

LLM RLVR 后训练：SDPG 在 GRPO 式 verifier 优势上叠加 privileged on-policy 全词表自蒸馏与 reference KL 正则；项目页链出 arXiv、GitHub 与 Hugging Face Papers。

## 项目页核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| GitHub | **已开源** — [lauyikfung/SDPG](https://github.com/lauyikfung/SDPG) |
| Hugging Face | [papers/2606.04036](https://huggingface.co/papers/2606.04036) |
| 基座框架 | [verl](https://github.com/volcengine/verl) |
| 硬件 | README 默认 8× GPU（A100/H100 级） |

## 同名消歧

- 机器人视觉 RL 另有 **Stochastic Decoupled Policy Gradient**（arXiv:2605.26478）→ [sdpg-haoxiangyou-website.md](./sdpg-haoxiangyou-website.md)

## 交叉链接

- 论文专档：[sources/papers/sdpg_self_distilled_policy_gradient_arxiv_2606_04036.md](../papers/sdpg_self_distilled_policy_gradient_arxiv_2606_04036.md)
- 仓库归档：[sources/repos/sdpg-lauyikfung.md](../repos/sdpg-lauyikfung.md)
- Wiki：[wiki/entities/paper-sdpg-self-distilled-policy-gradient.md](../../wiki/entities/paper-sdpg-self-distilled-policy-gradient.md)
