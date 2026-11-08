# Robota-Embed 官方项目页：Unified VLA Embeddings for Robot In-Context Learning and Evolution

- **类型：** 官方研究项目 / technical report
- **项目主页：** <https://robota-ai.github.io/robota-embed/>
- **论文：** [arXiv:2610.19495v1](https://arxiv.org/abs/2610.19495)
- **研究机构：** NVIDIA、MIT、UC Berkeley、University of Oxford、CMU（作者署名）
- **核查日期：** 2026-11-08
- **代码：** **待发布**，官方“Code — Coming soon”；未提供可用 GitHub 仓库
- **模型/Benchmark：** **待发布**，官方“Model & benchmark — Coming soon”
- **论文深读记录：** [sources/papers/robota_embed_arxiv_2610_19495.md](../papers/robota_embed_arxiv_2610_19495.md)

## 官方内容与开放范围核查

1. 项目页介绍查询端的 **图像 + 文本指令 + 原生动作** 与候选端历史 **4 秒未来视频 + 动作** 的统一预测式检索，列出 EgoDex、DROID、LIBERO、RoboMIND–Franka-dual 四源。
2. 检索 Benchmark：4,000 queries、8,000 gallery、clip R@1 47.33%、episode R@1 81.77%。训练为 1,695,684 个 current/future anchor。
3. Embed2Evolve：ManiSkill 五任务、冻结 GPT-6 Astra 策略，Direct 24%→68%；同 Direct 比较 policy tokens 0.921M→0.518M（−43.7%），与独立演化 Omni 对比为 −79.5%。
4. Embed2ICL：DROID-Sim 47.0%→56.3%；LIBERO 困难五任务 86.4%→92.8%；真机 Split Cup 进度评分 66.67→87.50（非任务成功率）。
5. **可复现性结论：** 截至本次核查仍未列可执行源码、模型权重或 benchmark 下载。源码时序图不适用；论文里的算法流程图属于研究方案，并非已验证官方代码结构。

## 对 wiki 的映射

- [Robota-Embed 统一实体](../../wiki/entities/paper-robota-embed.md)
