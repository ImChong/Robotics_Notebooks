# Discrete Forcing（arXiv:2609.39526）

> 来源归档（ingest）

- **标题：** Discrete Forcing: Infusing Discrete Guidance into Continuous Denoising for Few-Step Action Experts
- **类型：** paper / vla / action-expert / flow-matching / efficient-inference
- **作者：** Jingbo Wang、Wenxuan Song、Wenhao Yu、Han Zhao、Xi Wang、Jiayi Chen、Donglin Wang、Yan Wang、Haoang Li
- **机构：** 香港科技大学（广州）、华南理工大学、中国科学技术大学、西湖大学、浙江大学、清华大学
- **arXiv：** <https://arxiv.org/abs/2609.39526>
- **项目页：** <https://discrete-forcing.github.io/> — 归档见 [sources/sites/discrete-forcing-github-io.md](../sites/discrete-forcing-github-io.md)
- **代码：** **已开源** — <https://github.com/Jbo-Wang/discrete_forcing>（MIT；LIBERO 训练与评测代码）；归档见 [sources/repos/discrete-forcing.md](../repos/discrete-forcing.md)
- **模型 / 权重：** 项目页标记 Coming Soon；GitHub README 说明 checkpoint 不随仓库提供
- **日期：** 2026-09-30（arXiv v1）；2026-10-03（入库）
- **一句话说明：** 同一个 VLA action expert 先预测离散动作 token 形成粗结构，再用一次连续 flow-matching refinement 补精度；总共两次 action-expert 前向，在 LIBERO 上以 1.5B 参数取得 97.6% 平均成功率。

## 核心摘录与 wiki 映射

1. **粗到细生成：** 离散分支并行预测动作 token；反量化 token 与噪声组成连续分支的起点，连续分支以一次前向恢复精细动作 chunk。总计 2 次 function evaluation，而不是连续去噪迭代多步。映射到 [论文方法页](../../wiki/entities/paper-discrete-forcing.md)。
2. **共享架构：** 前部 DiT 层共享，后部离散与连续分支分开建模；参数量与单分支 action DiT 相当。训练时离散 token 来自真值动作量化，连续分支同时有 flow-matching 与 one-step reconstruction 目标。映射到 [模型结构与训练](../../wiki/entities/paper-discrete-forcing.md)。
3. **评测：** LIBERO 1.5B / 2 NFE 的平均成功率为 97.6%，对照 StarVLA 2 NFE 的 95.1%；RoboTwin 2.0 为 59.32%，StarVLA 为 49.80%；RoboCasa-GR1 为 56.8%，StarVLA 为 43.9%。真机四任务平均报告的是 task progress（74.9%），不要误读为二元成功率。映射到 [实验与评测读法](../../wiki/entities/paper-discrete-forcing.md)。
4. **复现边界：** 官方仓库目前发布 LIBERO 训练 / 评测管线；数据与预训练权重须另行下载，仓库不含 checkpoint。README 将 RoboTwin 训练和评测代码列为 TODO。映射到 [开源状态与工程实践](../../wiki/entities/paper-discrete-forcing.md)。

## 当前提炼状态

- [x] arXiv 摘要、HTML 正文、项目页、代码链接核查（2026-10-03）
- [x] 项目页与公开仓库互相指向；项目页模型权重标记 Coming Soon
- [x] 将论文节点关联到 [VLA](../../wiki/methods/vla.md)、[Action Chunking](../../wiki/methods/action-chunking.md) 与 [Diffusion Policy](../../wiki/methods/diffusion-policy.md)
