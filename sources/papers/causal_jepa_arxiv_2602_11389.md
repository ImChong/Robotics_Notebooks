# Causal-JEPA: Learning World Models through Object-Level Latent Interventions（arXiv:2602.11389）

> 来源归档（ingest）

- **标题：** Causal-JEPA: Learning World Models through Object-Level Latent Interventions
- **类型：** paper / JEPA / object-centric world model / planning
- **arXiv：** <https://arxiv.org/abs/2602.11389>（ID 来自项目页，非文章给出）
- **项目页：** <https://hazel-heejeong-nam.github.io/cjepa/>
- **代码：** <https://github.com/galilai-group/cjepa>（页面未声明许可证）
- **权重：** <https://huggingface.co/HazelNam/CJEPA/tree/main>
- **作者：** Heejeong Nam；Quentin Le Lidec\*；Lucas Maes\*；Yann LeCun；Randall Balestriero（\* 同等贡献）
- **机构（项目页）：** Brown University；New York University；Mila 与 Université de Montréal
- **入库日期：** 2026-10-20
- **来源文章：** [wechat_deepblue_lecun_jepa_11_papers_2026-10-20.md](../blogs/wechat_deepblue_lecun_jepa_11_papers_2026-10-20.md)

## 开源核查（步骤 2.5，2026-10-20）

项目页列出 arXiv、代码与 Hugging Face 权重，代码入口齐全 → **已开源**。项目页另有网站模板致谢链接，非项目代码。

## 摘录要点（归纳）

- 学习单位从 patch 换成物体：场景表示为多个 object-level latent。
- 训练时遮挡某个物体的 latent，要求模型依据其他物体与动作推断其状态，以迫使模型学习物体间影响。
- 文章转述的结果：反事实视觉推理准确率比无遮挡版本高约 20 个百分点；控制任务中仅用 patch 型世界模型约 1% 的 latent 特征即可接近其规划表现。

## 对 wiki 的映射

- 主节点：[paper-causal-jepa](../../wiki/entities/paper-causal-jepa.md)
