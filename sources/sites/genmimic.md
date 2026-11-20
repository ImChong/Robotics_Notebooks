# GenMimic

> 来源归档（Humanoid Motion Intelligence 开源项目主表）

- **标题：** GenMimic
- **类型：** project-site
- **技术路线分组：** 动作跟踪与全身控制
- **链接：** https://genmimic.github.io/
- **论文：** <https://arxiv.org/abs/2512.05094>（`arXiv:2512.05094v2`）
- **数据集：** <https://huggingface.co/datasets/wlin21at/GenMimicBench>（428 条合成视频：217 Wan2.1 NTU + 211 Cosmos-Predict2 PennAction）
- **入库日期：** 2026-07-30（2026-11-20 复核更新）
- **通读：** 已完成
- **核对范围：** 2026-11-20 读取 <https://genmimic.github.io/> 全部正文并对照 `arXiv:2512.05094v2` 全文（另见 [genmimic_arxiv_2512_05094.md](../papers/genmimic_arxiv_2512_05094.md)），确认已公布 arXiv 与 HuggingFace 评测基准 `GenMimicBench`，但训练/跟踪代码仍标注 coming soon（未开源）。
- **一句话说明：** 把生成视频重建成四维人体动作，重定向后以三维关键点条件训练物理跟踪策略，并用对称性和分部奖励改善全身动作；最终产物是可执行跟踪器，视频生成只是参考入口，代码仍待发布。
- **开源状态（2026-11-20 复核）：** **基准已公开 / 代码仍待发布**（`wlin21at/GenMimicBench` 已上 HF；官网 Code 仍未释放）
- **策展入口：** [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-hrl-stack-04-from-generated-human-videos-to-physi.md`](../../wiki/entities/paper-hrl-stack-04-from-generated-human-videos-to-physi.md)（不另建重复实体；合并入已有论文页）

## 为什么值得保留

主表将该项列为人形运动智能六条路线之一的工程/研究入口；本库复用已有 GenMimic 论文实体页，便于选型与交叉引用，而不镜像主表全文。

## 对 wiki 的映射

- [GenMimic / From Generated Human Videos…](../../wiki/entities/paper-hrl-stack-04-from-generated-human-videos-to-physi.md)
- [Humanoid Motion Intelligence](../../wiki/entities/humanoid-motion-intelligence.md)
