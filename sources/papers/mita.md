# Mixture-of-Top-k Attention: Efficient Attention via Scalable Fast Weights

> 论文来源归档（ingest）

- **论文：** <https://arxiv.org/abs/2602.01219v6>；[HTML 全文](https://arxiv.org/html/2602.01219v6)
- **作者：** Qishuai Wen、Zhiyuan Huang、Xianghan Meng、Wei He、Chun-Guang Li
- **机构：** 北京邮电大学
- **会议：** NeurIPS 2026（论文及作者官方仓库标注）
- **代码：** <https://github.com/QishuaiWen/MiTA>
- **版本：** arXiv v6，2026-09-27；核对日期 2026-11-06。
- **核对范围：** 已通读 arXiv v6 HTML 全文：摘要、引言、相关工作、§3 方法（快权重等价式、五维分类、MiTA 构造与复杂度）、§4 实验（表 2–5、LRA）、§5 结论、附录 A 消融（表 6）、附录 B 分类与算法 1、附录 C 算法泛化（图 9–10、表 7）、附录 D 局限；图中数值仅按正文描述核对，未逐点读图；官方源码另见 [repos 归档](../repos/mita.md)。
- **通读：** 已完成

## 内容摘录

- 自注意力可写成输入生成的、宽度等于序列长度 N 的两层快权重 MLP（权重为缩放后的键与值，偏置为零，激活为 exp）；注意力运算本身仍在执行。论文把这一"无界快权重增长"视为二次复杂度的另一种表述（§3.1）。
- 五维分类：扩展策略（压缩 / 路由）、专家数量、专家类型、专家构造、路由拓扑；线性注意力、Linformer、TTT、Agent Attention 属压缩，MoBA、BiFormer、DAT、Spark、DSA 属路由，NSA 兼有（表 1）。
- MiTA：用 m 个 landmark query（默认对均匀等分窗口做平均池化）探测全部键值；每个 landmark 取 top-k 激活键值对作为可变形、可复用的路由专家，同时经交叉注意力得到 landmark value，构成共享压缩专家。查询的注意力是「共享专家键值 + 路由到的专家键值」的拼接，实现取每查询路由 1 个专家，并用在线 softmax 合并（§3.2、算法 1）。
- 复杂度与瓶颈：核心项远小于全注意力的 N²；gather 的随机访存使流水线偏内存受限；通用的路由多个专家需借助 MoBA，速度更低（§3.2）。
- ImageNet-1K（DeiT 配方，表 2）：DeiT-T 72.2% / 1.2G；Agent-DeiT-T 70.3%（加 agent bias 71.1%）；MiTA-DeiT-T 71.1% / 1.1G，加深度卷积 73.4%；DeiT-S 79.9% / 4.6G，MiTA-DeiT-S 79.8% / 4.4G，加 DWC 80.6%，再加 Gate 81.2%。表 3：MiTA-ViT-5-S 81.3%，加 DWC 81.7%，ViT-5-S 为 82.2%。
- ADE20K（表 4，直接替换注意力而非原生预训练）：ViT-T CBT 39.1 mIoU / 12G，MiTA 36.5 / 7G（-42% FLOPs）；ViT-S 45.8→44.5；ViT-B 49.3→48.1；ViT-L 53.3→50.5。
- LRA（表 5，RTX 4090，标准注意力用融合内核）：标准注意力平均 58.19、总训练 10.6 h；MiTA 58.91、2.4 h（减少 77%）；route-only 变体 58.81、5.5 h；Agent Attention 58.22、1.9 h。
- 消融（表 6，同表 2 设置）：landmark 用 2D 平均池化 71.1，可学习参数 66.3，卷积 67.0；仅压缩 70.3，仅路由 70.8；增大专家数 m 比增大宽度 k 更有益。
- 算法泛化（附录 C）：标准注意力训练、MiTA 推理保留 95% 以上性能；ImageNet-21K 预训练微调（表 7）MiTA 相对标准注意力 Tiny 75.6 vs 76.9、Large 85.5 vs 85.9；推理时增大 m、k 常可超过训练配置。
- 局限（附录 D）：只做视觉，未评估 LLM 解码阶段（算力有限）；论文主要贡献为概念性框架；正文自述 m 的取值仅为初步研究，未充分发挥 MiTA。

## 对 wiki 的映射

- [MiTA 实体节点](../../wiki/entities/paper-mita.md)
