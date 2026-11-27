# RynnVLA-Latent 项目页与权重发布

> 来源归档

- **项目页：** <https://alibaba-damo-academy.github.io/RynnVLA-Latent.github.io/>
- **论文：** <https://arxiv.org/abs/2610.34356v1>
- **代码：** <https://github.com/alibaba-damo-academy/RynnVLA-Latent>
- **权重合集：** [Hugging Face](https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnvla-latent) · [ModelScope](https://www.modelscope.cn/collections/DAMO_Academy/RynnVLA-Latent)
- **核查日期：** 2026-11-27
- **通读：** 已完成
- **核对范围：** 项目主页的真机演示、方法、训练数据、评测和表示可视化；Hugging Face 合集、上游仓库权重表、ModelScope 的三条官方模型记录。未下载或运行模型权重。

| 发布项 | Hugging Face | ModelScope | 能做什么 |
|---|---|---|---|
| RynnLAM | [RynnLAM](https://huggingface.co/Alibaba-DAMO-Academy/RynnLAM) | [RynnLAM](https://modelscope.cn/models/DAMO_Academy/RynnLAM) | 从视频帧对离线提取 608 维潜动作，不是操作策略 |
| RynnVLA-Latent-2B | [2B](https://huggingface.co/Alibaba-DAMO-Academy/RynnVLA-Latent-2B) | [2B](https://modelscope.cn/models/DAMO_Academy/RynnVLA-Latent-2B) | Stage-1 潜动作预训练初始化 |
| RynnVLA-Latent-4B | [4B](https://huggingface.co/Alibaba-DAMO-Academy/RynnVLA-Latent-4B) | [4B](https://modelscope.cn/models/DAMO_Academy/RynnVLA-Latent-4B) | Stage-1 潜动作预训练初始化 |

仓库权重表注明 2B/4B 为 bfloat16、EMA 第 236442 步。模型合集未列论文测分的 Stage-2 各具身策略，也不是 30.1K 小时原始语料入口。项目主页的演示与可视化不改变这一边界。

- [论文归档](../papers/rynnvla-latent.md)
- [代码归档](../repos/rynnvla-latent.md)
- [知识页](../../wiki/entities/paper-rynnvla-latent.md)
