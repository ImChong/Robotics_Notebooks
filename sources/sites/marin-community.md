# Marin 项目主页｜开放基础模型研发平台

- **官方主页：** <https://marin.community/>
- **官方仓库：** <https://github.com/marin-community/marin>
- **官方文档：** <https://marin.readthedocs.io/en/latest/>
- **官方训练叙事：** <https://openathena.ai/blog/scaling-marins-open-llm-pre-training/>
- **核查日期：** 2026-11-19
- **核对范围：** Marin 官网 Home 的项目定位、GitHub、Docs、Models、数据来源、公开实验与 selected writing 资源区；交叉核对 2026-11-17 官方 Mochi 权重公告
- **通读：** 已完成
- **开源状态：** **代码已公开**；官网列出训练语料来源、实验与预训练 checkpoint。具体 checkpoint 是否适合部署和数据是否全量可直接下载，要逐项核对官方公告及相关资产权限。
- **wiki 映射：** [Marin 开放式大模型预训练](../../wiki/entities/marin-open-llm-pretraining.md)

## 项目定位

Marin 由 Stanford 项目发展为 Open Athena 主导的开放基础模型研发项目，力图开放决策、训练代码、数据来源、实验结果及权重，而不止开放终态模型。

官网当前报告 Mochi **535B-A23B**（总参数 / 激活参数）在 **18.2T tokens** 上完成主预训练，已放出 [Hugging Face 上的预训练 checkpoints](https://huggingface.co/open-athena/Marin-Mochi-535B-A23B)；另有 Snowball 67B-A2B。

**注意：** 官方 2026-11-17 声明 Mochi checkpoint 尚无指令及安全后训练；并披露基准数据污染复核问题，不能将这些权重当成现成生产助手，也不能把博客规划的约 20T token 当作最终实训数。

## 关联

- [官方博客归档](../blogs/openathena-scaling-marins-open-llm-pretraining-2026-11-05.md)
- [官方源码归档](../repos/marin-community-marin.md)
- [Mochi checkpoint 公告](https://marin.community/blog/2026/11/17/marin-mochi-pretraining-checkpoints/)
