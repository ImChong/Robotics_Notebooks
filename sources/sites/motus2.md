# Motus2 项目页（motus-robotics.github.io/motus2）

> 来源归档（ingest 配套站点；2026-10-05 复核）

- **URL：** <https://motus-robotics.github.io/motus2/>
- **论文：** <https://arxiv.org/abs/2608.30237>（v2，2026-09-10）
- **代码：** <https://github.com/shengshu-ai/Motus2>
- **模型入口：** <https://huggingface.co/motus-robotics>
- **机构：** GensPI（生数科技）；清华大学；北京航空航天大学；北京理工大学
- **前作：** Motus（arXiv:2512.13030）— <https://motus-robotics.github.io/motus>
- **入库日期：** 2026-09-01
- **最近复核：** 2026-10-05
- **一句话说明：** Motus2 灵巧操作自进化通用世界模型：共享 policy / simulator / evaluator 三接口、人数据金字塔、MBRL + Best-of-N、触觉专家与多本体真机 demo。项目页目前提供 GitHub 与模型页入口。

## 公开资料链接

| 资源 | 说明 |
|------|------|
| 项目页 | 论文介绍、数据/模型/触觉/记忆说明与真实机器人演示 |
| arXiv v2 | <https://arxiv.org/abs/2608.30237> |
| GitHub | <https://github.com/shengshu-ai/Motus2> |
| Hugging Face | <https://huggingface.co/motus-robotics>（团队模型主页，当前未确认 Motus2 专属权重） |

## 最新开源核查（2026-10-05）

- 项目页顶部新增 **Code** 与 **Model** 按钮，分别指向上述 GitHub 仓库和 Hugging Face 组织页。
- GitHub 仓库 README 声明计划在 2026 年 9 月逐步发布代码和 checkpoints，并列出 Stage 1/2 checkpoints、video pretraining、video-action/value training、MBRL、memory、tactile 等路线图。
- 截至本次复核，仓库文件区只显示 README；README 路线图各项均未勾选。准确表述是“官方代码仓已公开、发布路线图已公布”，不能据此认为可运行实现或权重已发布。
- Hugging Face 入口链接到 motus-robotics 团队主页；当前目录内模型名称属于 Motus / Robotwin2 等，未确认 Motus2 专属 checkpoint。不要将前作 Motus 权重视作 Motus2 权重。

## 项目页公开信息要点

| 模块 | 要点 |
|------|------|
| **General World Model** | 共享参数 backbone 暴露 **Policy（WAM）**、**Simulator（动作条件世界模型）**、**Evaluator（价值模型）** 三个控制接口 |
| **Chunk masks** | 联合预训练使用 joint mask；中后期使用 action-first mask，跨 chunk 保持因果 |
| **数据金字塔** | 约 **130K h** 原始人类第一视角录制；机端 mid-training 使用 **>100 h** 机器人轨迹与人机对齐数据 |
| **自进化** | **DiffusionNFT** MBRL + **Best-of-N** 测试时规划；失败/次优交互提供动力学与价值学习信号 |
| **记忆** | 默认 sliding window；另评 global autoregression 与 Hybrid Memory |
| **触觉** | 轻量 tactile expert 使用近期触觉窗口精修短动作子块；训练含未来力预测辅助任务 |
| **硬件** | WuJi-1/2、Sharpa 双手与 Tianji 双臂配置；项目页含真机视频 |

## 对 wiki 的映射

- [Motus2 实体页](../../wiki/entities/paper-motus2.md)
- [arXiv 摘录](../papers/motus2_arxiv_2608_30237.md)
- [官方代码仓档案](../repos/motus2.md)
- [前作 Motus](../../wiki/entities/paper-sa-2512-13030-motus-a-unified-latent-action-world-model.md)
- [同族产品 Motubrain](../../wiki/entities/paper-motubrain.md)
