# Science Prediction Universe：官方项目入口

> 官方项目资料归档（非网页镜像）。

- **类型：** site / benchmark / AI for Science
- **机构：** UniPat AI（论文牵头）
- **项目主页：** https://spu.unipat.ai/
- **对应论文：** https://arxiv.org/abs/2611.15049
- **GitHub：** https://github.com/UniPat-AI/SPU
- **数据集：** https://huggingface.co/datasets/UnipatAI/SPU
- **入库日期：** 2026-11-14
- **核对范围：** 论文首页、UniPat-AI/SPU 官方 README、Hugging Face 官方数据卡中的项目指向与开放说明；官网直读失败，因此不以官网未取得的页面信息作事实依据。
- **通读：** 已完成

## 核心内容

Science Prediction Universe 是对真实湿实验的**隐藏结果排序**进行评测的科学智能体基准。提交时给出实验协议和候选条件，由 agent 预测各组测量终点排序；评分器保留真实数据并计算 Spearman 相关。

## 官方开放状态

- GitHub 公开的是社区任务投稿、错误修订与检查流程，README 指向公开 Harbor 任务集；不能说仓库公开了全部论文 agent 执行系统。
- Hugging Face 发布 100 项公开任务、grader、oracle、实验测量值；16 项未发表实验任务不随公开数据发布。
- 官网地址由 arXiv 和官方 GitHub/HF 数据卡互相指向，**本次工具无法直接访问官网内容**，不据此声明榜单、平台功能或访问权限。

## 对 wiki 的映射

- [SPU 论文实体](../../wiki/entities/paper-science-prediction-universe.md)
- [论文归档](../papers/science_prediction_universe_arxiv_2611_15049.md)
- [仓库归档](../repos/unipat-ai-spu.md)
