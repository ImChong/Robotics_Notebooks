# λ₀: Scaling Open-World Egocentric Human Data for Humanoid Loco-Manipulation

> 来源归档（paper / arXiv v1）

- **作者：** Lambda-humanoid Team
- **机构：** 阿里巴巴集团 Alibaba Token Hub、Amap
- **提交日期：** 2026-10-13（arXiv:2610.18950v1）；截至 2026-10-16 收录 arXiv 预印本。
- **论文：** <https://arxiv.org/abs/2610.18950>
- **全文：** <https://arxiv.org/html/2610.18950v1>
- **项目页：** <https://lambdahumanoid.github.io/>
- **官方代码：** <https://github.com/LambdaHumanoid/openlambda>
- **模型 checkpoint：** <https://huggingface.co/lambda-humanoid/lambda0-midtrain-100k>
- **HumanVerse 样例数据：** <https://huggingface.co/datasets/lambda-humanoid/HumanVerse-500-sample>（项目页链接的样例数据页面当前返回需授权）
- **来源归档：** [项目页核查](../sites/lambda-humanoid-project-page.md)；[源码核查](../repos/openlambda.md)
- **Wiki 主实体：** [λ₀ / OpenLambda](../../wiki/entities/paper-lambda0.md)

## 摘要式定位

λ₀研究如何把大规模第一视角人体移动操作数据转成可供人形策略学习的机器人对齐监督。工作同时发布 HumanVerse-500 数据集描述和三阶段训练方案：第一视角手部交互预训练、机器人对齐的全身人体动作中训练，以及用人类与机器人任务数据进行机器人后训练。

## 方法与数据要点

- **HumanVerse-500：** 500 小时、32,993 个 episode、约 8700 万姿态帧，覆盖 827 类任务；同步第一视角视频、身体动作与手部动作。500 小时指转换、质量筛选前采集的人类语料。
- **采集与对齐：** 颈挂 GoPro 采第一视角视频，PICO 头显及追踪器记录身体，视频估计手部动作；同步、动作重建、观察视角对齐和动作对齐后形成机器人可用的人体监督。
- **三阶段训练：** Stage I 从多个第一视角数据集学习手物交互；Stage II 用机器人对齐的全身人体动作扩大协调动作先验；Stage III 用机器人任务示范将策略落到可执行的机器人动作接口。模型以 Qwen3.5-2B 视觉语言骨干结合生成式动作专家，G1 执行侧接 SONIC 与 Revo2 手部。
- **评测：** G1 上四项真实移动操作任务共 40 次评测，论文报告 62.5% 任务成功率和 80.4% 平均进度；SIMPLE 仿真成功率 90.0%。移除 Stage II 后真实成功率由 62.5% 降至 22.5%，说明全身中训练在该实验配方中的贡献较大。

## 证据边界

- 仿真 SIMPLE 的 90.0% 结果采用每任务配置选择，不能直接视为部署时单一配置的普遍表现。真实评测为四项任务、40 次试验，需结合任务成功和部分进度两类指标阅读。
- 官方代码已公开，但项目 README 将完整 Stage III 后训练流程列为准备中；HumanVerse 训练数据不包含在代码发布中。
- Hugging Face 的 `lambda0-midtrain-100k` 是 100k 步 Stage II checkpoint，不含 EMA 权重或任务后训练策略；模型卡明确说明它不是可部署策略。项目页另列 HumanVerse 样例数据链接，但当前访问返回需授权，不能据此称完整数据集已公开。
