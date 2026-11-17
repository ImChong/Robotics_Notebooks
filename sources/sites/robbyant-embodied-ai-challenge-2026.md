# 首届蚂蚁灵波具身大模型挑战赛（2026）官方赛事资料

> 来源归档：竞赛主办方阿里云天池的一手赛事介绍与赛题页面，非获奖新闻或第三方转载。

- **标题：** 首届蚂蚁灵波具身大模型挑战赛（Robbyant Embodied AI Challenge）
- **类型：** site / 官方竞赛规则与赛题
- **主办方：** 蚂蚁灵波科技、魔搭社区、阿里云天池
- **赛事介绍：** <https://tianchi.aliyun.com/competition/entrance/532514/introduction>
- **赛题与数据：** <https://tianchi.aliyun.com/competition/entrance/532514/information>
- **赛事模型代码：** <https://github.com/Robbyant/lingbot-vla-v2>（**模型**官方仓库，并非赛事专用评测仓库）
- **赛事指定 6B 权重：** <https://huggingface.co/robbyant/lingbot-vla-v2-6b/tree/main> / <https://modelscope.cn/models/Robbyant/lingbot-vla-v2-6b>
- **仿真平台：** <https://robotwin-platform.github.io/>；<https://huggingface.co/datasets/TianxingChen/RoboTwin2.0>
- **归档日期：** 2026-11-17
- **核对范围：** 已阅读天池「赛事介绍」所列大赛概况、赛程、组队/合规、奖励、开发者资源及声明；「赛题与数据」所列线上仿真/真机赛题、训练与评测约束、交付清单以及线下真机决赛流程；对照 LingBot-VLA 2.0 官方 README 与 RoboTwin 2.0 官方项目介绍核验资源身份。
- **通读：** 已完成
- **一句话说明：** 以 LingBot-VLA 2.0 开源权重为统一基座，用 RoboTwin 2.0 的 clean 数据完成线上后训练、双设置评测，再以线下真机黑客松考察部署与实测能力的赛事协议。
- **沉淀到 wiki：** [蚂蚁灵波具身大模型挑战赛](../../wiki/entities/robbyant-embodied-ai-challenge-2026.md)

## 官方规定的评测协议

| 环节 | 官方要求 |
|------|----------|
| 线上必做 | Aloha-AgileX 本体；RoboTwin 2.0 共 50 个任务 |
| **训练集** | 每任务 **50 条 clean** 示范；**禁止使用 randomized 数据参与训练** |
| **评测集** | clean、randomized 两种 setting；**各任务每种 setting 100 次测试**，提交成功次数与测试次数 |
| 模型基座 | 指定 LingBot-VLA 2.0 6B 预训练权重；可优化训练参数、数据处理、模型结构或推理策略 |
| 可选真机 | 自选任务/设备/数据的真实机器人 Demo；作为筛选参考，**无固定分值** |
| 初赛材料 | results.json、训练与评测代码及配置、相应 checkpoint、复现与调优报告；真机 Demo 另附视频与项目报告 |
| 决赛 | 在组委会统一硬件、场地、数据及边缘算力支持下，完成现场适配和真机推理评测 |

## 时间与评审边界

- 2026-10-26 24:00：初赛作品截止。
- 2026-10-27 至 11-05：初赛评审；11-06 公示入围名单。
- 2026-11-14 至 11-16：上海线下决赛（以官方赛事介绍更新后的日程为准）。
- **初赛成绩只作晋级筛选，不与决赛分数累加**；最终奖项依据线下真机评测与官方评审。
- 禁止人工遥控、剪辑伪造、硬编码演示、伪造分数和使用未授权数据。

## 开源与复现边界

- **已公开：** LingBot-VLA 2.0 模型训练/推理代码、预训练权重，RoboTwin 2.0 仿真平台与数据集；已有资料分别归档于 [LingBot-VLA 2.0](../repos/lingbot-vla-v2.md)、[RoboTwin 2.0](../repos/robotwin-2.md)。
- **不是同一协议：** LingBot-VLA 2.0 README 的常规 RoboTwin 后训练示例允许混用 clean / randomized；**赛事专用协议明确只准 clean 用于训练**。复现赛事不能直接照搬通用示例的数据选择。
- **未确认：** 主办方是否公开了独立的竞赛复现仓库、统一评测脚本及决赛完整任务包；天池页面的部分下载材料需要报名登录，不把选手提交材料等同于官方开源。
- 此赛事本身不是新算法论文；无对应独立源码执行入口，源码分析应进入各自的 [模型实体](../../wiki/entities/lingbot-vla-v2.md) 与 [仿真实体](../../wiki/entities/robotwin.md)。
- **赛事获奖结果**：本次官方规则页未列可直接核实的完整最终榜单，故不将媒体报道的队伍成绩作为官方已核实数据入库。

## 与知识库的关系

- [LingBot-VLA 2.0](../../wiki/entities/lingbot-vla-v2.md)：指定模型与训练/推理开源栈。
- [RoboTwin 2.0](../../wiki/entities/robotwin.md)：必做仿真数据与基准任务。
- [具身评测纵深路线](../../roadmap/depth-embodied-eval.md)：数据隔离、按设置逐任务评测与最终真机验证的真实赛事实例。
