# Dream-1 官方项目页

> 来源归档（site）

- **标题：** Zero-Shot to New Robots: Introducing Dream-1
- **类型：** site
- **链接：** https://dreamlabs.ai/research/dream-1
- **机构：** Dream Labs
- **页面日期：** 2026-10
- **入库日期：** 2026-10-20
- **一句话说明：** Dream Labs 发布的 World Action Model API，面向此前未见的双臂平行夹爪机器人与未演示任务。
- **沉淀到 wiki：** [Dream-1](../../wiki/entities/dreamlabs-dream-1.md)

## 开源状态

- **API 可用；代码未开源。** 官方页面提供公共 API 入口和 Python 客户端示例，但未列出 GitHub、Hugging Face 或可下载权重链接；不能据此称其为开源模型。
- 页面称服务起价为 **$3/小时**，此处记录为官方标示，具体费用以 API 实际价格为准。
- Dream-1 是 **research beta**；官方提示其可能出错，尚未通过安全关键用途验证。

## 项目页要点

- **模型定位：** 面向双臂、平行夹爪机器人的 World Action Model，可尝试零样本控制未见本体、未见任务与未见环境。
- **部署路径：** 安装 Dream Labs 客户端、配置 API key、选择 `dreamlabs/dream-1` checkpoint，提交观测并请求动作。
- **评测口径：** 官方报告在 27 种相对 YAM 未见机器人上的仿真零样本评测；每种本体共 300 次 rollout，分别覆盖指令跟随和未见任务泛化。表格分数是相对 YAM 的结果，不等同于成功率。
- **工程边界：** 零样本表现随机器人而异，并部分依赖本体的逆运动学（IK）求解器；实机部署应先在受控环境中验证。