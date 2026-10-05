# 银河通用 AstraBrain 官方技术入口

- **类型：** 官方公司技术页 / 项目入口核查
- **官方入口：** <https://galbot.com/about/>
- **核查日期：** 2026-10-05
- **新闻入口：** <https://galbot.com/news/>
- **WBC 项目页：** <https://qizekun.github.io/Humanoid-GPT/>
- **WBC 代码：** <https://github.com/GalaxyGeneralRobotics/Humanoid-GPT>
- **WBC 论文：** <https://arxiv.org/abs/2606.03985>
- **一句话说明：** 官网分别介绍 AstraBrain WAM 的异构数据学习与 WBC 的运动跟踪；WBC 官方仓库明确标注 AstraBrain-WBC 0.5，可复用已有 Humanoid-GPT 归档。

## 官方介绍摘录与证据边界

| 模块 | 官网提供的信息 | 可以确认的范围 |
| --- | --- | --- |
| AstraBrain WAM | 跨本体隐式世界–动作基座，LDA 利用人类/机器人、真实/仿真、有/无动作标注数据 | 有方法定位；本页未给出可核对的完整架构、损失函数或统一评测协议 |
| AstraBrain WBC | 8040 万参数、约 2 万小时动作语料；数据从 200 万帧扩至 20 亿帧时，报告成功率 83.26%→92.58% | 数字属于公司自报；论文和项目页提供运动跟踪的独立上下文 |
| WAM-TTT | 用无动作标注人视频做后训练部署 | 官网页另列部署技术，不能当成 WAM 预训练代码已开放的证据 |

## 开源核查

- **WAM：未确认公开实现。** 此次打开公司 About / News 页，未见对应 WAM 的训练代码、权重或全量数据下载入口；没有单独论文链接可据以推断推理时是否生成未来视频。
- **WBC：部分开源。** 项目页的 Code 指向 Humanoid-GPT；仓库描述明确写 AstraBrain-WBC 0.5。推理、评测、部署与 checkpoint 已发布，完整训练代码和训练数据仍列为待发布项。
- 不能因 WBC 或 GraspVLA 有公开仓库，就把整个 AstraBrain 系列标成开源。此结论以 2026-10-05 实际核查为准。

## 对 wiki 的映射

- [AstraBrain 技术路线](../../wiki/entities/galbot-astrabrain.md)
- [Humanoid-GPT / AstraBrain-WBC 0.5](../../wiki/entities/paper-humanoid-gpt.md)
- [官方代码归档](../repos/humanoid_gpt_galaxy_general_robotics.md)
- [项目页归档](humanoid-gpt-qizekun-github-io.md)
