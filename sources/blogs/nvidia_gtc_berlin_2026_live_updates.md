# NVIDIA GTC Berlin 2026：机器人与 Physical AI 相关动态

> 原始资料归档（仅摘录与 Robotics_Notebooks 主线相关内容，不转存整篇滚动新闻）

- **标题：** NVIDIA GTC Berlin: Live Updates on What’s Next in AI
- **类型：** blog（NVIDIA 官方活动动态汇总）
- **来源：** NVIDIA Blog
- **原始链接：** <https://blogs.nvidia.com/gtc-berlin-2026-news/>
- **活动日期：** 2026-10-20 至 2026-10-22
- **页面标注日期：** 2026-10-12 至 2026-10-24（滚动更新页）
- **入库日期：** 2026-10-21
- **一句话说明：** 汇总 GTC Berlin 及欧洲巡回活动消息；机器人主线包括 Isaac GR00T 参考人形 Agile ONE SN、Microduck 模拟训练演示、Build-a-Claw 工作坊，以及工厂数字孪生和机器人仿真案例。

## 机器人与 Physical AI 摘录

### Isaac GR00T 与参考人形

NVIDIA CEO Jensen Huang 在 [GTC Berlin 主题演讲动态](https://blogs.nvidia.com/blog/gtc-berlin-2026-keynote/)中发布 **Agile ONE SN** 参考人形：31 自由度，基于 Isaac GR00T，计划于 2027 年推出。NVIDIA 将其定位为机器人研究起点，并称其模块化工作流可把研究人员准备时间从数周缩短至数天。演讲将平台栈概括为 Isaac Capture（采集）、GR00T（机器人智能）、Isaac Lab（仿真训练）与 Isaac OS（运行时），并提及 Agility Robotics、Figure、KUKA、NEURA Robotics 等合作伙伴。

同一主题演讲还演示了 Hugging Face **Microduck** 的训练流程：智能体在 Isaac Sim 中设置训练场景和仿真机器人，将训练任务交给云端，再将策略带回 RTX Spark 评估并部署到实机。台上的四只 Microduck 演示了由仿真训练得到的动作。此处是演讲演示，新闻稿未提供完整复现配置。

### Build-a-Claw 开发者工作坊

新闻汇总记录了巴塞罗那、巴黎等地的 Build-a-Claw 活动：参与者使用开放模型与 NVIDIA 平台搭建能抓取物体的机器人夹爪；NVIDIA 将其作为快速动手体验 Physical AI 的示例。报道描述了机器人通过失败后调整来改善抓取，但没有给出统一任务基准、成功率或独立评测数据。

### 工厂数字孪生与机器人仿真

- NVIDIA 的 [欧洲工业案例汇总](https://blogs.nvidia.com/blog/europe-industrial-leaders-engineering-manufacturing/)介绍 Factory Operations Blueprint（FOX）早期访问版，用于构建协调工厂机器与专业任务代理的方案。西门子将代理用于工厂计划建议，并在数字孪生中测试变更，再由生产控制人员审阅。
- 同篇介绍西门子使用 NVIDIA Cosmos Data Factory 工具生成电路板缺陷图像，以准备检测系统；这属于厂商与合作方报告，不能视作通用检测性能基准。
- 工业案例还提到 Audi 与 Agile Robots 在 Omniverse / DELMIA 虚拟孪生中模拟、验证移动人形机器人与物流流程；Schaeffler 将人形机器人执行器数字孪生接入 Isaac 仿真框架，以便合作方在仿真中评估部件行为。

## 解读边界

- GTC 新闻与主题演讲用于了解产品路线、演示和合作案例，不等同于论文评测或可复现工程说明。
- Agile ONE SN 的发布信息与“研究准备时间”属于 NVIDIA 的产品定位和预期，不代表已公开可运行整机或第三方验证结果。
- Microduck、Build-a-Claw 与工业案例的公开报道未附完整训练配置、硬件细节或标准化对照评测。
- 汇总页会继续滚动更新；本归档以 2026-10-21 可见内容为准。具体产品、代码与许可状态应以各自官方项目页为准。

## 对 wiki 的映射

- [NVIDIA GTC（机器人 / Physical AI 一手资料）](../../wiki/entities/nvidia-gtc.md) — 增补 2026 Berlin 活动索引。
- [Isaac GR00T](../../wiki/entities/isaac-gr00t.md)、[Isaac Lab](../../wiki/entities/isaac-lab.md)、[Isaac Sim](../../wiki/entities/isaac-sim.md) — 参考人形与训练演示涉及的栈。
- [NVIDIA Omniverse](../../wiki/entities/nvidia-omniverse.md)、[NVIDIA Cosmos](../../wiki/entities/nvidia-cosmos.md) — 工厂数字孪生与合成数据案例。
