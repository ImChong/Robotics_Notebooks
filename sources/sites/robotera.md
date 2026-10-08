# 星动纪元（ROBOTERA）：公司与产品时间线核查

- **类型：** 官网、展商介绍与政府发布记录
- **官网：** https://www.robotera.com/
- **展商自述：** https://wrc.cie.org.cn/expo/company/438.html
- **核查日期：** 2026-10-08
- **沉淀到 wiki：** [星动纪元](../../wiki/entities/robotera.md)

## 事件日期与来源

| 事件 | 日期口径 | 原始入口 |
| --- | --- | --- |
| 公司成立 | 2023-08；2026 世界机器人大会展商自述明确到月 | [展商介绍](https://wrc.cie.org.cn/expo/company/438.html) |
| Humanoid-Gym | 2024-04-08 arXiv v1；不是仓库首发日期 | [论文](https://arxiv.org/abs/2404.05695)、[项目页](https://sites.google.com/view/humanoid-gym/) |
| VPP | 2024-12-19 arXiv v1；2025 ICML Spotlight；论文日期不等于代码首次公开 | [论文](https://arxiv.org/abs/2412.14803)、[项目页](https://video-prediction-policy.github.io/) |
| ERA-42 | 2024-12-23 产品发布；资料页发布时间为 12-25 | [中关村科学城发布记录](https://www.ncsti.gov.cn/kjdt/scyq/zgckxc/zgcdt/202412/t20241225_190563.html) |
| 星动 L7 | 2025-07-22 产品发布；资料页发布时间为 07-23 | [中关村科学城发布记录](https://www.ncsti.gov.cn/kjdt/scyq/zgckxc/zgcdt/202507/t20250723_211232.html) |

## 官网与展商页面观察

官网列出 L7、Q5、M7、XHAND 1 / Lite / PRO 和 ERA-42。展商介绍将公司技术分为数据、大脑、运控、灵巧手、人形整机，L7 展示全身遥操作，M7 为固定工位半身平台。这里记录公司的公开定位，不将商业演示视为统一基准结果；本轮不为尚未核准首发日期的硬件各建时间节点。

## 开放范围核查

- [Humanoid-Gym](../repos/humanoid-gym.md)：官方项目页链接训练框架；Isaac Gym 训练、MuJoCo sim2sim 与 XBot 真机验证。
- [VPP](../repos/video-prediction-policy.md)：项目页 Code 指向官方仓，README 提供训练、评测、真机部署入口和部分权重/潜数据链接。
- [robotera_vla](../repos/robotera_vla.md)：当前为 M7 / π₀.₅ 基线；采集文档、训练/推理示例公开，机器人侧遥操作和 recorder 服务由已有环境提供。不是 ERA-42 完整实现的开放声明。
- [xbot_sdk_api](../repos/xbot_sdk_api.md)、[teleop_client](../repos/teleop_client.md)：公开接口/命令文档依赖厂商 developer 环境、消息定义或授权文件；不能据此推定控制服务、全身策略和数据池均开放。
- ERA-42 官网产品入口及上述发布记录未列出其完整训练代码、权重和数据下载；VPP 与 M7 基线开放资产分别核查，不与 ERA-42 混用。

## 对 wiki 的映射

- [星动纪元](../../wiki/entities/robotera.md)
- [公司路线对照](../../wiki/comparisons/robot-foundation-model-company-paths-2026.md)
