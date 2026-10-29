# muscle-supermimic 项目页

> 来源归档

- **标题：** muscle-supermimic
- **类型：** site
- **URL：** <https://muscle-supermimic.beanpow.top/>
- **代码：** <https://github.com/kaibohex/muscle-supermimic>（[源码归档](../repos/muscle-supermimic.md)）
- **入库日期：** 2026-10-27
- **一句话说明：** 用可微世界模型训练肌肉驱动人体模型跟踪动作，展示 MS-Human-700 和 MyoFullBody 的实验。

## 核查摘要

项目页公开代码、训练命令及演示，并称代码、技术报告和 checkpoint 持续更新。MS-Human-700 锁手配置使用 694 条肌肉；在 972 条 KIT 训练片段上训练，作者报告单张 RTX 5090 训练 4 小时后，108 条未见测试片段归一化回报 0.83，其中 99 条跟踪至结束。每条测试片段从首帧评估一次。结果为项目作者报告，不能推断真实人体或机器人部署效果。

数据下载需接受数据集条款；页面说明 MyoFullBody 的 MoBL-ARMS 部分限非商业使用。当前未进行域随机化。

## 对 wiki 的映射

- [muscle-supermimic](../../wiki/entities/muscle-supermimic.md)
