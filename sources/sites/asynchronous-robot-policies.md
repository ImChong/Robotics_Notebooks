# ARP：Asynchronous Robot Policies 官方项目页

> 来源归档（site）

- **标题：** Asynchronous Robot Policies / 异步机器人策略
- **类型：** site
- **链接：** <https://arp.insait.ai/>
- **arXiv：** <https://arxiv.org/abs/2611.05155>
- **论文版本：** arXiv v1（2026-11-03）
- **入库日期：** 2026-11-05；再次核查：2026-11-16
- **沉淀到 wiki：** [paper-asynchronous-robot-policies](../../wiki/entities/paper-asynchronous-robot-policies.md)

## 官方项目页核查

- **代码：待发布。** 页面资源区标注 **Code (soon)**，没有可核实的可运行 GitHub 源码入口。
- **模型权重：待发布。** 页面资源区标注 **Models (soon)**，只列拟发布 checkpoint 与其训练配置，不能视为已有可下载资产。
- **已公开资料：** 方法描述、示意图、LIBERO / LIBERO-Perturb / Dyn-Bump 实验、RoboLab 任务表、Franka 真实运行视频和局限讨论。
- **工程边界：** 未读取源码，无法核对调度、进程间通信与驱动接口实现；wiki 中的 Mermaid 仅根据论文机制绘制。

## 主要信息与审阅提醒

项目页阐明双进程多频率：低频 VLM 写入定宽语义记忆，高频视觉动作专家直接查看最新相机帧、读取最近的缓存，训练包含时间错配与执行延迟。项目页特别提醒：RoboLab 上 ARP-3B 20.0% 低于 π0.5-DROID 37.9%，真机只有三个任务各十次，动态实验中部分增益来自快专家本身的视觉输入，而不是异步调度的独立作用。

本次阅读以 [arXiv v1 全文归档](../papers/asynchronous_robot_policies_arxiv_2611_05155.md) 为事实主线；项目页部分协议/结果展示与 v1 存在差异，不合并为未标口径的单一数字。
