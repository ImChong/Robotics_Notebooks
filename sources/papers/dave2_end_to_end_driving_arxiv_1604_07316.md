# End to End Learning for Self-Driving Cars（arXiv:1604.07316）

- **类型：** paper / first-party original reading
- **标题：** End to End Learning for Self-Driving Cars
- **作者：** Mariusz Bojarski、Davide Del Testa、Daniel Dworakowski 等
- **原文：** https://arxiv.org/abs/1604.07316v1
- **HTML：** https://arxiv.org/html/1604.07316v1
- **PDF：** https://arxiv.org/pdf/1604.07316v1
- **核对范围：** 2026-11-09 阅读 arXiv v1 全文§1–8、图表及说明、参考文献目录；未独立复现原实验。
- **通读：** 已完成
- **一句话说明：** DAVE-2 用单前视相机在部署时预测 1/r 曲率；采集时有三相机与驾驶员 CAN 数据。

## 原文方法和评测核对

采集约72小时视频，训练只选车道保持段并降采样至10Hz。五层卷积、三层全连接，Torch 7/DRIVE PX 30FPS。侧相机与平移/旋转增强制造离中心恢复场景，并计算约2秒回正目标；变换依赖地面近似平面假设。仿真偏离车道中心1m时触发虚拟接管；作者报告约98%代表性道路的自动转向时间，并非完整自主驾驶成功率。

## 局限与复现边界

相机恢复样本是近似扰动，不能覆盖全部闭环分布；无独立验证的全栈可复现代码。

## 对 wiki 的映射

- [论文实体](../../wiki/entities/paper-dave2-end-to-end-driving.md)
- [11 篇阅读路线](../../wiki/overview/robot-learning-foundations-11-papers.md)
