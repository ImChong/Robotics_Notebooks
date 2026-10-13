# Axis Robotics Platform Technical Report（项目页核查）

> 来源归档（project site）

- **标题：** Axis Robotics Platform: A Cross-Simulator, Cross-Machine Infrastructure for Scalable Teleoperation and Generalizable Robot Learning
- **类型：** official technical report / project page
- **URL：** <https://techreport.axisrobotics.ai/>
- **PDF：** <https://techreport.axisrobotics.ai/Axis_White_Paper.pdf>
- **平台入口：** <https://hub.axisrobotics.ai/?tab=hub>
- **代码：** <https://github.com/AxisAIOrg>
- **数据：** 报告页未列出可直接下载的数据集链接
- **机构：** Axis Robotics Team
- **入库日期：** 2026-10-13
- **开源状态：** 部分开源；组织公开若干采集、任务生成、清洗和训练模块，不能据此认定整个平台、完整训练数据或所有权重均已开放。

## 页面内容

报告描述浏览器 MuJoCo-WASM 采集端、统一轨迹格式、Linux GPU 服务器回放、IsaacSim 逼真增广与域随机化，并将任务生成、轨迹清洗、VLA 训练、Sim2Real 仿真与真机评估放进统一工作流。报告页还提供 LIBERO 16 个任务的离线轨迹清洗汇总指标；解释和数值见[技术报告归档](../papers/axis-robotics-platform-tech-report.md)。

## 项目页入口核查

- **Paper**：页面链接官方 PDF。PDF 文件约 11.6 MB；当前抓取器未能提取全文。
- **GitHub**：页面链接 AxisAIOrg 组织；公开模块列表见[仓库归档](../repos/axisaiorg.md)。
- **Platform**：页面链接 hub.axisrobotics.ai 产品入口。
- **数据/权重**：本报告页未提供直达数据集或模型权重链接；不要把组织中存在训练代码等同于权重或训练数据已公开。

## 相关知识页

- [Axis Robotics 实体](../../wiki/entities/axis-robotics.md)
- [技术报告归档](../papers/axis-robotics-platform-tech-report.md)
