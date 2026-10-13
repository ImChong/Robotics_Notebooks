# Axis Robotics Platform: A Cross-Simulator, Cross-Machine Infrastructure for Scalable Teleoperation and Generalizable Robot Learning

> 来源归档（technical report）

- **标题：** Axis Robotics Platform: A Cross-Simulator, Cross-Machine Infrastructure for Scalable Teleoperation and Generalizable Robot Learning
- **作者：** Axis Robotics Team
- **类型：** 官方技术报告 / 平台白皮书
- **项目页：** <https://techreport.axisrobotics.ai/>
- **PDF：** <https://techreport.axisrobotics.ai/Axis_White_Paper.pdf>
- **GitHub：** <https://github.com/AxisAIOrg>
- **机构：** Axis Robotics
- **入库日期：** 2026-10-13
- **一句话说明：** 描述一套跨仿真器、跨设备的遥操作与机器人学习基础设施，将浏览器 MuJoCo 演示连接到 GPU 仿真增广、轨迹清洗、VLA 训练和 Sim2Real 评测。

## 摘要与系统主张

报告提出统一的端到端平台：用户可从浏览器端 MuJoCo WebAssembly 环境采集演示，不要求采集端配备专用硬件或 GPU；轨迹以统一格式上传，在 Linux GPU 服务器上回放，并可用 IsaacSim 做面向真实感的增广与域随机化。系统希望支持从遥操作数据到训练、仿真评测及真机部署的一体化工作流。

报告列出两条生产管线：轨迹清理/精炼，以及模型训练与 Sim2Real 评估。模块包括浏览器遥操作、由 LLM 与 MuJoCo 物理求解器辅助的任务生成、基于质量过滤/迟疑片段移除/平滑/重采样的数据清洗，以及下游 VLA 训练和实体机器人评测。

## 可核查的清洗验证

报告以 LIBERO 16 个任务的轨迹清洗作为离线验证。网页给出汇总值：

| 指标 | 清洗前 | 清洗后 / 结果 |
|------|---------|----------------|
| 平均加速度 | 0.2128 | 0.0985 |
| 平均 jerk | 2.7844 | 1.4670 |
| 位置偏差 | — | 0.0398 m |
| 移除比例 | — | 4.79% |

流程描述包括质量筛选、迟疑移除、Savitzky–Golay 平滑和 cubic spline 重采样。上述数据支持「清洗降低轨迹高频不平滑」这一离线结论；不能单独推出下游策略成功率提升或真机泛化增益。

## 代码与复现边界

报告项目页直接链接到 AxisAIOrg。组织公开仓库包含浏览器遥操作（AxisWebInfra）、任务生成（AxisTaskGen）、数据清洗（AxisDataCleaning）、训练基础设施（Axis-V1-Training）和 AXIS-V1。AxisWebInfra README 给出浏览器 MuJoCo、Franka Panda、轨迹记录及本地开发启动说明；仓库标注 Apache-2.0。详见[组织仓库归档](../repos/axisaiorg.md)。

项目页可访问 PDF 链接，但本次抓取器因文件较大未能解析 PDF 全文；此归档据官方网页正文与表格整理，不将网页未披露的训练配置、完整数据集或模型权重可用性视作已验证。

## 对 wiki 的映射

- [Axis Robotics 实体](../../wiki/entities/axis-robotics.md)：平台架构、清洗验证及开放边界。
- [AxisAIOrg 仓库索引](../repos/axisaiorg.md)：可复现模块和仓库状态。
- [技术报告项目页归档](../sites/axisrobotics-techreport.md)：项目页、PDF、平台与代码入口核查。
