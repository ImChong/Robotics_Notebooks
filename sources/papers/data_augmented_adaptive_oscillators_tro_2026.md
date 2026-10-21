# Data-Augmented Adaptive Oscillators（IEEE T-RO, 2026）

> 来源归档（ingest）

- **标题：** Data-Augmented Adaptive Oscillators: A Framework for Robust Gait Phase Estimation in Exoskeleton Assistance
- **类型：** paper / exoskeleton / gait-phase-estimation / adaptive-oscillator
- **期刊：** *IEEE Transactions on Robotics*（T-RO, 2026）
- **DOI：** <https://doi.org/10.1109/TRO.2026.3742234>
- **摘要级书目信息（次级索引）：** [AMiner 记录](https://www.aminer.org/academic/journal/-/5eb499cfedb6e7d53c09f628)；原始论文出处以 IEEE DOI 页面为准
- **入库日期：** 2026-10-17
- **一句话说明：** DA-AO 将自适应振荡器与步态模式识别结合，为外骨骼在平地、坡道、楼梯及户外地形转换中估计步态相位；本文档只归纳目前可检索到的摘要级信息。

## 摘要级要点

- **问题：** 纯数据驱动的步态估计可能缺少实际部署所需的可预测性与泛化能力；传统自适应振荡器有明确的模型结构，但难以应对地形变化。
- **方法：** Data-Augmented Adaptive Oscillator（DA-AO）将模型式振荡器与学习式感知结合，根据检测到的运动模式动态检索对应的振荡器超参数，使估计器运行在各模式适用的参数区间。摘要未披露模式分类器、参数库构建或检索算法的具体实现。
- **摘要报告的验证：** 髋部外骨骼主动助行实验覆盖平地、坡道上行、楼梯上行；户外连续地形转换也进行测试。摘要报告的步态相位 RMSE 分别为 0.025、0.021、0.032，户外转换为 0.025。
- **摘要报告的助行结果：** 相对无辅助条件，平地、坡道上行、楼梯上行的平均代谢成本分别降低 15.3%、10.6%、5.7%（均报告 $p<0.05$）；户外坡地行走的运输成本降低 12.3%（报告 $p<0.05$）。指标定义、实验样本量及统计细节应以论文全文为准。

## 对 wiki 的映射

- [DA-AO：外骨骼步态相位估计](../../wiki/entities/paper-data-augmented-adaptive-oscillators.md)
- [步态生成与相位组织](../../wiki/concepts/gait-generation.md)
- [腿式机器人运动控制任务](../../wiki/tasks/locomotion.md)

## 资料边界

本次归档依据 DOI 书目信息与 AMiner 可检索的摘要级记录；尚未核对论文全文，故不补写公式、采样频率、传感器配置或消融细节。当前资料未提供项目主页或代码链接，因此不据此断言源码未开源。
