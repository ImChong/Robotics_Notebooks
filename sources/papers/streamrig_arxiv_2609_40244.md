# StreamRig: Exploiting Intra-Rig Geometry for Streaming Multi-Camera Odometry（arXiv:2609.40244）

- **类型：** paper / multi-camera visual odometry / causal streaming
- **论文：** <https://arxiv.org/abs/2609.40244>（PDF：<https://arxiv.org/pdf/2609.40244>；HTML：<https://arxiv.org/html/2609.40244>）
- **项目页：** <https://weiyufei0217.github.io/StreamRig/>
- **代码：** <https://github.com/WeiYuFei0217/StreamRig>
- **提交日期：** 2026-09-30（arXiv v1）
- **作者：** Yufei Wei, Shuhao Ye, Qi Wang, Xin Zheng, Qing Huang, Rong Xiong, Yue Wang
- **机构：** 浙江大学；华南理工大学（按项目页作者上标）
- **投稿状态：** 项目作者主页列为 ICRA 2027 under review；arXiv 预印本不等于会议录用
- **一句话说明：** 冻结多视图 3D 基础模型来联合感知同步标定相机组，再用轻量因果时序模块压缩历史并估计相机组在线位姿。

## 摘要级方法与结果

StreamRig 面向移动机器人和车辆的同步多相机 rig。冻结前端把每一时刻的同步视图及 rig 标定一起编码；Rig-Resampler 压缩相机特征，CausalBridge 以因果注意力和 KV cache 处理历史，pose head 回归 rig 位姿。通过周期性 re-anchor 支撑长序列；可训练模块共 74.6M 参数，监督只使用相对位姿。训练分两阶段：group relocalization 预训练，再做 causal rig training。

论文在 NCLT、TartanGround、KITTI-360 和自采 ZJH 人形机器人数据集上评测。ZJH 的真机评测使用仿真数据训练的权重进行零样本迁移。论文摘要称，在四组数据上，相比所测非 oracle 单目流式方法与 rig-aware 离线方法，平移和旋转漂移更低；该结论适用于其比较对象和协议，不代表任意平台上的普遍保证。

## 术语与证据边界

- **rig / 相机组：** 多台相机经外参标定后组成的刚性传感器组。
- **streaming odometry：** 按时间到达的相机组连续更新位姿，不必等整段序列离线处理。
- **relative-pose supervision：** 训练目标为时间片段之间的相对位姿；不能据此说系统完全不需要训练数据或标定。
- 项目页补充报告了不同相机数量、重锚距离与训练窗口的消融。论文结果应回看完整表格，注意不同数据集的相机布局和指标不可直接混算。

## 来源归档

- [StreamRig 项目与论文页归档](../sites/streamrig-weiyufei0217-github-io.md)
- [StreamRig 官方代码仓库归档](../repos/streamrig.md)
- [StreamRig 论文详情](../../wiki/entities/paper-streamrig.md)
- [StreamRig 项目详情](../../wiki/entities/streamrig.md)
