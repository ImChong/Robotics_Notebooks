# LVI-SAM — 一手资料归档

> 来源层（原论文或论文出版入口）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** LVI-SAM
- **类型：** paper
- **链接：** https://arxiv.org/abs/2104.10831
- **入库日期：** 2026-11-08
- **一句话说明：** 激光、视觉和惯性子系统协同的平滑与建图。
- **核对范围：** 已通读 arXiv:2104.10831v2 七页正文、Fig. 1/2/3/5、§II VIS/LIS 方法、§III Urban/Jackal/Handheld 实验和 Table I/II、§IV 结论及参考文献；另核对官方 README 和主要节点代码调用入口。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [LVI-SAM 论文](https://arxiv.org/abs/2104.10831) **（本归档主类型）**
- [LVI-SAM 代码](https://github.com/TixiaoShan/LVI-SAM)

## 论文关键证据（2026-11-08 复核）

- **VIS↔LIS：** LIS 提供 VIS 初始化所需位姿与 IMU 偏置、通过累积激光帧为视觉特征估计深度；VIS 为 LIS 的 scan-to-map 提供运动初值并召回视觉回环候选，候选由激光进一步验证（§II）。
- **退化与因子图：** VIS 用特征跟踪数量/异常 IMU 偏置检测失败；LIS 用优化线性化正规矩阵最小特征值判断扫描匹配退化，拒绝不可信激光因子（§II）。
- **Urban 消融：** 开启 VIS 特征深度将无回环起终点平移误差 45.42m 降到 32.18m；全模块开启回环后为 0.28m（Table I，不能将此数当 ATE）。
- **Jackal/Handheld：** 回环启用时 GPS 轨迹 RMSE 分别 0.67m、0.83m（Table II），属于论文报告的特定数据集结果。
- **来源版本：** [arXiv 2104.10831v2, 2021-05-30](https://arxiv.org/pdf/2104.10831)，官方代码 master `0d822f6`；代码 README 仍有 `imuPreintegration.cpp` 图优化 TODO，论文架构与代码实现应分开说明。

## 方法与边界

激光、视觉和惯性子系统协同的平滑与建图。本条目仅做**来源身份与方法类别**定位，不将课程的中文意译表述直接等同论文定量结论；所有实验结果、具体损失函数、模块调用和代码运行性能，必须对照原文或官方源码进一步查证。

**已核对 README：** LVI-SAM 组合 VIS 与 LIS 子系统；官方 ROS 构建为 `catkin_make`，配置 `config/` 下 YAML 后以 `roslaunch lvi_sam run.launch` 启动，随后可播放 rosbag。官方 README 明确提及对 VINS-Mono 与 LIO-SAM 的借鉴，并在 TODO 中提示部分图优化工作仍需完善。上述仅为 README 层验证，不代表逐行阅读所有源码。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 已通读主论文 7 页及表格、图示、局限；论文没有单独在此归档的补充附录。
- [x] 已核对 ROS1 launch、七个核心节点和话题/回调，源码流程与原文流程图已写入 wiki；未运行 rosbag。
- [x] 新实体页 [LVI-SAM](../../wiki/entities/paper-lvi-sam.md) 包含原文流程图、源码时序图、实验和结论；健康度需以仓库脚本/CI 复核。
