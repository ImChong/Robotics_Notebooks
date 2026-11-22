# RigVis：双目惯性 SLAM 项目公开说明（作者动态）

> 来源归档（blog / 项目线索，待核验完整源码）

- **标题：** RigVis — stereo-inertial SLAM written from scratch
- **作者：** Ekrem Öztürk（原作者公开动态中的署名）
- **类型：** 作者项目说明 / 技术简介（非论文，非代码仓库）
- **公开线索：** https://co.linkedin.com/in/emanuel-sanguino-rivera-b16410250 （该公开动态页引用了作者的 RigVis 内容）
- **作者给出的代码/图表/地图短链：** https://lnkd.in/dFMwckqB （2026-11-22 无法在当前访问环境解析重定向，非已验证 GitHub 地址）
- **入库日期：** 2026-11-22
- **核对范围：** 阅读公开动态页对作者 RigVis 原帖的引用文本，包含主要模块、ICP 融合描述与 KITTI 00 自报误差；未打开原帖独立页面、项目完整 README、源码或实验日志。
- **通读：** 未完成
- **开源状态（复核）：** **已开源**。2026-11-22 后续定位作者 GitHub 账号及 [官方仓库](https://github.com/ekrrems/Stereo-inertial-slam)，详见 [已通读的官方源码归档](../repos/rigvis.md)；本条 LinkedIn 短链尚未直接解析。
- **对应 wiki：** [RigVis 项目实体](../../wiki/entities/rigvis.md)

## 2026-11-22 官方源码复核更新

- 作者公开仓库：[ekrrems/Stereo-inertial-slam](https://github.com/ekrrems/Stereo-inertial-slam)，已阅读完整 README 与关键运行入口，源码归档另见 [sources/repos/rigvis.md](../repos/rigvis.md)。
- 原先说“1 m 回环阈值”与实际默认 `loopSearchRadiusMeters_=50.0` m 不符；“8 次回环”和“约 570 万点”未从 README 的正式结果表证实。
- 官方 README 确认 KITTI 00 `t_rel=1.20%`、`r_rel=0.50°/100m`、`ATE(RMSE)=2.81m`，是**作者报告**，本仓库尚未复现实测。
- 源码明确 **loosely coupled ESKF**；IMU 进入 BA 是作者 Roadmap，稠密 ICP 默认 `dense_icp: 0` 且作为可选局部 BA 相对位姿因子，并非默认运行。
- 本文件是历史公开动态来源记录，因未独立通读原帖仍保持「通读：未完成」；RigVis wiki 的 `original` 已指向完成阅读的官方仓库归档。

## 作者公开说明中可核对的信息

- 项目定位：使用双目相机而非激光雷达，兼顾稀疏 SLAM 位姿估计与稠密深度地图。
- 技术栈：C++、Eigen、OpenCV、Ceres；ORB 特征与双目匹配；局部 BA、SE(3) 位姿图；使用 ESKF 融合 IMU。
- 稠密地图：每关键帧双目深度关联关键帧位姿，优化后可随关键帧位姿修正；相邻关键帧点云采用 point-to-plane ICP，作为额外约束加入 Ceres 图中。
- 作者报告 KITTI 00（约 3.7 km）漂移 1.2%、全局误差 2.8 m。未见误差计算定义、评价脚本或复现实验，**不能作为第三方测评结果**。

## 用户提供但尚未从一手源码/实验核实的内容

- 行约束立体匹配、PnP + RANSAC；SGBM 稠密深度；具体是否完全不依赖 ROS；
- 回环检测 1 m 阈值、合计 8 次回环；稠密地图约 570 万点；
- “相对平移误差 1.20%、ATE RMSE 2.8 m”的**精确指标名称**：作者贴文只使用 drift / global error，名称需通过评测定义核实。

## 后续核查任务

1. 解析作者短链，取得官方源码/项目 URL、许可证、分支与 commit；核实是否有可运行 KITTI/IMU 入口。
2. 阅读 README、front-end、ESKF、BA、回环/ICP 与稠密地图实现；确认数据依赖、传感器输入、线程/运行时调用关系。
3. 查 KITTI 00 评测脚本、配准方法、轨迹数据与日志；核对是否使用真实 IMU、是否只用 KITTI stereo，验证 1.2% / 2.8 m / 570 万点。
4. 完成阅读后更新 wiki 的流程图、补源码运行时序图，再执行健康度与 CI 门禁。

## 资料定位

本归档是**第一手技术公开说明的线索**，不是已完成的源码深读，也不是经过验证的独立评测。与 [ORB-SLAM3](../../wiki/entities/orb-slam3.md) 等成熟项目对照时，应保留不同实验设置和指标不可直接比较的边界。
