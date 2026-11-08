# Ceres Solver — 一手资料归档

> 来源层（官方代码/仓库）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** Ceres Solver
- **类型：** repo
- **链接：** https://github.com/ceres-solver/ceres-solver
- **入库日期：** 2026-11-08
- **一句话说明：** 非线性最小二乘优化求解器。
- **核对范围：** 已完整阅读 Ceres 官方 README、docs/source/nnls_tutorial.rst 的模型/残差/微分入门核心内容，并核对 examples/helloworld.cc 和 examples/slam/pose_graph_2d/pose_graph_2d.cc 的调用路径；未执行 C++ 程序。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [Ceres Solver](https://github.com/ceres-solver/ceres-solver) **（本归档主类型）**

## 完整使用资料与主要源码归纳

1. 对变量参数块 `x` 定义多个残差 `f_i(x)`，最小化 `1/2 * sum_i rho_i(||f_i(x)||^2)`，`rho_i` 为可选鲁棒损失；需符合参数范围与 manifold 结构。
2. 官方入门 `examples/helloworld.cc` 定义一个代价 functor，再将 `AutoDiffCostFunction` 注册进 `ceres::Problem::AddResidualBlock`，最后传入 `ceres::Solver::Options` 并执行 `ceres::Solve`。
3. `docs/source/nnls_tutorial.rst` 依次讨论残差块、参数、损失、约束和不同求导策略；SLAM 示例 `examples/slam/pose_graph_2d/pose_graph_2d.cc` 读取 g2o 格式轨迹约束并构建优化问题。
4. 闭环误匹配会引入大残差，可采用 Huber/Cauchy 等鲁棒损失而非所有残差同等加权；但损失函数不是错误数据的万能补丁。
5. 对四元数/旋转使用合适的 manifold/local parameterization，不能把冗余参数当成任意欧氏向量调整。

## 运行边界和工程复现

官方 `examples/helloworld.cc` 是最小入门；视觉工程先从重投影残差的 BA 开始，再加入内参/外参的适当固定策略与鲁棒核。对大规模稀疏问题选合适的 Schur / 线性求解配置，考虑线程数和内存。姿态图示例 `examples/slam/pose_graph_2d/pose_graph_2d.cc` 可用 g2o 格式追踪位姿图与回环约束。具体 build 命令须参考当前官方 Ceres 文档。

微分方法、参数尺度和初值决定数值稳定性；错误的噪声与鲁棒核容易造成看似收敛但有偏的解。优化器不自动处理时间同步或坐标系标定，也不能证明残差来自正确的数据关联。最新 master 可能和旧依赖的 Ceres API 不一致，需锁定发布版本。

## 方法与边界

非线性最小二乘优化求解器。本档案定位官方**开源软件项目**，已审阅主 README、技术教程关键流程及源码入口；未把未运行结果标为实测。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 本软件项目官方 README、官方教程与核心使用流程已阅读，保留真实工程限制。
- [x] 已核对官方 CLI 或源码样例的实际调用入口，时序图见独立实体。
- [x] 已新增 [Ceres Solver](../../wiki/entities/ceres-solver.md)，健康度由 CI/规则复核。
