# GTSAM — 一手资料归档

> 来源层（官方代码/仓库）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** GTSAM
- **类型：** repo
- **链接：** https://github.com/borglab/gtsam
- **入库日期：** 2026-11-08
- **一句话说明：** 因子图平滑与建图优化库。
- **核对范围：** 完整阅读官方 README、USAGE.md 与 examples/Pose2SLAMExample.cpp；核对版号 develop ba592c7、CMake quickstart、因子图与 NavState IMU 预积分文档入口。未执行编译及端到端测试。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [GTSAM](https://github.com/borglab/gtsam) **（本归档主类型）**

## 原始技术资料深读证据

1. **Values** 存储变量估计；**NonlinearFactorGraph** 保存 PriorFactor、BetweenFactor、投影与惯性因子，因子对应残差和噪声模型。
2. **流形状态**：SO(3)/SE(3) 的优化步骤先将非线性问题在线性切空间近似，再用 retract 更新到流形上，不能把旋转四元数当无约束欧氏向量相加。
3. **批量优化**：例如 `examples/Pose2SLAMExample.cpp` 构造 Pose2 的先验、相邻里程计和回环 BetweenFactor，通过 `GaussNewtonOptimizer` 求出轨迹并计算 `Marginals`。
4. **增量式优化**：iSAM2 面向图逐渐增长的在线估计；LIO-SAM 等系统可利用它更新关键帧、GPS 和回环约束。
5. **IMU**：README 明确给出 Forster 等的流形 IMU 预积分，以及当前 GTSAM 4 的 tangent-space NavState 实现；是否使用旧公式取决于 `GTSAM_TANGENT_PREINTEGRATION` 配置。

## 工程、实验与局限

GTSAM 是**通用算法库，不存在一项统一的精度冠军值**。官方 README 提供快速编译和测试命令：`cmake -S . -B build`、`cmake --build build --target check`、`cmake --build build --target install`。优化精度、运行时间和可观性须由具体因子图、噪声、初值、线性求解器及回环数据决定，不能拿 LIO-SAM 的评测结果当 GTSAM 的独立 benchmark。

通过 `find_package(GTSAM REQUIRED)` 与 `target_link_libraries(my_program PRIVATE gtsam)` 使用 C++ 库。建议先运行 `examples/Pose2SLAMExample.cpp`，再沿 `examples/CombinedImuFactorsExample.cpp` 学习 IMU 约束。开发分支 README 明确需要 C++17 且未来 API 可能变化，部署使用稳定 Release，CMake/TBB/Boost 可选配置需匹配工具链。

因子图求解器不会自动确定观测误差分布、异常值、量测的时间一致性或状态可观性；图结构与噪声选错，结果可能精确地优化了错误模型。`develop` 分支与 Release 的 C++/Python API 和数学设置可能不完全一致，复现实验必须记录源码版本；有 IMU 预积分的残差约定时尤其注意旋转和 bias 的定义。

## 方法与边界

因子图平滑与建图优化库。已针对本官方软件项目完整审读原始 README 和核心技术说明，并核对主执行入口；并未独立运行指标复现。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 完整审读官方 README / 使用文档；本归档是官方工具库，不冒充某一篇论文。
- [x] 检查作者提供的主要执行入口、模块、参数与约束；绘制官方代码时序图。
- [x] 对应独立 [GTSAM wiki](../../wiki/entities/gtsam.md) 已新增；健康度由规则和 CI 复核。
