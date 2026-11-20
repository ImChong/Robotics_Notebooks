# Kinetics Observer：官方 C++ 代码、入口与实现边界

- **类型：** repo
- **代码：** https://github.com/jrl-umi3218/state-observation
- **论文：** https://arxiv.org/abs/2406.13267
- **核对日期：** 2026-11-20
- **开源状态：** 有可辨认的 KineticsObserver C++ 实现、头文件与 CTest 测试；未见本次验证过的论文全套 RHP Friends / HRP-5P 记录数据与一键复现实验入口。仓库许可证与历史文件声明可能不同，复用前核查。
- **核对范围：** README、CMakeLists.txt、src/kinetics-observer.cpp 中构造、updateMeasurements、update；include/state-observation/dynamics-estimators/kinetics-observer.hpp 中接触/IMU/动力学设置 API；src/extended-kalman-filter.cpp 预测接口；unit-testings/test-kinetics-observer.cpp 与 unit-testings/CMakeLists.txt。没有在本环境编译或执行。
- **通读：** 已完成

## 实际文件和接口

| 文件 | 已核对行为 |
|---|---|
| 根 CMakeLists.txt | C++ 编译、依赖、BUILD_TESTING 选项，增加单测目录 |
| src/kinetics-observer.cpp | 构造时初始化 MEKF，updateMeasurements 汇集当前 IMU、力矩、可选绝对姿态测量；update 调滤波器预测/更新，回填世界和接触运动学 |
| include/state-observation/dynamics-estimators/kinetics-observer.hpp | KineticsObserver、setSamplingTime、setIMU、addContact/removeContact、updateContactWithWrenchSensor / NoSensor、setCoMInertiaMatrix、setCoMAngularMomentum、update |
| src/extended-kalman-filter.cpp | 状态预测和传感器预测实现 |
| unit-testings/test-kinetics-observer.cpp | main 调用 testKineticsObserverCodeAccessor：构造 KineticsObserver(4,2)，设置 dt、初始状态、陀螺偏置和接触，调用 update 并验证 stateSum/stateDifference 后清理 |
| unit-testings/CMakeLists.txt | 通过 add_so_test(test-kinetics-observer) 注册单元测试 |

## 可核对执行顺序

- 单测入口是 test-kinetics-observer.cpp 的 main → testKineticsObserverCodeAccessor()，不是 mc_rtc 真机运行入口。
- 初始化时设置采样周期、质心运动学、陀螺偏置和接触 rest pose、弹簧阻尼参数，测试中的 update() 触发 updateMeasurements → ekf_.updateStateAndMeasurementPrediction → getEstimatedState → updateLocalKineAndContacts_ / updateGlobalKine_。
- 真正传感数据接入应针对每个 IMU 更新 setIMU，针对每个有效接触按周期调用 updateContactWithWrenchSensor 或 updateContactWithNoSensor；不再接触必须显式 removeContact。质心惯量及角动量通过 setCoMInertiaMatrix、setCoMAngularMomentum 等设置，最终 update 取估计。
- 类构造代码默认 finiteDifferencesJacobians_=true；论文小于 1ms 的耗时基准不能未经实测直接外推到默认库或目标机器人。
- 论文中对扰动力/接触漂移的效果来自作者实验，不是本次单元测试的实测结果。

## 复现边界

对接足式真机时仍需完善接触检测、动力学辨识、FK/质心模型、IMU+wrench 坐标系、时钟同步和安全回退。本次源码核读不代表 CMake/CTest 已通过，也不代表作者的全部数据集开放。
