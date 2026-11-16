# From Depth to Safe Depth: Uncertainty-Aware Monocular Perception for Mobile Robot Navigation — 官方代码归档

- **仓库：** https://github.com/saeed5959/depth_safe_robot
- **论文：** https://arxiv.org/abs/2611.13769
- **核对范围：** 已查公开 README / 项目目录与入口；未完成逐文件代码审阅、环境安装或复测。
- **通读：** 未完成
- **开源状态：** 官方公开仓库存在；具体依赖与完整可复现性以 README 为准。

## 执行入口

`ros2 launch ./eval_stack.launch.py world:=test_02 method:=mono_safe rviz:=true`

仿真中采用真值里程计，跨地图/相机/材质的校准失配与真机可靠性仍需验证。

## 对 wiki 的映射

- [论文独立实体](../../wiki/entities/paper-safe-depth.md)
