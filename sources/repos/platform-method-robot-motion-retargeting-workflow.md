# GMR · SMPL-X 示例入口与求解器实现

> 一手资料归档 · 2026-11-26

- [官方离线入口](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/scripts/smplx_to_robot.py)
- [官方求解器](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/motion_retarget.py)
- **核对范围：** 固定提交 25f56db79ac411258b377b6d4ccba0c8ff3372ac，离线入口与重定向实现：加载、配对配置、目标更新、迭代求解、预览与保存；未运行可视化或硬件。
- **通读：** 已完成

## 核对摘记

离线入口载入人体数据并对齐帧率，以实例方法 retarget 逐帧求得 qpos。保存时根旋转由 wxyz 改为 xyzw，重定向实现不求动力学可行性。

## 对 wiki 的映射

- [机器人动作重定向流程](../../wiki/methods/robot-motion-retargeting-workflow.md)
