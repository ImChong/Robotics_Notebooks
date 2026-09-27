# MuJoCo Menagerie — ALOHA 2（`aloha/` 子目录）

- **标题：** ALOHA Description (MJCF)
- **类型：** repo / asset-subtree
- **链接：** <https://github.com/google-deepmind/mujoco_menagerie/tree/main/aloha>
- **README：** <https://github.com/google-deepmind/mujoco_menagerie/blob/main/aloha/README.md>
- **父仓库归档：** [`mujoco-menagerie.md`](mujoco-menagerie.md)
- **项目页：** <https://aloha-2.github.io/>
- **入库日期：** 2026-09-27
- **License：** BSD-3-Clause（子目录 LICENSE）
- **一句话说明：** 双臂 ALOHA 2 工位 MJCF：由 ViperX 300 分叉、ALOHA 2 夹爪与 **11 轨迹系统辨识** 参数，含顶/仰视/双腕四相机且内参匹配 **RealSense D405**。

## 为什么值得保留

- **Sim 与真机同构：** 执行器与摩擦来自真机最小二乘拟合，不是默认 ViperX 参数直拷。
- **视觉对齐：** 四路相机内参按 D405 标定，腕部近距操作与 [手眼标定](../../wiki/concepts/hand-eye-calibration.md) 工程直接相关。
- **入口文件：** `scene.xml`（铝型材机架 + 木桌 + 双臂）；单臂 kinematic 树见子目录 XML。

## 对 wiki 的映射

- [ALOHA 2](../../wiki/entities/aloha-2.md)
- [MuJoCo Menagerie](../../wiki/entities/mujoco-menagerie.md)
- [Intel RealSense](../../wiki/entities/intel-realsense.md)
