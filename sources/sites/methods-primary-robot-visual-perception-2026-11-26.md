# Robot Visual Perception · 机器人视觉感知流程：一手资料复核

- **类型：** 方法一手资料归档
- **核对日期：** 2026-11-26
- **通读：** 已完成
- **核对范围：** 以下所列文档、接口和实现范围；未运行实机或评测。
- **对应页面：** [方法页](../../wiki/methods/robot-visual-perception.md)

## 一手资料

- [ROS 2 sensor_msgs CameraInfo](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/CameraInfo.msg)：阅读消息定义、光学坐标、原始与校正图像参数、ROI 与 binning。
- [ROS image_geometry PinholeCameraModel](https://github.com/ros-perception/vision_opencv/blob/rolling/image_geometry/src/pinhole_camera_model.cpp)：核对 projectPixelTo3dRay 的输入为校正像素，输出 z=1 的射线。

## 交叉核对

- 首先将图像与匹配的 `CameraInfo` 对齐。区分原始图像内参 K 和畸变 D、立体校正 R 与校正图像的投影矩阵 P；分辨率、裁切和校正模式变化后，不能沿用旧像素标定。
- 单帧高置信检测不构成距离、法向或姿态可靠性。对运动物体，使用当前 TF 配准旧图像会引入随速度增加的误差。
