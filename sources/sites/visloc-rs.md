# visloc-rs — 官方项目站点与浏览器演示（2026-11-23）

> 来源归档（site / 官方一手资料）

- **标题：** visloc-rs — SfM, visual-inertial SLAM, 3D Gaussian Splatting and map-based localization in Rust
- **类型：** site
- **项目主页：** https://rsasaki0109.github.io/visloc-rs/
- **代码：** https://github.com/rsasaki0109/visloc-rs
- **SfM Web Demo：** https://rsasaki0109.github.io/visloc-rs/sfm/
- **VIO Web Demo：** https://rsasaki0109.github.io/visloc-rs/slam/
- **3DGS Viewer：** https://rsasaki0109.github.io/visloc-rs/splat/
- **官方网页核查日期：** 2026-11-23
- **核对范围：** 通读官方项目首页、SfM 网页演示说明和 VIO 演示说明；比对 GitHub README 的网页入口与功能声明。未在浏览器实际运行重建或移动设备 VIO。
- **通读：** 已完成
- **开源状态：** **已开源**（Rust 实现 MIT OR Apache-2.0；网页演示基于 WebAssembly / WebGPU，浏览器设备兼容性需另验）
- **对应知识页：** [visloc-rs](../../wiki/entities/visloc-rs.md)

## 官方网页确认的信息

1. 首页展示 SfM、视觉惯性里程计和 3D Gaussian Splatting 的演示入口。
2. SfM 浏览器版在本地设备上运行 SIFT、图像匹配、两视图几何验证和增量重建，支持导出 COLMAP、点云和相机模型；网页声明用户图片不会上传。
3. VIO 浏览器版使用相机与运动传感器，说明了初始运动以恢复度量尺度、相机/IMU 标定、跟踪丢失等使用边界。站点提示可选诊断上传及关闭选项，不能笼统宣传为“任何遥测数据都不会上传”。
4. 在线 3DGS 展示的是重建场景的渲染能力，不意味着直接生成机器人碰撞体、物理质量、摩擦和关节约束。

## 关联原始资料

- [官方仓库/源码与复现路径](../repos/visloc-rs.md)
- [官方 README](https://github.com/rsasaki0109/visloc-rs/blob/main/README.md)
- [官方导航仿真说明](https://github.com/rsasaki0109/visloc-rs/blob/main/docs/navigation.md)
