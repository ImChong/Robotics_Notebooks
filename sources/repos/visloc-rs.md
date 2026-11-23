# visloc-rs — 官方源码、实验口径与运行时入口核对

> 来源归档（repo / GitHub 官方仓库一手资料）

- **项目：** visloc-rs
- **作者/维护者：** rsasaki0109 / visloc-rs contributors（以仓库信息为准）
- **类型：** repo
- **仓库：** https://github.com/rsasaki0109/visloc-rs
- **官方主页：** https://rsasaki0109.github.io/visloc-rs/
- **文档：** https://github.com/rsasaki0109/visloc-rs/tree/main/docs
- **核对版本：** `6a63f7f2b5f9b94152ed7e3d930d2f883f331172`（2026-11-23 获取的主分支 HEAD）
- **许可证：** MIT OR Apache-2.0（Cargo.toml、LICENSE-MIT、LICENSE-APACHE）
- **入库日期：** 2026-11-23
- **核对范围：** 通读官方 README（含完整功能、传感器矩阵、运行指令、对比和已知劣势）；阅读 `Cargo.toml`、`docs/photos_to_splat.md`、`docs/navigation.md`、`crates/gsplat-train/examples/gsplat_photos.rs`、`examples/localize_dummy.rs` 的命令入口与主要代码流程；官方 Web 首页、SfM 和 VIO 演示页面另存于 [站点归档](../sites/visloc-rs.md)。**没有克隆运行、下载测试数据或独立复现 benchmark；未对全仓源码逐行审计。**
- **通读：** 已完成
- **代码开放：** **已开源**，Rust 1.88+，主流程依赖 Rust/wgpu；有 Python bindings 和单独 ROS 2 组件，但环境适配需另测。
- **对应知识页：** [visloc-rs](../../wiki/entities/visloc-rs.md)

## 实现与数据流核对

- **SfM / 图像建图：** 官方 README 和 `docs/photos_to_splat.md` 描述 EXIF/相机内参 → SIFT → GPU 图像匹配与几何验证 → COLMAP-port 增量 mapper → BA / 内参优化；`build_photo_dataset` 生成训练视图、稀疏地标及注册报告。
- **3DGS 与 mesh：** `gsplat_photos` 主函数先 `build_photo_dataset`，再 `seed_scene`、`Trainer::new` 和循环 `trainer.step`；调用 `save_ply` 写 `scene.ply`；非 `--no-mesh` 时调用 `extract_from_splat` 并 `write_ply` 写 `mesh.ply`。
- **定位最小入口：** `examples/localize_dummy.rs` 建立相机、8 个 3D landmark 和对应 2D keypoint / descriptor，构造 `QueryImage`、`VisualMap`，调用 `localize(query, map)` 输出定位成功、内点、重投影误差和位姿。
- **VIO / 地图定位：** README 指出 Basalt Rust 移植的双目惯性紧耦合 VIO + 在线关键帧 mapper；可通过 `basalt_euroc_online_slam_demo` 保存地图，再运行 `basalt_euroc_localize_demo` 或 ROS 2 组件定位。
- **Nav2 接入：** `docs/navigation.md` 明确由 `visual_map_to_occupancy` 生成占据栅格，ROS 2 节点供给 `map -> odom`；**visloc-rs 本身不是 Nav2 路径规划/控制实现**。

## 可复现入口（官方命令，未运行）

```bash
# 仅验证合成查询定位
cargo run --example localize_dummy

# 图片 -> SfM 相机位姿 -> 3DGS -> 三角 mesh
cargo run --release -p visloc-gsplat-train --features gpu,euroc --example gsplat_photos -- \
  --images /path/to/my_photos --out runs/my-scene

# EuRoC 双目惯性在线 SLAM（需要外部数据集）
cargo run --release --example basalt_euroc_online_slam_demo \
  --features basalt-lm-workspace-reuse -- \
  --euroc-dir /path/to/MH_01_easy \
  --calibration configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json \
  --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
  --out-dir target/basalt_mh01_online
```

## 作者报告的实验与适用条件

| 任务 | 官方报告 | 对照与边界 |
|---|---|---|
| SmallCity 图片重建 | 5,822 张 → 注册 5,583 相机 → 470 万高斯 → 590 万三角面，2.5 小时 | `docs/photos_to_splat.md`：Google Colab A100；SfM 82 min、训练 30 min、mesh 34 min；单次报告，不是普适时长 |
| ETH3D Electro SfM | 1,200 张，3.46× faster、相机中心 RMSE 低 25.2% | 官方与 COLMAP 的给定 CPU8 实验设置，不能外推全部场景 |
| EuRoC 在线双目惯性 SLAM | 11/11 在作者机器上达到实时；作者测得 9/11 ATE 优于 ORB-SLAM3 | MH_04/MH_05 失败于该比较；不同运行 profile 的实时性能不等同 |
| OpenLORIS 地图定位 | 1,250 held-out 帧 98.96% 定位，median 2.9 mm | 机器人 rig 静态地图实验，不是任意环境绝对精度 |
| Nav2 联动 | 模拟 apartment 5/5 goal、3.8 cm pose RMSE；Gazebo 亦有 5/5 | `docs/navigation.md`：**暂无基于 visloc 定位驱动 Nav2 的真机导航验证** |

- 官方也明确指出：GPU SfM 在 EuRoC 8 组中有 3 组比 COLMAP 更不准确；多次会话与大规模数据需考察闭环、纹理、内存和地图覆盖。
- `ros2/visloc-ros2` 通过 RustDDS 内部 DDS 验证过，README 指明**未验证与实际 ROS 2 安装互通**。
- 项目另展示 `real2sim.py` 以 360° 实景重建的 3DGS/地图运行 Nav2 与移动行人仿真，但**不能将视觉 mesh 或 3DGS 直接等同于通用的物理可交互数字孪生资产**。

## 官方证据索引

- [README：总览、支持矩阵及复现命令](https://github.com/rsasaki0109/visloc-rs/blob/main/README.md)
- [Photos to splat and mesh：SmallCity 日志、管线和局限](https://github.com/rsasaki0109/visloc-rs/blob/main/docs/photos_to_splat.md)
- [gsplat_photos 运行入口源码](https://github.com/rsasaki0109/visloc-rs/blob/main/crates/gsplat-train/examples/gsplat_photos.rs)
- [localize_dummy 运行入口源码](https://github.com/rsasaki0109/visloc-rs/blob/main/examples/localize_dummy.rs)
- [导航集成与实验边界](https://github.com/rsasaki0109/visloc-rs/blob/main/docs/navigation.md)
- [VI-SLAM 评测](https://github.com/rsasaki0109/visloc-rs/blob/main/docs/vi_slam_benchmarks.md)
- [SfM 评测](https://github.com/rsasaki0109/visloc-rs/blob/main/docs/sfm_benchmarks.md)
