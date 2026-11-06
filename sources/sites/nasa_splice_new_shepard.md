# NASA SPLICE × Blue Origin New Shepard 精确着陆技术飞行试验

> 来源归档（ingest）

- **页面：** <https://www.nasa.gov/centers-and-facilities/johnson/nasa-tipping-point-partnership-with-blue-origin-to-test-precision-lunar-landing-technologies/>
- **类型：** NASA 官方新闻
- **入库日期：** 2026-11-06
- **核对范围：** 页面全文（headless Chromium 渲染读取，2026-11-06）。
- **通读：** 已完成

## 核心摘录

1. **项目。** SPLICE = Safe and Precise Landing – Integrated Capabilities Evolution，NASA 的精确着陆传感器、算法与计算机套件。
2. **飞行。** 编者注："Blue Origin's New Shepard (NS-13) mission successfully launched and landed Tuesday, Oct. 13, 2020."；这是 Tipping Point 合作下两次飞行中的第一次，"the first integrated test for the computer with two of the three SPLICE sensor systems"。
3. **地形相对导航（TRN）。** 惯性测量单元 + 相机，软件（Draper 开发）把实时图像与预载地图比对定出火箭位置；页面称其与 Perseverance 2021 年火星着陆所用系统同类。
4. **导航多普勒激光雷达（NDL）。** NASA Langley 研制，向地面发射激光束、由回波测出着陆器速度与高度。
5. **未上本次飞行。** 危险检测激光雷达（hazard detection lidar）用于生成着陆区 3D 地图、选择安全着陆点。
6. **算法相关。** 页面只说测试 "advanced algorithms" 与 "descent and landing computer"；SCvx 作为该项目实验载荷的说法来自 [Malyuta 等教程](../papers/malyuta_2021_convex_trajectory_generation_tutorial.md)，本页未提 SCvx。

## 对 wiki 的映射

- [动力下降制导（Powered Descent Guidance）](../../wiki/concepts/powered-descent-guidance.md) — 导航与传感器
- [火箭回收控制纵深路线](../../roadmap/depth-rocket-landing.md) — Stage 5
