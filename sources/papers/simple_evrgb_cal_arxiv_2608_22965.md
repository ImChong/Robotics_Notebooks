# simple-evrgb-cal（无运动的事件—RGB 跨模态标定）

> 来源归档（ingest）

- **标题：** Simplified Cross-Modal Calibration for Heterogeneous Event-RGB Stereo Systems
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2608.22965>
- **代码：** <https://github.com/nhessenthaler/simple-evrgb-cal>
- **机构：** 海尔布隆应用科学大学（Heilbronn University of Applied Sciences）
- **入库日期：** 2026-08-26
- **一句话说明：** 在普通显示器上以 15 Hz 在原始与部分混合的 ChArUco 之间切换，使靶标持续对 RGB 可见并稳定触发事件；事件粗粒度成帧 + 中值去噪后走标准 OpenCV 内参/外参标定。

## 核心摘录（MVP）

### 1) 运动式重建 vs 闪烁空白帧

- **摘录要点：** E2Calib 等需运动 + E2VID 重建；静态闪烁 LED/显示器常要专用硬件或严格同步，且 blank 周期会让 RGB 丢特征。本文用 \(\alpha\)-混合叠加，避免全黑帧，同步只需粗触发对齐。
- **对 wiki 的映射：**
  - [simple-evrgb-cal](../../wiki/entities/paper-simple-evrgb-cal.md)
  - [AMI-EV](../../wiki/entities/paper-microsaccade-inspired-event-camera.md) — 事件相机静止纹理问题对照。

### 2) 管线：调制靶标 → 事件帧 → ChArUco

- **摘录要点：** \(I_\alpha=(1-\alpha)I_p+\alpha I_w\)，\(f=15\) Hz（30 次状态切换/秒）。事件按 RGB 帧间隔累积，中值滤波后 OpenCV。推荐 \(\alpha=0.6\)：事件检测 100%，立体重投影 **0.38 px**。
- **对 wiki 的映射：**
  - [simple-evrgb-cal](../../wiki/entities/paper-simple-evrgb-cal.md) — 流程总览。
  - [仓库归档](../repos/simple-evrgb-cal.md)

### 3) 对照与眼在手案例

- **摘录要点：** 相对最强运动式参考（E2Calib+Kalibr）平均重投影误差 **↓44%**，相对 Plasberg 静态参考 **↓6%**。亮度 ≥20% 时角点成功率 >98%；视角稳定到约 50°。手持 OLED 略优于固定 LCD。机器人眼在手标定在部分遮挡下几何测量仍稳。
- **对 wiki 的映射：**
  - [simple-evrgb-cal](../../wiki/entities/paper-simple-evrgb-cal.md) — 评测。

### 4) 开源状态（截至 2026-08-26）

- **摘录要点：** **已开源** Apache-2.0。`main.py` GUI 标定工具；默认硬件 IDS uEye + Prophesee EVK4，但 `src/core/calibration.py` 可换相机驱动。依赖 uv、libjpeg-turbo、OpenEB。
- **对 wiki 的映射：**
  - [仓库归档](../repos/simple-evrgb-cal.md)

## 当前提炼状态

- [x] arXiv HTML + GitHub README 已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-simple-evrgb-cal.md` 新建
