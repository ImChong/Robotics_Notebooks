# Intel RealSense Depth Camera D405（产品规格页）

- **标题：** RealSense™ Depth Camera D405
- **类型：** site / product-spec
- **URL：** <https://www.realsenseai.com/products/stereo-depth-camera-d405/>
- **SDK：** RealSense SDK 2.0（`librealsense`）
- **入库日期：** 2026-09-27
- **一句话说明：** 短距立体深度相机：理想工作距离 **7–50 cm**，全局快门 RGB+深度，87°×58° FOV，适合腕部操作与近场 pick-and-place。

## 规格摘录（页面 datasheet 区，截至入库日）

| 项 | 数值 |
|----|------|
| 深度技术 | 立体（Stereoscopic） |
| 理想量程 | 7 cm – 50 cm |
| Min-Z @ 480p | 7 cm |
| 深度精度 | ±2% @ 50 cm |
| 深度 FOV | 87° × 58°（最高 1280×720 @ 90 fps） |
| RGB | 左目 RGB + ISP；1280×720 @ 30 fps（页内亦列 90 fps 能力） |
| RGB / 深度传感器 | **Global Shutter** |
| 外形（外设） | 42 × 42 × 23 mm，约 60 g |
| 功耗 | 空闲 ~35 mW；Depth+IR 流 ~1.55 W |
| 接口 | USB 2 / USB 3.1 |
| 安装 | 1/4‑20 UNC ×1；M3 ×2 |
| 环境 | 室内/室外；多机可同时使用（无 IR 互扰设计） |

## 机器人语境（为何与 ALOHA 2 同 ingest）

- ALOHA 2 用 **更小 D405** 替换原 USB 网络摄像头：腕部 footprint 更小、**深度 + 全局快门** 利于动态操作与 Sim 对齐。
- MuJoCo Menagerie `aloha/` 声明仿真相机 **内参与 D405 匹配** — 真机外参仍依赖 [手眼标定](../../wiki/concepts/hand-eye-calibration.md)。

## 对 wiki 的映射

- [Intel RealSense 实体](../../wiki/entities/intel-realsense.md)
- [ALOHA 2](../../wiki/entities/aloha-2.md)
