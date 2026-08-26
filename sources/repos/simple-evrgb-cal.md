# nhessenthaler/simple-evrgb-cal

> 来源归档

- **标题：** Simplified Event-RGB Calibration 官方工具
- **类型：** repo
- **代码：** <https://github.com/nhessenthaler/simple-evrgb-cal>
- **论文：** [arXiv:2608.22965](https://arxiv.org/abs/2608.22965) — 归档见 [`sources/papers/simple_evrgb_cal_arxiv_2608_22965.md`](../papers/simple_evrgb_cal_arxiv_2608_22965.md)
- **入库日期：** 2026-08-26
- **一句话说明：** 显示器调制 ChArUco 的跨模态标定 GUI；`main.py` + `src/core/calibration.py`；默认 IDS uEye × Prophesee EVK4。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [simple-evrgb-cal 实体页](../../wiki/entities/paper-simple-evrgb-cal.md) | 方法与标定数字 |
| [AMI-EV](../../wiki/entities/paper-microsaccade-inspired-event-camera.md) | 事件相机传感对照 |

## 复现入口（README 摘要）

- 依赖：uv、libjpeg-turbo 3.x、IDS uEye SDK、OpenEB（Metavision）
- 入口：`main.py`；预生成靶标在 `data/`
- 换相机：替换 `src/core/ueye.py` / `prophesee.py`，标定例程可复用
- 许可：Apache-2.0

## 开源状态

**已开源** — 可运行标定工具；硬件驱动绑默认传感器，算法层可移植。
