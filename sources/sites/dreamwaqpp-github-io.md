# dreamwaqpp.github.io（DreamWaQ++ 项目页）

> 来源归档（ingest）

- **标题：** DreamWaQ++ — Obstacle-Aware Quadrupedal Locomotion
- **类型：** site / project-page
- **官方入口：** <https://dreamwaqpp.github.io/>
- **入库日期：** 2026-05-30
- **一句话说明：** T-RO 2026 配套站点：多模态感知–控制架构图解、楼梯竞速/障碍感知/探查行为/分布外鲁棒/陡坡/多机台与大障碍等 **视频与定量结果**；含 BibTeX 与 arXiv 链接。

## 页面公开资源

| 资源 | URL |
|------|-----|
| 项目首页 | <https://dreamwaqpp.github.io/> |
| 论文 abs | <https://arxiv.org/abs/2409.19709> |
| 主演示视频 | <https://youtu.be/DECFbMdpfps> |
| 盲走基线 DreamWaQ 对比视频 | 站点「More Videos」区（相对 DreamWaQ 盲走） |

## 站点章节（2026-05-30 检索）

- **Abstract：** 盲走 vs 外感知地图维护的两难；多模态融合与分布外恢复叙事；关键数字（35° 坡度外推、97.8% 楼梯、4 平台、50 Hz 控制）。
- **How It Works：** Sense（10 Hz 点云 + 200 Hz 本体）→ Encode（PointNet 置信滤波 + 随机本体潜变量）→ Act（MLP-Mixer 融合 + 50 Hz 策略 + 200 Hz PD）。
- **Stair Climbing Race：** 50 级楼梯与盲走 DreamWaQ、Unitree 内置控制器对比；主动抬身与摆腿 vs 盲走拖脚。
- **Obstacle Awareness：** 足端摆动轨迹在线适应（组合抬升最高约 30 cm）；1000 机仿真楼梯配置成功率 +20–40% vs 视觉基线。
- **Emergent Probing：** 不确定地形边缘 **停步探足**（无显式奖励）。
- **OOD Robustness：** 移动平台突然抽走时的支撑多边形扩大与潜变量聚类切换。
- **Extreme Slopes：** 10° 训练 → 35° 爬行步态，后足力矩约降 1.5×。
- **Multi-Robot Scalability：** R1–R4 实机配置表与成功率图。
- **Large Obstacles：** Go1 / ANYmal-C / Hound 仿真障碍高度与带载实机沙发攀爬。
- **Ablation：** 去掉潜变量融合楼梯成功率 97.8%→60.7%；PacMAP 与传感器失效 **foot-trapping** 回退。

## 对 wiki 的映射

- [`wiki/entities/dreamwaq-plus.md`](../../wiki/entities/dreamwaq-plus.md)
- [`sources/papers/dreamwaq_plus_arxiv_2409_19709.md`](../papers/dreamwaq_plus_arxiv_2409_19709.md)
