# GR00T-WholeBodyControl 文档站（nvlabs.github.io）

> 来源归档（ingest 配套站点）

- **URL：** <https://nvlabs.github.io/GR00T-WholeBodyControl/>
- **标题：** GR00T Whole-Body Control Documentation
- **类型：** docs site
- **代码：** <https://github.com/NVlabs/GR00T-WholeBodyControl>
- **关联：** [GEAR-SONIC](https://nvlabs.github.io/GEAR-SONIC/) · [MotionBricks](https://nvlabs.github.io/motionbricks/)
- **机构：** NVIDIA NVlabs / GEAR
- **入库日期：** 2026-07-22
- **一句话说明：** NVlabs 人形全身控制统一平台官方文档：Decoupled WBC、GEAR-SONIC 训练/部署、MotionBricks 预览、VR 遥操作、VLA 数据链与真机 C++ 推理；与 Git 单仓 README 同步。

## 开源状态（2026-07-22）

| 产物 | 状态 |
|------|------|
| 代码 | **已开源** · [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)（~2.9k★） |
| SONIC 权重 | HF `nvidia/GEAR-SONIC` |
| 文档 | 本站（Installation Training/Deploy、Tutorials、Model Card） |
| 在线 Demo | [GEAR-SONIC Live Demo](https://nvlabs.github.io/GEAR-SONIC/demo.html) |

## 页面要点（2026-07 快照）

- **三大块：** Decoupled WBC（N1.5/N1.6）· GEAR-SONIC · MotionBricks
- **近期文档更新叙事：** 低延迟遥操作 checkpoint（4-frame SMPL lookahead）；G1 端到端 VLA workflow（采集 → 微调 Isaac-GR00T N1.7 → SONIC 部署）
- **工程约束：** Git LFS；按用途拆分独立 venv；代码与权重许可分离

## 对 wiki 的映射

- [GR00T-WholeBodyControl 实体](../../wiki/entities/gr00t-wholebodycontrol.md)
- [仓库归档](../repos/gr00t_wholebodycontrol.md)
- [SONIC 方法页](../../wiki/methods/sonic-motion-tracking.md)
