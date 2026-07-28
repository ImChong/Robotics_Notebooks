# iCrowdNav（BRoln7/icrowdnav）

> 来源归档（repo）

- **标题：** iCrowdNav — Learning Robot Visual Navigation in Crowds via Intention-Aware Scene Representations
- **类型：** repo / visual-crowd-navigation / drl / sim2real（占位）
- **来源：** BRoln7（GitHub）
- **链接：** <https://github.com/BRoln7/icrowdnav>
- **项目页：** <https://broln7.github.io/socialbev.io/>
- **论文：** [arXiv:2606.26047](https://arxiv.org/abs/2606.26047) · RA-L 2026
- **演示视频：** <https://www.youtube.com/watch?v=8q0dhAiWCEA>
- **Stars：** ~13（2026-07-28）
- **入库日期：** 2026-07-28
- **一句话说明：** iCrowdNav 官方代码仓；README 徽章指向 **Isaac Sim 4.0 / Pegasus / Python 3.8 / Ubuntu 20.04 / ROS 1 / stable-baselines3 2.0**，但正文 **TODO: Release codes of iCrowdNav**——截至入库日 **仅附录与演示素材，无可运行实现**。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-icrowdnav.md`](../../wiki/entities/paper-icrowdnav.md)

---

## 开源状态（步骤 2.5）

| 项 | 状态（2026-07-28） |
|----|-------------------|
| 训练 / 推理代码 | **未发布**（README TODO） |
| 权重 / checkpoint | **未发布** |
| 附录 | 已提供 `icrowdnav_appendix.pdf` |
| 演示资产 | `assets/` 下 GIF / 封面图 |
| 许可证 | 仓库未在 README 明示（待代码发布时再核） |

**结论：** **部分开放（仅项目页素材 + 附录）/ 代码待发布**。wiki「源码运行时序图」标 **不适用**，待正式 release 后补。

---

## README 宣称的技术栈（待代码验证）

| 组件 | 徽章 / 文案 |
|------|-------------|
| 仿真 | Isaac Sim **4.0.0**；Pegasus Simulator |
| 训练 | **stable-baselines3 2.0.0**（与论文 PPO 叙述一致方向） |
| 运行时 | Python 3.8；Ubuntu 20.04；**ROS 1** |
| 真机感知（论文） | 双 Intel RealSense D435；YOLO 姿态；板载 RTX 2060 ~15 Hz |

---

## 目录快照（浅克隆）

```
icrowdnav/
  README.md                 # 论文链接 + TODO Release codes
  icrowdnav_appendix.pdf
  assets/                   # social-bev.jpg, sfm_demo.gif, demo GIFs
```

---

## 交叉链接

- 论文归档：[`sources/papers/icrowdnav_arxiv_2606_26047.md`](../papers/icrowdnav_arxiv_2606_26047.md)
- 项目页：[`sources/sites/broln7-socialbev-io.md`](../sites/broln7-socialbev-io.md)
- wiki 实体：[`wiki/entities/paper-icrowdnav.md`](../../wiki/entities/paper-icrowdnav.md)
