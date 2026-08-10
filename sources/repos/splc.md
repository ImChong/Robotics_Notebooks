# SPLC（sklus949/SPLC）

> 来源归档（repo）

- **标题：** SPLC — Social Preference Learning for Crowd Robot Navigation
- **类型：** repo / crowd-navigation / offline-rl / preference-learning（占位）
- **来源：** sklus949（GitHub）
- **链接：** <https://github.com/sklus949/SPLC>
- **论文：** [arXiv:2607.01925](https://arxiv.org/abs/2607.01925)
- **演示视频：** <https://youtu.be/vkWjg4Qcybg>
- **Stars：** ~1（2026-08-10）
- **入库日期：** 2026-08-10
- **一句话说明：** SPLC 官方代码仓；README 写明 **The source code is coming soon**——截至入库日仅有 README 与 Graphical Abstract，**无可运行训练/推理入口**。注释掉的历史说明暗示未来可能提供 `mechanism.py` / `train_reward_model.py` / `offline/iql.py`。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-splc.md`](../../wiki/entities/paper-splc.md)

---

## 开源状态（步骤 2.5）

| 项 | 状态（2026-08-10） |
|----|-------------------|
| 训练 / 推理代码 | **未发布**（README：coming soon） |
| 权重 / 离线数据集 | **未发布** |
| 演示 | YouTube 视频 + Graphical Abstract 图 |
| 许可证 | 仓库未明示 |
| 拟议脚本（README 注释，不可代替发布） | `crowd_nav/mechanism.py` → `train_reward_model.py` → `offline/iql.py` |

**结论：** **宣称将开源 / 截至入库日无可运行实现**。wiki「源码运行时序图」标 **不适用**，待正式 release 后按注释入口补 sequenceDiagram。

---

## 目录快照（API 树）

```
SPLC/
  README.md
  Graphical Abstract.png
```

---

## 交叉链接

- 论文归档：[`sources/papers/splc_arxiv_2607_01925.md`](../papers/splc_arxiv_2607_01925.md)
- wiki 实体：[`wiki/entities/paper-splc.md`](../../wiki/entities/paper-splc.md)
