# Rolling-WAM（arXiv:2609.30247）

> 来源归档（ingest）

- **标题：** Rolling-WAM: World Action Models with Rolling Imagination
- **缩写：** **Rolling-WAM**
- **类型：** paper / world-action-model / manipulation / real-time / humanoid-g1
- **arXiv：** <https://arxiv.org/abs/2609.30247>
- **PDF：** <https://arxiv.org/pdf/2609.30247>
- **项目页：** <https://rolling-wam.github.io/> — 归档见 [`sources/sites/rolling-wam-github-io.md`](../sites/rolling-wam-github-io.md)
- **代码占位仓：** <https://github.com/zyinghua/Rolling-WAM> — 归档见 [`sources/repos/rolling-wam.md`](../repos/rolling-wam.md)
- **作者：** Yinghua Zhou*、Junjie Ye*、Yiqi Zhao、Hao Dong、Celina Shiyu Wang、Ruohai Ge、Tingyi Yang、Basile Van Hoorick、Gaurav Sukhatme、Vitor Guizilini†、Yue Wang†
- **机构：** 南加州大学（USC）；布朗大学；复旦大学；丰田研究院（TRI）
- **入库日期：** 2026-09-26
- **开源状态（步骤 2.5，2026-09-26）：** GitHub 存在但 README 首行 **「Code and checkpoints are being prepared and will be released soon」** → **待发布**。

## 核心论文摘录（MVP）

### 1) Rolling imagination：跨 replan 分摊联合去噪

- **链接：** <https://arxiv.org/abs/2609.30247>
- **核心贡献：** 标准 Joint-WAM 每轮 replan 需 **从头联合去噪整段 video–action horizon**，steady-state **~978 ms**（RoboTwin 2.0 对照表）。Rolling-WAM 维护 **滑动窗口内 staggered noise level** 的 video–action chunks：每步 **完全去噪 imminent action chunk 执行**，远处 chunk **部分 refine**；窗口随新观测滚动，**保留的未来 chunk 继续去噪**，把算力摊到多步并携带跨 chunk 上下文。
- **对 wiki 的映射：**
  - [Rolling-WAM 论文实体](../../wiki/entities/paper-rolling-wam.md)
  - [World Action Models](../../wiki/concepts/world-action-models.md)
  - [MotionWAM](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md)（对照：hook 单次前向 vs rolling 多步 partial denoise）

### 2) MoT 架构与 mask

- **核心贡献：** **Mixture-of-Transformers**：预训练 **video DiT** + 轻量 **action Transformer**；语言与机器人状态条件双 expert；**masked joint attention** — 未来 video token 可看当前观测与已预测 video chunk、**不可看 action**；action token 看全部 visual，action–action 仅 chunk 内；**per-chunk noise conditioning** 支持窗口内不同去噪阶段。

### 3) 评测 headline（项目页，2026-09-26）

- **LIBERO 平均 SR 98.1%**；**RoboTwin 2.0 平均 93.3%**；**真机 Unitree G1 平均 85.0%**。
- **Steady-state replanning latency 215 ms**（A100，对照 Joint-WAM 978 ms、Fast-WAM 548 ms）→ **4.5×** speedup。
- **对 wiki 的映射：** [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)（G1 真机操纵）
