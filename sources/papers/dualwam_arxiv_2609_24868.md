# DualWAM: Dual-System World Action Models（arXiv:2609.24868）

> 来源归档（ingest）

- **标题：** DualWAM: Dual-System World Action Models for Asynchronous Global Planning and Local Refinement
- **类型：** paper / wam / manipulation / async-inference
- **arXiv abs：** <https://arxiv.org/abs/2609.24868>
- **PDF：** <https://arxiv.org/pdf/2609.24868>
- **项目页：** <https://steveouo.github.io/DualWAM-Web/> — 归档见 [`sources/sites/dualwam-steveouo-github-io.md`](../sites/dualwam-steveouo-github-io.md)
- **代码：** **未列链接** — 项目页仅 BibTeX；GitHub 仅有网站模板仓 `SteveOUO/DualWAM-Web`
- **机构：** 中国科学院自动化研究所（CASIA）、银河通用（Galbot）、北京大学（PKU）、上海交通大学（SJTU）等
- **入库日期：** 2026-09-24
- **一句话说明：** 双系统 WAM：System 2 低频全局 world–action 规划（高噪声双向去噪），System 1 高频腕部观测局部低噪声 refinement；Franka + Galbot G1 零样本任务 SR +4.5 pp、关键路径 **16.6×** 加速。

## 核心摘录

### 1) 双系统分工

| | System 2 | System 1 |
|---|----------|----------|
| 角色 | 全局规划 | 局部 refinement |
| 状态 | $[Z^g, Z^w, A]$ | $[Z^w, A]$ 对齐窗口 |
| 噪声区间 | $[\tau, 1]$ | $[0, \tau)$ |
| 更新率 | 低（~1 Hz 全局计划） | 高（每控制步） |
| 观测 | 多视角全局 | **仅腕部** |

### 2) headline 数字

- 相对最强 baseline 平均 SR **+4.5 pp**；关键路径 latency **16.6×** speedup。
- 角色匹配 egocentric + UMI 数据 **+14 pp**。
- 边云部署：平均下行流量约为 baseline **1/104.6**。

### 3) 开源状态（2026-09-24）

- 项目页 **无** 训练/推理代码链接 → **待发布**。

## 对 wiki 的映射

- 新建：[paper-dualwam](../../wiki/entities/paper-dualwam.md)
- 交叉：[world-action-models](../../wiki/concepts/world-action-models.md)、[vla](../../wiki/methods/vla.md)、[manipulation](../../wiki/tasks/manipulation.md)
