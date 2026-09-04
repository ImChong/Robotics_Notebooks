# MINERVA（arXiv:2609.03715）

> 来源归档（ingest）

- **标题：** MINERVA: How Small Can a Manipulation Policy Be and Still Solve LIBERO?
- **简称：** MINERVA
- **类型：** paper / vla / efficient-policy / libero
- **arXiv：** <https://arxiv.org/abs/2609.03715>
- **PDF：** <https://arxiv.org/pdf/2609.03715>
- **代码：** <https://github.com/k1000dai/MINERVA> — 归档见 [`sources/repos/k1000dai-minerva.md`](../repos/k1000dai-minerva.md)
- **权重：** <https://huggingface.co/k1000dai/MINERVA>
- **机构：** 东京大学（The University of Tokyo）松尾–岩泽研究室
- **入库日期：** 2026-09-04
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_9_papers_open_source_2026-09-04.md)
- **一句话说明：** 刻意压小的 task-ID 条件视觉运动策略，用来量 LIBERO 闭集 40 任务的容量下限；0.54M 参数约 95% 平均成功率。

## 开源状态（步骤 2.5，2026-09-04）

| 组件 | 状态 |
|------|------|
| GitHub | **已开源** Apache-2.0；`lerobot-train` / `lerobot-eval`、锁定 LIBERO 环境 |
| HF | `k1000dai/MINERVA` 检查点 |

**结论：已开源** — 可复现评测与训练配方。

## 核心摘录

### 摘录 1：容量曲线

- 论文：0.54M、2000 rollouts、四 suite 平均 **95.1%**，比大约 7700× 的 LeRobot π₀.₅ 低 2.4 个百分点。
- README 同协议 headline：**95.75%**（单训练种子）；~1M 饱和，&lt;0.25M 崩塌。
- flow matching 相对 L1 regression 无稳定优势；regression GPU 上最高快 3.8×。
- LIBERO-Plus 扰动下 46–56%，标准 LIBERO 可能高估鲁棒性。

**对 wiki 的映射：** [paper-minerva-libero](../../wiki/entities/paper-minerva-libero.md)

## 当前提炼状态

- [x] 仓库与 README 核查（2026-09-04）
- [x] wiki 映射：`wiki/entities/paper-minerva-libero.md`
