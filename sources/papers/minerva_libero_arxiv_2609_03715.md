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
- **入库日期：** 2026-09-04；**再核：** 2026-09-05
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_9_papers_open_source_2026-09-04.md)
- **一句话说明：** 刻意压小的 task-ID 条件视觉运动策略，用来量 LIBERO 闭集 40 任务的容量下限；0.54M 参数 headline 95.75%，CPU 5.1 ms/chunk。

## 开源状态（步骤 2.5，2026-09-04；2026-09-05 再核）

| 组件 | 状态 |
|------|------|
| GitHub | **已开源** Apache-2.0；`lerobot-train` / `lerobot-eval`、锁定 LIBERO 环境 |
| HF | `k1000dai/MINERVA` 检查点（revision `1b4fb1743f00a7d8eb87c7059c446447907d12bf`） |

**结论：已开源** — 可复现评测与训练配方。2026-09-05 再核仓库仍公开，README 补齐 CPU 测速与 teacher 蒸馏配方。

## 核心摘录

### 摘录 1：容量曲线

- 论文：0.54M、2000 rollouts、四 suite 平均 **95.1%**，比大约 7700× 的 LeRobot π₀.₅ 低 2.4 个百分点。
- README 同协议 headline：**95.75%**（单训练种子）；~1M 饱和，&lt;0.25M 崩塌。
- 1M 三种子：L1 与 flow 成功率不可分；chunk 16 优于 8/32。
- LIBERO-Plus 扰动下 46–56%，标准 LIBERO 可能高估鲁棒性。

**对 wiki 的映射：** [paper-minerva-libero](../../wiki/entities/paper-minerva-libero.md)

### 摘录 2：CPU 推理成本（README `results_speed_cpu.json`）

- 0.54M L1：GPU 2.2 ms/chunk，**CPU 5.1 ms/chunk**，VRAM **0.03 GB**。
- 同机 π₀.₅：GPU 196 ms，CPU 12,781 ms，VRAM 9.36 GB。
- GPU 对 MINERVA 推理可选；全量评测约 15 GB RAM。锁 `mujoco==3.3.2`。

**对 wiki 的映射：** [paper-minerva-libero](../../wiki/entities/paper-minerva-libero.md)

## 当前提炼状态

- [x] 仓库与 README 核查（2026-09-04）
- [x] 2026-09-05 再核：CPU 时延、蒸馏 teacher、架构细节写回 wiki
- [x] wiki 映射：`wiki/entities/paper-minerva-libero.md`
