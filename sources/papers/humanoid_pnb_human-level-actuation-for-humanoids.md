# Human-Level Actuation for Humanoids

> 来源归档（ingest · Robot Learning Paper Notebooks 深读笔记）

- **标题：** Human-Level Actuation for Humanoids
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Human-Level_Actuation_for_Humanoids/Human-Level_Actuation_for_Humanoids.html>
- **分类：** 12_Hardware_Design
- **arXiv：** <https://arxiv.org/abs/2511.06796>
- **项目页 / 源码：** 截至 2026-11-21 未发现；本文为评测框架与测量协议
- **入库日期：** 2026-07-10（2026-11-21 完成 `arXiv:2511.06796v1` 全文通读与公式/数值核对）
- **通读：** 已完成
- **一句话说明：** 针对商用与学术人形常宣传「人级驱动」却缺乏统一度量与防作弊协议的问题，提出 DoF Atlas（ISB 规范 + 106R/4T 自由度清单与 ADL 功能 ROM）、Human-Equivalence Envelopes（HEE，要求同一 `(q, ω)` 点同时满足人类力矩与正机械功要求）及 Human-Level Actuation Score（HLAS，汇总 $\rho_{\text{ROM}}, d_{\text{DoF}}, h^{(w)}, b^\tau, \eta, \theta^{\text{therm}}$ 六因子并在人类参考上严格归一化为 1.0），并给出 Benchmark 1–9 可复测台架与整机协议。

## 核心摘录（原文 PDF 精读实证）

- **参考人基准与缩放**：以 **75 kg、1.75 m**（5'9"）成年男性为参考基准，采用 de Leva（1996）修正 Zatsiorsky–Seluyanov 回归按体重比例缩放到不同体型。
- **ISB DoF Atlas（§2 & Table 1–2）**：全人体双侧共 **106R + 4T** 自由度（其中手脚指段占 68R）；区分 AROM / PROM / 功能 ROM（ADL），只对任务真实使用的轴向与功能角度区间计分。
- **任务包络（§3 & Table 3–4）**：1.2–1.4 m/s 平地走中踝 push-off（0–25° 跖屈、8–12 rad/s）贡献约 45–50% 正功（1.2–1.9 Nm/kg → **90–143 Nm**，2.5–3.5 W/kg → **190–260 W**），上楼踝推进 **105–135 Nm / 225–260 W**、膝维持 **~75 Nm @ 0.5–2 rad/s**，跑动（2.5–3.0 m/s）踝髋正功达 **300–525 W**，重复举升膝伸力矩 **225–255 Nm**（需 ~150 Nm 连续 / 300 Nm 峰值），手开合 **4–6 Hz**，投掷肩内旋速率极值 **113–134 rad/s**。
- **HLAS 数学定义与防刷分（§4）**：默认特征权重 $\alpha = [0.10, 0.10, 0.50, 0.10, 0.10, 0.10]^\top$，重点加权 $h^{(w)}_{j,t}$（Eq. 1 同时性条件，堵死高减速比堆静态扭矩 + 轻载高转速拼参数）；全部测功必须经 3–5 min 热浸泡且温升斜率 `< 0.5 °C/s`（Benchmarks 1–5）、预注册权重，可选安全余量 $(1+\delta)$（如 $\delta=0.10$）。
- **Worked Example（§7, Table 6–7）**：12-DoF 假想人形在 Walk ($w=0.4$)、Stairs ($w=0.3$)、Reach ($w=0.3$) 三任务上测得总分 **HLAS = 0.636**（$s_{\text{Walk}}=0.671$、$s_{\text{Stairs}}=0.539$、$s_{\text{Reach}}=0.687$；其中 Walk 踝 $h^{(w)}=0.546$，而 Stairs 膝/髋因低速持续力矩不足导致 $h^{(w)}\approx 0.087/0.085$）；施加 $\delta=0.10$ 裕量后降至 **0.515**（暴露约 19% 基线分贴边通过）。
- **知识归纳与实体页面**：[paper-notebook-human-level-actuation-for-humanoids](../../wiki/entities/paper-notebook-human-level-actuation-for-humanoids.md).

## 对 wiki 的映射

- [paper-notebook-human-level-actuation-for-humanoids](../../wiki/entities/paper-notebook-human-level-actuation-for-humanoids.md)
- 分类父节点：[paper-notebook-category-12-hardware-design](../../wiki/overview/paper-notebook-category-12-hardware-design.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Human-Level_Actuation_for_Humanoids/Human-Level_Actuation_for_Humanoids.html>
- 论文：<https://arxiv.org/abs/2511.06796>
