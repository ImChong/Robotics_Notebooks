# Fail-Passive Gap 论文通读（系统工程 / 实时与安全视角）

> 来源归档（原文通读 · 支撑 `wiki/overview/hub-systems-engineering.md`）

- **标题：** Toward Certified Functional Safety for Industrial Humanoid Robots: The Fail-Passive Gap and a Feasibility Study
- **作者：** Caiwu Ding, Tao Cui, Lingyun Wang, Chengtao Wen（Siemens Foundational Technologies）
- **类型：** paper / functional-safety / humanoid
- **原文：** <https://arxiv.org/abs/2608.02809>（2026-11 通过 arXiv PDF v1 获取全文，pdftotext）
- **一句话说明：** 用已认证的外部安全链（光幕、急停、F-DI、F-PLC、无线 PROFIsafe）作"量尺"，把人形不可认证的部分精确定位到机侧反应链与 SDA-平衡策略接口。
- **核对范围：** 通读摘要、§I–§X 全部正文、Table I–III、Fig. 1–4 图注及参考文献列表；文中引用的 ISO/IEC 标准、Siemens 应用示例（Entry ID 21064024）与 arXiv:2603.22703 未单独阅读，仅依据本文转述。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题（§I）：** 传统机械安全以断电为安全态（Stop Category 0）；平衡双足断电即摔倒，安全态是需持续控制的"主动安全态"，违反 ISO 13849-1 / EN 60204-1 的 fail-passive 假设，称 fail-passive gap。
- **架构（§III）：** Detection（光幕、急停、F-DI）→ Evaluation（F-CPU、复位、PROFIsafe 监督）→ Reaction（SCALANCE W 无线 PROFIsafe、机载 SDA 软 PLC、平衡站停、通信丢失即静止）。无外部继电器 / 接触器。SDA 控制器以容器形式跑在 G1 机载非安全级 Linux 上，IEC 61131-3。
- **安全功能（§IV）：** 入侵或急停 → 稳定平衡站停；停后锁存，须 HMI 人工确认才可恢复（无自动重启）；通信丢失由 PROFIsafe 看门狗触发静止；机载 LiDAR 避障刻意排除在认证论证之外；平衡站停暂拟 Stop Category 1，未验证。
- **量化方法（§V）：** 类别、MTTFD、DC、CCF → PFHD → PL；参考例 Detection 9.06e-10、Evaluation 5.00e-9、Reaction（两接触器）1.45e-9，总 7.35e-9，PL e / SIL 3。人形链路无接触器，机侧反应链无 PFHD，不声称端到端 PL e / SIL 3。
- **分离距离（§V-F）：** S = K·T + C（ISO 13855，K=1600 mm/s 行走），T 被不可认证的 t_stop 主导。
- **时延预算（§VI，式 2、Table III）：** t_response = t_detect + t_FDI + t_scan + t_PROFIsafe + t_rx + t_stop；光幕 11 ms、F-DI 4–10 ms、PLC 周期 15–40 ms、PROFIsafe 30–39 ms（看门狗上界 ≤192 ms）、接收 5–20 ms、t_stop 0.3–1.0 s；最坏约 1.1 s，由 t_stop 主导。
- **结果（§VII）：** 四种方式诱发通信丢失（AP 断电、天线断开、有线上行断开、F-CPU 停机，各 3 次）均在 0.5–1.3 s 内稳定站停无摔倒；误触发率、可用性、丢包看门狗余量、停止距离曲线留待后续。
- **人形分析（§VIII）：** 过猛的停止可能引发摔倒；单支撑相需先落步（capture point）使 t_stop 有相位下界；平衡策略故障无 PFHD；学习式 safe-stoppability 监测可缩小但不能给出可认证论证。
- **局限（§IX）：** 机侧反应链未认证；机载算力非安全级，SDA 端点非认证 F-host；停止类别未定；参考数值为方法示例而非本单元重算；仅单区域、单场景。

## 对 wiki 的映射

- [`wiki/overview/hub-systems-engineering.md`](../../wiki/overview/hub-systems-engineering.md)
- [`wiki/entities/paper-fail-passive-gap.md`](../../wiki/entities/paper-fail-passive-gap.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2608.02809>
