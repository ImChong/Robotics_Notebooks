# A Comparative Study of Floating-Base Space Parameterizations for Agile Whole-Body Motion Planning（arXiv:2508.11520）

> 来源归档（ingest）

- **标题：** A Comparative Study of Floating-Base Space Parameterizations for Agile Whole-Body Motion Planning
- **缩写 / 框架：** **SE(3) Tangent TO** / SE3_TrajOpt
- **类型：** paper / trajectory-optimization / floating-base / se3 / comparative-study
- **arXiv：** <https://arxiv.org/abs/2508.11520>（v1，2025-08）
- **会议：** 2025 IEEE-RAS 24th International Conference on Humanoid Robots，pp. 1187–1194
- **DOI：** <https://doi.org/10.1109/Humanoids65713.2025.11203204>
- **代码（论文声明）：** <https://github.com/upatras-lar/se3_trajopt>（BSD-2-Clause）— [`sources/repos/se3_trajopt.md`](../repos/se3_trajopt.md)
- **社区扩展（用户指定 ingest）：** <https://github.com/yusongmin1/go2_flip_TO> — [`sources/repos/go2_flip_to.md`](../repos/go2_flip_to.md)
- **项目页：** <https://lar.upatras.gr/projects/ibrics.html>
- **视频：** <https://www.youtube.com/watch?v=zBJSsiUExCw>
- **作者：** Evangelos Tsiatsianas、Chairi Kiourt、Konstantinos Chatzilygeroudis
- **机构：** 帕特雷大学 LAR；Archimedes / Athena RC
- **入库日期：** 2026-08-18
- **一句话说明：** 在同一套直接配点全身 TO + Ipopt 下对比五种浮动基参数化；SE(3) 切空间在大转角空翻上唯一稳定翻出动作，且仍用现成欧式 NLP、不必上 ProxDDP 一类流形求解器。

## 开源状态（步骤 2.5）

- **论文声明：** `https://github.com/upatras-lar/se3_trajopt`（截至 2026-08-18：61★，last push 2025-12-20，`master`）。
- **用户指定仓：** `yusongmin1/go2_flip_TO`（GitHub **未标 fork**；README 与论文/项目页一致，并多出 Go2 AMP 50 Hz 导出、MUMPS 线性求解器说明）。维护者在 LAR 二人之外增加 yusongmin。last push 2026-08-15。
- **结论：** **已开源**。复现论文六任务用官方仓即可；要 Go2 空翻轨迹当 RL/AMP 参考，跟扩展仓 `src/examples/agile_exps/quad_*.py`。**论文未做真机。**

## 摘录 1：五种表示（Table I）

| 名称 | 优化变量 | 差分 | 积分 |
|------|----------|------|------|
| SE(3) Tangent | \(\xi_k\) | \(\mathrm{Exp}(\xi_2)\ominus\mathrm{Exp}(\xi_1)\) | \(\mathrm{Log}(\mathrm{Exp}(\xi)\oplus\mathcal{V}_b h)\) |
| Quaternion #1 | \(p_k,\rho_k\) | 欧式相减 | 四元数欧拉积分 |
| Quaternion #2 | 同上 | 欧式相减 | 经 \(\mathrm{Exp}_q/\mathrm{Log}_q\) 的流形积分 |
| Quaternion #3 | 同上 | 流形差分 | 流形积分 |
| RPY | \(p_k,\theta_k\) | 欧式相减 | \(W(\theta)\omega_b\) 映射 |

实现：半隐式欧拉；SE(3) 切空间用 Pinocchio 解析雅可比 + 链式法则；四元数/欧拉用 CasADi–Pinocchio 自动微分；求解器 Ipopt。接触日程**预先给定**（与 AHMP 外层发现相反）。

**对 wiki 的映射：** [`wiki/entities/paper-se3-tangent-to.md`](../../wiki/entities/paper-se3-tangent-to.md) 把「变量 / 差分 / 积分」三决策写成选型表。

## 摘录 2：任务与公平协议（§V-A）

六任务、非信息性中立站姿暖启动、跨机几乎同一代价（仅倒立/后空翻加极小正则）。G1 后空翻在中间若干节点把基座朝向设为倒置，避免优化器只做后跳。

| 任务 | 平台 | 时长 / 步长 |
|------|------|-------------|
| Walk 2 m | Talos | 3.3 s / 0.05 s |
| Hopscotch 2 m | Talos | 4.3 s / 0.05 s |
| Big jump 1 m | Talos | 2.3 s / 0.05 s |
| Handstand | Talos | 3.4 s / 0.05 s |
| Back-flip 0.5 m 后 | Unitree G1 | 2.4 s / 0.05 s |
| Side-flip 0.3 m 侧 | Unitree Go2 | 2.4 s / 0.02 s |

另对 Walk / Jump / Back-flip / Side-flip 加暖启动高斯噪声 \(\sigma\in\{10^{-6},10^{-3},0.1,0.5\}\)，每格 10 次，共 640 次。

**对 wiki 的映射：** 强调「同一转录、同一求解器、弱任务代价」才使表示法对比可读。

## 摘录 3：结果读法（Table III–IV）

- 小转角任务（走、跳房子、大跳、倒立）：除 Quaternion #3 全失败外，其余收敛次数接近。
- **G1 后空翻 / Go2 侧空翻：只有 SE(3) 切空间翻成功。** 欧拉不收敛；Quaternion #1/#2 常收敛到可行**跳**而非翻（表中 `*`）。
- 噪声鲁棒：空翻任务上 SE(3) 切空间在 \(\sigma\le 0.1\) 保持 100% 可行；\(\sigma=0.5\) 时 Go2 侧空翻降至 60%。四元数的「成功」多数不是翻。
- 选型：不需要大转角时 Quaternion #1 仍可用；要空翻/大姿态变化优先切空间。

**对 wiki 的映射：** 结论写清「收敛 ≠ 完成技能」；`go2_flip_TO` 把切空间解导出成 50 Hz AMP 行，供下游跟踪 RL。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-se3-tangent-to.md`**
- 交叉 AHMP、轨迹优化、SE(3)/李群、Crocoddyl（DDP vs Ipopt NLP）
