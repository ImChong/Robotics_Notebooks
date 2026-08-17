# MPPI Planning with Gaussian-Based Human Cost Function for Social Navigation（arXiv:2608.08323）

> 来源归档（ingest）

- **标题：** MPPI Planning with Gaussian-Based Human Cost Function for Social Navigation
- **缩写 / 框架：** **PGIF** / **PGIF-MPPI**
- **类型：** paper / social-navigation / mppi / mpc
- **arXiv：** <https://arxiv.org/abs/2608.08323>
- **会议：** HFR 2026（Springer Proceedings in Advanced Robotics）
- **代码：** <https://github.com/ChinmayMundane/PGIF_MPPI>（归档见 [`sources/repos/pgif-mppi.md`](../repos/pgif-mppi.md)）
- **作者：** Chinmay Mundane
- **机构：** VJTI（印度）；本库机构表无对应 alias，暂不注册
- **入库日期：** 2026-08-17
- **一句话说明：** 把行人运动学预测沿整个 MPPI horizon 铺成沿速度方向拉长的各向异性高斯排斥场（motion cone），闭式可并行，几乎零额外算力。

## 开源状态（步骤 2.5）

- **项目页：** 无独立站点。
- **代码仓核查（2026-08-17）：** [ChinmayMundane/PGIF_MPPI](https://github.com/ChinmayMundane/PGIF_MPPI)（MIT）。入口 `mppi_dynamic_humans.py`（可视化）、`evaluate_mppi.py`（100 seed 批评）、`plot_paper_figures.py`。依赖 `numpy matplotlib jax jaxlib`。
- **结论：** **已开源、可运行仿真评测**（走廊人群；非真机 ROS 栈）。

## 摘录 1：代价

unicycle 模型；\(K=512\)、\(T=40\)、\(\Delta t=0.1\) s、\(\lambda=1000\)。代价：goal + terminal + path + human（\(w_{\mathrm{human}}=10^6\)）。行人用运动学前向，高斯前向展宽随速度增大。软代价不是硬约束，无形式化防撞保证。

## 摘录 2：数字（100 trial × 3 密度）

| | Easy | Medium | Hard |
|--|-----:|-------:|-----:|
| Vanilla 成功 / 碰撞 | 78 / 22 | 29 / 71 | 18 / 82 |
| PGIF 成功 / 碰撞 / 超时 | 93 / 0 / 7 | 78 / 0 / 22 | 41 / 0 / 59 |
| PGIF 路径 (m) / 步时 (ms) | 14.43 / 3.06 | 16.87 / 3.23 | 18.75 / 3.42 |

Hard 上安全换超时：预测场占满走廊，机器人选择等待。

**对 wiki 的映射：** [`wiki/entities/paper-pgif-mppi.md`](../../wiki/entities/paper-pgif-mppi.md)；交叉 [MPPI](../../wiki/methods/mppi.md)、[MPC](../../wiki/methods/model-predictive-control.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（JAX 仿真仓）
