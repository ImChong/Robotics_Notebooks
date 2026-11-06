# EmbersArc/SCvx

> 来源归档（ingest）

- **标题：** SCvx — Implementation of the Successive Convexification algorithm
- **类型：** repo
- **链接：** <https://github.com/EmbersArc/SCvx>
- **语言 / License：** Python / MIT
- **入库日期：** 2026-11-06
- **核对范围：** 读了 GitHub 仓库首页 README 与文件列表；未运行代码。
- **通读：** 已完成
- **一句话说明：** 用 CVXPY 实现 Mao 等 SCvx 算法（arXiv:1804.06539），附 [Szmuk & Açıkmeşe 6-DoF 自由终端时间火箭着陆](../papers/szmuk_2018_scvx_6dof_free_final_time.md) 模型、2D 火箭着陆与差速小车示例。

## 开源状态（2026-11-06 核查）

| 产物 | 状态 |
|------|------|
| 固定 / 自由终端时间 SCvx（`fixedfinaltime.py`、`freefinaltime.py`） | **已开源** |
| 模型（`Models/`：6-DoF 火箭、2D 火箭、差速小车） | **已开源** |
| 依赖 | README："Reqires `matplotlib`, `numpy`, `scipy` and `cvxpy`"（原文拼写），底层锥规划求解器未写明 |

## 对 wiki 的映射

- [序列凸化（Successive Convexification, SCvx）](../../wiki/methods/successive-convexification.md)
- [火箭回收控制纵深路线](../../roadmap/depth-rocket-landing.md) — Stage 3
