# lzyang2000.github.io/perceptive_cbf_rl（PAC-MAN）

> 来源归档（ingest）

- **标题：** PAC-MAN: Perception-Aware CBF-RL for Whole-Body Safety in Humanoid Dodgeball
- **类型：** site / project-page
- **官方入口：** <https://lzyang2000.github.io/perceptive_cbf_rl/>
- **浏览器 Demo：** <https://lzyang2000.github.io/perceptive_cbf_rl/demo/>
- **论文：** <https://arxiv.org/abs/2607.28623>
- **代码：** <https://github.com/lzyang2000/perceptive_cbf_rl>
- **机构：** 加州理工学院（Caltech）AMBER Lab — Lizhi Yang, Junheng Li, Aaron D. Ames
- **入库日期：** 2026-08-01
- **一句话说明：** 感知感知 CBF-RL 人形躲避球项目页：训练期全身屏障 + AMP，部署机载掩膜深度；含浏览器交互 Demo 与 G1 真机 19/20 结果。
- **开源状态（2026-08-01 项目页核查）：** **已开源** — 页头/资源区列 GitHub；仓库含训练管线、benchmark 与硬件 `deploy/`。

## 页面公开信息

| 资源 | URL |
|------|-----|
| 项目首页 | <https://lzyang2000.github.io/perceptive_cbf_rl/> |
| 浏览器 Demo | <https://lzyang2000.github.io/perceptive_cbf_rl/demo/> |
| arXiv | <https://arxiv.org/abs/2607.28623> |
| PDF | <https://arxiv.org/pdf/2607.28623> |
| Code | <https://github.com/lzyang2000/perceptive_cbf_rl> |

## 方法摘要（项目页）

- **观测：** 头戴相机分割掩膜深度（球-only）+ 本体感觉；训练期特权几何供 CBF。
- **安全：** Link-CBF（每连杆 clearance 奖励，部署配置）与 Joint-CBF（关节空间投影指导 / 特权 +filter）。
- **风格：** AMP 对抗运动先验，涌现 duck / sidestep 等全身躲避模式。
- **真机：** Unitree G1 零样本；EfficientTAM 分割 + ZED 深度；**19/20** 躲开、**0** 跌倒。

## 对 wiki 的映射

- [`wiki/entities/paper-pac-man-perceptive-cbf-rl.md`](../../wiki/entities/paper-pac-man-perceptive-cbf-rl.md)
- [`sources/papers/pac_man_perceptive_cbf_rl_arxiv_2607_28623.md`](../papers/pac_man_perceptive_cbf_rl_arxiv_2607_28623.md)
- [`sources/repos/perceptive_cbf_rl.md`](../repos/perceptive_cbf_rl.md)
