# dstate.github.io/odeworld_website（ODEWorld 项目页）

- **标题：** ODEWorld — A Continuous Predictive Architecture via Physical-Time Flow
- **类型：** site / project-page
- **URL：** <https://dstate.github.io/odeworld_website/>
- **配套论文：** [ODEWorld（arXiv:2607.27924）](https://arxiv.org/abs/2607.27924) — 归档见 [`sources/papers/odeworld_arxiv_2607_27924.md`](../papers/odeworld_arxiv_2607_27924.md)
- **代码：** <https://github.com/Dstate/ODEWorld> — 归档见 [`sources/repos/odeworld.md`](../repos/odeworld.md)
- **权重：** <https://huggingface.co/collections/ldxxx/odeworld>
- **入库日期：** 2026-08-16

## 一句话摘要

清华 AIR × Berkeley BAIR 的 **ODEWorld** 官方站点：用交互时间轴展示 **物理时间连续预测**（任意帧密度、反向预测、缺帧插值），并给出 AgiBot / LIBERO / AgileX 上的 ODE rollout → PCA latent 轨迹 → 解码视频 demo。

## 公开信息要点（截至入库日）

- **机构：** Institute for AI Industry Research (AIR), Tsinghua；Berkeley Artificial Intelligence Research (BAIR), UC Berkeley。通讯：Haoyi Niu（`niu@berkeley.edu`）、Xianyuan Zhan。
- **页首卖点：** *Physical-Time World Modeling* — PT-Flow 把动力学写成压缩 latent 上的连续速度场，而不是离散 next-step。
- **交互 demo：** AgiBot / LIBERO / AgileX 三组 case；时间轴可拖、倍速 0.1×–4×；展示 initial / goal、任务指令、PCA latent 轨迹与解码预测。
- **能力板块：** Bidirectional generation；Any-resolution temporal generation（训练采样率之外补中间帧）。
- **Critical Designs：** 公式 (1)–(4) 与论文一致——动力学解耦 + JVP 一阶监督。
- **定量表（站点口径，与论文互补）：**
  - 视频：ODEWorld 短程 PSNR **20.53** / LPIPS **0.109** / FPS **33.67**；长程 **19.46** / **0.134** / **13.83**；参数 **86.08 M**（对照 LDP 110.9 M、V-JEPA 2 452.34 M）。
  - 策略：LIBERO-LONG 序列子目标平均成功率 **83.6%**（Velocity 82.3 / Single 82.6）。
- **步骤 2.5：** 页上可定位到论文与代码入口；推理权重在 HF collection。站点本身是展示页，不含训练脚本。

## 为何值得保留

- **非 PDF 证据：** 任意时间查询、反向预测与 PCA 速度场可视化，比表格更能说明「连续时间」不是口号。
- **与 arXiv / GitHub 三角互证：** demo 管线（ODERollout → Decode）对齐仓内 `demo_infer.py` 的 `rollout_ode` + RAE decode。
- **开源边界以页上实际链接为准：** 有 GitHub + HF，不是「宣称将开源」。

## 关联资料

- 论文归档：[`sources/papers/odeworld_arxiv_2607_27924.md`](../papers/odeworld_arxiv_2607_27924.md)
- 代码仓库：[`sources/repos/odeworld.md`](../repos/odeworld.md)
- 升格：[`wiki/entities/paper-odeworld.md`](../../wiki/entities/paper-odeworld.md)
