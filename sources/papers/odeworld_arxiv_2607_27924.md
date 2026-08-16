# ODEWorld: A Continuous Predictive Architecture via Physical-Time Flow（arXiv:2607.27924）

> 来源归档（ingest）

- **标题：** ODEWorld: A Continuous Predictive Architecture via Physical-Time Flow
- **类型：** paper / continuous-time latent world model / video generation / policy guidance
- **arXiv：** <https://arxiv.org/abs/2607.27924>（PDF：<https://arxiv.org/pdf/2607.27924.pdf>；HTML：<https://arxiv.org/html/2607.27924>）
- **项目页：** <https://dstate.github.io/odeworld_website/>
- **代码：** <https://github.com/Dstate/ODEWorld>
- **权重：** HF collection [`ldxxx/odeworld`](https://huggingface.co/collections/ldxxx/odeworld)
- **作者：** Dongxiu Liu*、Haoyi Niu*✉、Peng Cheng、Yuan Gao、Xirui Kang、Sangli Teng、Koushil Sreenath、Xianyuan Zhan✉（* 同等贡献，抛硬币定序）
- **机构：** 清华大学智能产业研究院（AIR, Tsinghua）；加州大学伯克利分校 BAIR（UC Berkeley）
- **入库日期：** 2026-08-16
- **一句话说明：** 用 **Physical-Time Flow（PT-Flow）** 在压缩 latent 上学 **物理时间** 上的连续速度场，把未来预测改写成 ODE 积分；实例化 **ODEWorld** 同时做长程视频生成与 latent 子目标策略引导。相对离散 next-step / JEPA，强调任意时间分辨率、反向预测，以及用 JVP 一阶监督缓解表征坍塌。

## 开源状态（项目页 + 仓库核查，2026-08-16）

- **已开源、可运行推理：** 项目页链到 [`Dstate/ODEWorld`](https://github.com/Dstate/ODEWorld) + HF [`ldxxx/odeworld`](https://huggingface.co/collections/ldxxx/odeworld)。README 入口是 `demo_infer.py`（`torchdiffeq.odeint` RK4）+ 五套预训练权重（PT-Flow / RAE / Goal-Predictor × LIBERO；PT-Flow / RAE × AgiBot）。仓内有 `models/`（`DINOv2PTFlow.py`、`DINOv2RAE.py`、`DINOv2GoalPred.py`、`DINOv2Latent.py`）与 `dataloader/`、`assets/` 示例。
- **边界：** GitHub **未挂 LICENSE**；README **未列训练脚本**（无 `train.py`）。策略学习（LIBERO-LONG / AgileX + X-VLA）与完整训练管线 **未随仓发布**。
- **数据：** 训练用公开 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 全量与 **AgiBotWorldChallenge-2025** 子集（约 30K 轨迹）；仓内 demo 只带 PNG case，不随仓发全量数据。

## 摘要级要点

- **问题：** 现有世界模型几乎全是离散时间 next-step / 视频序列预测，对不规则采样、任意时间查询与高阶连续动力学不友好；JEPA 族还容易表征坍塌。
- **PT-Flow 两件套：** (1) **动力学表征解耦**——\(f_{\mathrm{dyn}}(s_t;s_0)\) / \(g_{\mathrm{dyn}}(z_t;s_0)\) 把静态内容交给 \(s_0\) 条件；(2) **直接一阶监督**——用 JVP 把 \(\dot s_t\) 投影成 \(\dot z_t\)，监督速度场 \(v_\theta\)（stop-gradient 可选）。
- **ODEWorld：** 冻结 DINOv2 → DINO 特征空间上跑 PT-Flow；单 token \(z_t\in\mathbb{R}^{1\times 768}\)；3 层 MLP + FiLM 时间调制；RAE 解码回像素。推理用现成 ODE solver（RK4）。
- **能力：** 任意时间分辨率 / 缺帧插值、反向积分、长程开环仍可重建；latent 速度场可作 velocity / 单子目标 / 序列子目标策略条件。
- **数字（论文口径）：** LIBERO 视频短程 PSNR **20.53** / LPIPS **0.109** / 延迟 **0.030 s**（@16 帧），长程 **19.46** / **0.134** / **0.072 s**（@64 帧，单 A100）；LIBERO-LONG 序列子目标平均成功率 **83.6%**；AgileX 真机四任务 X-VLA **55%→80%**。

## 核心论文摘录（MVP）

### 1) PT-Flow：物理时间上的 latent ODE

- **链接：** §3；Eq. (1)–(4)
- **摘录要点：** \(\frac{dz_t}{dt}=v_\theta(z_t,t;z_0,c)\)，未来预测 = 在压缩 latent 上对物理时间积分。与 flow matching 都学速度场 + ODE solver，但时间轴是 **物理时间** 而非噪声时间。动力学解耦损失 \(\mathcal{L}_{\mathrm{dyn-recon}}\) + 速度损失 \(\mathcal{L}_v=\|v_\theta-\mathrm{sg}(\mathrm{JVP}(f_{\mathrm{dyn}},\dot s_t))\|^2\)。
- **对 wiki 的映射：**
  - [ODEWorld](../../wiki/entities/paper-odeworld.md) — 核心机制。
  - [Latent Imagination](../../wiki/concepts/latent-imagination.md) — 连续时间潜空间展开对照。

### 2) ODEWorld 架构与紧凑动力学 token

- **链接：** §4；Fig. 1(c)
- **摘录要点：** 冻结 DINOv2 \(f_{\mathrm{obs}}\) + RAE \(g_{\mathrm{obs}}\)；\(f_{\mathrm{dyn}}/g_{\mathrm{dyn}}\) 为 \(s_0\) 条件 cross-attention；单 token 已够用（4 token 增益有限）。时间重标 \(\tau=t/L\in[0,1]\)；Savitzky–Golay 窗 5 估 \(\dot s_t\)。损失 \(\lambda_{\mathrm{rec}}=\lambda_v=1\) 无需调。
- **对 wiki 的映射：**
  - [ODEWorld](../../wiki/entities/paper-odeworld.md) — 工程栈。
  - [V-JEPA 2](../../wiki/entities/paper-vjepa2.md) — 同属 latent WM，监督方式对照。

### 3) 视频生成 + 策略子目标 + 真机

- **链接：** §5.3–5.4；Tab. 1–3；App. E
- **摘录要点：** 相对 LDP / V-JEPA 2，短长程 PSNR/LPIPS 与延迟全面更好；序列子目标 \(\{\hat z_{\tau_i}\}_{i=1}^{5},\tau_i=0.05i\) 在 LIBERO-LONG 最高。真机用 X-VLA + 序列子目标。局限：数据规模小、冻结 DINO 时序噪声、**当前版本无动作条件**。
- **对 wiki 的映射：**
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 连续时间 latent 路线。
  - [LIBERO](../../wiki/entities/libero-benchmark.md) — 评测基准。
  - [world-models-route-01-cascade](../../wiki/overview/world-models-route-01-cascade.md) — 子目标再接策略的级联形态。

## BibTeX

```bibtex
@article{liu-niu2026odeworld,
  title={ODEWorld: A Continuous Predictive Architecture via Physical-Time Flow},
  author={Liu, Dongxiu and Niu, Haoyi and Cheng, Peng and Gao, Yuan and Kang, Xirui and Teng, Sangli and Sreenath, Koushil and Zhan, Xianyuan},
  journal={arXiv preprint arXiv:2607.27924},
  year={2026}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-odeworld.md`](../../wiki/entities/paper-odeworld.md)
- 代码归档：[`sources/repos/odeworld.md`](../repos/odeworld.md)
- 项目页：[`sources/sites/odeworld-website.md`](../sites/odeworld-website.md)
- 互链：[Generative World Models](../../wiki/methods/generative-world-models.md)、[Latent Imagination](../../wiki/concepts/latent-imagination.md)、[V-JEPA 2](../../wiki/entities/paper-vjepa2.md)、[LIBERO](../../wiki/entities/libero-benchmark.md)、[物理保真输出轴](../../wiki/overview/world-model-physics-fidelity-outputs.md)
