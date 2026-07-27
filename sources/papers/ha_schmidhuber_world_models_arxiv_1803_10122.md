# World Models（Ha & Schmidhuber，arXiv:1803.10122）

> 来源归档（ingest）

- **标题：** World Models
- **类型：** paper / world models / model-based RL / latent dynamics / VAE / MDN-RNN
- **arXiv：** <https://arxiv.org/abs/1803.10122>（PDF：<https://arxiv.org/pdf/1803.10122.pdf>）
- **交互式论文 / 项目页：** <https://worldmodels.github.io/>
- **历史实验仓（作者相关社区实现）：** <https://github.com/hardmaru/WorldModelsExperiments>
- **作者：** David Ha、Jürgen Schmidhuber
- **机构：** 谷歌（Google Brain）、IDSIA / NNAISENSE
- **入库日期：** 2026-07-27
- **一句话说明：** 把智能体拆成 **Vision (VAE) + Memory (MDN-RNN) + Controller (小线性策略)**：先无监督学压缩时空世界模型，再在「梦境」里用进化策略训紧凑控制器，并可把策略迁回真实环境；交互式论文展示 VizDoom / CarRacing 等实验。

## 开源状态（项目页核查，2026-07-27）

- **已开源（交互式论文 + 历史社区复现）：** 官方入口是 [worldmodels.github.io](https://worldmodels.github.io/)（可交互阅读与演示，非单一训练仓）；作者相关实验代码见 [hardmaru/WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments)。后续社区另有大量第三方复现；以交互站点与作者实验仓为导航锚点，勿假定存在与 2018 原文一一对应的「唯一官方 PyTorch 训练包」。

## 摘要级要点

- **动机：** 大网络难靠稀疏回报端到端训；把容量放进世界模型，把信用分配压到小控制器。
- **结构：** V 用 VAE 把像素压成 \(z_t\)；M 用 MDN-RNN 建模 \(P(z_{t+1}\mid a_t,z_t,h_t)\)；C 仅用 \([z_t;h_t]\) 线性映射动作。
- **梦中训练：** 可在 M 生成的 latent 环境中训 C，再迁回真实 env；用温度 \(\tau\) 调大不确定性，抑制利用模型漏洞。
- **与后续谱系：** 奠定「压缩表示 + 潜动态 + 在模型里练策略」叙事；PlaNet / Dreamer / TD-MPC2 等沿 latent planning / imagination 方向演化。

## 核心论文摘录（MVP）

### 1) V–M–C 分工

- **链接：** Agent Model 节；交互站流程图
- **摘录要点：** 观测 → VAE → \(z\)；RNN 隐状态 \(h\) 承载时序；控制器故意保持极小，使进化/RL 搜索空间小而表达力由世界模型承担。
- **对 wiki 的映射：**
  - [World Models 实体页](../../wiki/entities/paper-ha-schmidhuber-world-models.md)
  - [Latent Imagination](../../wiki/concepts/latent-imagination.md)
  - [Model-Based RL](../../wiki/methods/model-based-rl.md)

### 2) MDN-RNN 与温度 \(\tau\)

- **链接：** MDN-RNN (M) Model；CarRacing / VizDoom 实验讨论
- **摘录要点：** 下一 latent 用高斯混合输出；采样温度控制梦境随机性；过高/过低 \(\tau\) 分别导致策略保守或钻洞。
- **对 wiki 的映射：**
  - [world-model-physics-fidelity-outputs](../../wiki/overview/world-model-physics-fidelity-outputs.md) — 低维潜状态输出族代表

### 3) 在梦中训练并迁移

- **链接：** Learning Inside of a Dream；Experiments
- **摘录要点：** 可完全在生成环境训 C；迁移到真实环境时性能依赖模型保真与 \(\tau\) 选择。
- **对 wiki 的映射：**
  - [Generative World Models](../../wiki/methods/generative-world-models.md)
  - [PlaNet](../../wiki/entities/paper-planet-latent-dynamics.md) / [DreamerV3](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md)

## BibTeX

```bibtex
@article{ha2018worldmodels,
  title   = {World Models},
  author  = {Ha, David and Schmidhuber, J{\"u}rgen},
  journal = {arXiv preprint arXiv:1803.10122},
  year    = {2018}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-ha-schmidhuber-world-models.md`](../../wiki/entities/paper-ha-schmidhuber-world-models.md)
- 项目页：[`sources/sites/worldmodels-github-io.md`](../sites/worldmodels-github-io.md)
- 实验仓：[`sources/repos/world-models-experiments.md`](../repos/world-models-experiments.md)
- 博客策展：[`sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md`](../blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)
