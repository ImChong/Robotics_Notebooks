# Wan-Dancer: A Hierarchical Framework for Minute-scale Coherent Music-to-Dance Generation（arXiv:2607.09581）

> 来源归档（ingest）

- **标题：** Wan-Dancer: A Hierarchical Framework for Minute-scale Coherent Music-to-Dance Generation
- **类型：** paper / music-to-dance / hierarchical video generation / I2V / Wan
- **arXiv：** <https://arxiv.org/abs/2607.09581>（PDF：<https://arxiv.org/pdf/2607.09581.pdf>）
- **项目页：** <https://humanaigc.github.io/wan-dancer-project/>
- **代码：** <https://github.com/Wan-Video/Wan-Dancer>
- **权重：** HF <https://huggingface.co/Wan-AI/Wan-Dancer-14B>；ModelScope <https://www.modelscope.cn/models/Wan-AI/Wan-Dancer-14B>
- **作者：** Mingyang Huang、Peng Zhang、Li Hu、Guangyuan Wang、Ruoshi Zhang、Yi Lu、Gang Cheng、Bang Zhang（Tongyi Lab, Alibaba Group）
- **机构：** 阿里巴巴（Alibaba）通义实验室（Tongyi Lab）
- **入库日期：** 2026-07-31
- **一句话说明：** 在 **Wan-I2V** 上做分层 music-to-dance：全局关键帧规划 + 局部时序 refinement，配合 **time-mapped RoPE**、光流加权损失与运动速度分层，生成 **720p/30fps、分钟级** 五类舞种视频；权重与推理代码已开源（Apache-2.0）。

## 开源状态（项目页 + 仓库核查，2026-07-31）

- **已开源：** 项目页挂 [`Wan-Video/Wan-Dancer`](https://github.com/Wan-Video/Wan-Dancer) + HF/ModelScope **Wan-Dancer-14B** + ModelScope Studio。
- **误写提醒：** `github.com/Wan-AI/Wan-Dancer-14B` **404**；`Wan-AI/Wan-Dancer-14B` 是 **Hugging Face / ModelScope 模型 ID**，不是 GitHub 仓。

## 摘要级要点

- **瓶颈：** 通用视频扩散多卡在 **~5–20 s**；music-to-motion（3D 骨架）与端到端 music-to-video 在长时程上易漂移、身份抖动、动作重复。
- **核心：** 同一 DiT 框架用 **keyframe mask** 统一训练 Global（仅首帧锚定）与 Local（稀疏随机关键帧）；推理时先出稀疏全局视频，再按关键帧切片做局部高帧率细化并拼接。
- **技巧：** absolute-time **RoPE** 适配可变时长；VAE 光流 latent 加权 RF 损失；按关键点速度分层（慢/中/快）采样。
- **数据：** 自建约 **200 h**、≥720p@30fps、五类舞种近均匀；5 s 片段 50% overlap；不用 AIST/Finedance 作主合成数据（分辨率/时长不足）。
- **与机器人关系：** 本身是 **像素级舞蹈视频生成**，无关节/力矩接口；对人形 [WBT](../../wiki/overview/hub-wbt.md) / 高动态模仿，价值在「长时程、有节奏的参考视频先验」与 Wan 族可控生成谱系，而非可部署策略。

## 核心论文摘录（MVP）

### 1) Hierarchical Global-to-Local

- **链接：** §3.1；§4.1.3；Fig. 1–2
- **摘录要点：** Global mask 仅第一帧为 1，强迫从初始条件推演全曲结构；Local 随机关键帧 mask 学插值；推理：Global ~38 关键帧 → 按关键帧切 149 帧片段 → Local DiT 并行细化 → 拼接分钟级视频。
- **对 wiki 的映射：**
  - [Wan-Dancer](../../wiki/entities/paper-wan-dancer.md) — 核心管线。
  - [Wan](../../wiki/entities/paper-wan-video.md) — Wan-I2V 骨干。

### 2) Time-mapped RoPE + optical-flow loss + speed control

- **链接：** §3.1–3.2；Eq. (1)–(3)
- **摘录要点：** RoPE 注入绝对时间以适配动态 fps；RF 目标上对速度场乘以光流权重 \(w_{\text{optical\_flow}}\)；数据按运动速度分层，强调中速主导、兼顾快动作细节。
- **对 wiki 的映射：**
  - [Wan-Dancer](../../wiki/entities/paper-wan-dancer.md) — 长时程稳定技巧。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 长 horizon 视频生成对照。

### 3) 评测与 LoRA 定制

- **链接：** §4.2–4.3；Tables 1–3
- **摘录要点：** 对照 X-Dancer / MusicInfuser；Dance / Video / Prompt 三类用户向指标报告均值 **8.46 / 7.46 / 9.03**（相对基线全面更高）；LoRA rank 32、约 16 条同编舞参考、800 step 可定制特定套路。
- **对 wiki 的映射：**
  - [Wan-Dancer](../../wiki/entities/paper-wan-dancer.md) — 实验与定制路径。

## BibTeX

```bibtex
@article{wan-dancer-2026,
  title         = {Wan-Dancer: A Hierarchical Framework for Minute-scale Coherent Music-to-Dance Generation},
  author        = {Huang, Mingyang and Zhang, Peng and Hu, Li and Wang, Guangyuan and Zhang, Ruoshi and Lu, Yi and Cheng, Gang and Zhang, Bang},
  year          = {2026},
  eprint        = {2607.09581},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CV},
  url           = {https://arxiv.org/abs/2607.09581},
  note          = {Project page: \url{https://humanaigc.github.io/wan-dancer-project/}}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-wan-dancer.md`](../../wiki/entities/paper-wan-dancer.md)
- 代码归档：[`sources/repos/wan-dancer.md`](../repos/wan-dancer.md)
- 项目页：[`sources/sites/wan-dancer-project.md`](../sites/wan-dancer-project.md)
- 互链：[Wan](../../wiki/entities/paper-wan-video.md)、[Wan-Move](../../wiki/entities/paper-wan-move.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)、[WBT 枢纽](../../wiki/overview/hub-wbt.md)
