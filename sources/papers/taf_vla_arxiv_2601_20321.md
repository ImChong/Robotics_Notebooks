# TaF-VLA: Tactile-Force Alignment in Vision-Language-Action Models for Force-aware Manipulation

> 来源归档（ingest）

- **标题：** TaF-VLA: Tactile-Force Alignment in Vision-Language-Action Models for Force-aware Manipulation
- **类型：** paper / vla / vbts / force-alignment / taf-adapter / taf-dataset
- **arXiv abs：** <https://arxiv.org/abs/2601.20321>
- **arXiv HTML：** <https://arxiv.org/html/2601.20321>
- **PDF：** <https://arxiv.org/pdf/2601.20321>
- **项目页：** <https://peilin-666.github.io/projects/TaF_VLA/>
- **机构：** Beihang University；ShanghaiTech University；BIGAI；The University of Hong Kong
- **通讯作者：** Chenxi Xiao、Ziyuan Jiao（†）
- **入库日期：** 2026-09-23
- **一句话说明：** 提出 **触觉–力对齐**（相对触觉–视觉对齐）：**TaF-Device** 自动采集 **1000 万+** 同步视触觉 / 6 轴 F/T / 压力矩阵帧 → **TaF-Adapter**（对比学习 + VQ 码本 + 历史聚合）→ 接入 VLA；7 项力敏感日常任务平均较 SOTA 视触觉 VLA **+22%** 成功率。

## 开源核查（2026-09-23）

| 项 | 状态 |
|----|------|
| 项目页 | <https://peilin-666.github.io/projects/TaF_VLA/> — arXiv / **Code (Coming soon)** 按钮 |
| GitHub | [`github.com/mrHuangyz/TaF-VLA`](https://github.com/mrHuangyz/TaF-VLA) 存在（README 描述完整）— 项目页仍标 Coming soon |
| TaF-Dataset | HF [`jiamig/taf-dataset`](https://huggingface.co/datasets/jiamig/taf-dataset) — README 描述 **10,053,265** 帧 / 6 传感器；**文件区可能未完全上传**（以 HF 实际文件为准） |
| TaF-Adapter 权重 | **未列** 官方 checkpoint 下载 |
| 结论 | **部分开源 / 待发布**（仓库与数据集元数据可见；项目页声明代码即将发布） |

## 摘要级要点

- **问题：** VLA **力盲**；腕部 F/T 贵且低维；现有 VTLA 把触觉当「更多视觉纹理」，未 grounding 到 **物理力动态**。
- **范式：** **Tactile-Force Alignment** — 触觉 embedding 与力 profile 在 **共享 latent** 对齐（非显式 force regression）。
- **TaF-Device：** 平行施力结构；可换 indenter / VBTS；**6** 种传感器；**10 万帧/小时** 吞吐。
- **TaF-Adapter：** 时序触觉 → VQ 离散码本；对比学习对齐 6 轴 F/T + 压力矩阵；历史依赖、抗噪。
- **TaF-VLA：** 力对齐 token **interleave** 进 language-action 流；力感知语言指令微调。
- **任务：** 果冻切片、镊子取砝码、插件等；相对视触觉对齐 VLA 与 vision-only 基线 **+22%** 平均。
- **与 Awesome Touch 索引：** [`sources/papers/sun_awesome_touch_2601_20321_taf-vla-tactile-force-alignment-in-visio.md`](./sun_awesome_touch_2601_20321_taf-vla-tactile-force-alignment-in-visio.md) 为清单级摘录。

## 核心论文摘录（MVP）

### 1) TaF-Device 与 TaF-Dataset 规模化采集

- **链接：** <https://arxiv.org/html/2601.20321#S3>；项目页 TaF-Device
- **摘录要点：** 并行 actuator 对触觉传感器与 F/T 同步施力；**>10M** 帧；六传感器覆盖 GelSight / DIGIT 等 VBTS 族。
- **对 wiki 的映射：**
  - [TaF-VLA（Awesome Touch 实体）](../../wiki/entities/paper-sa-2601-20321-taf-vla-tactile-force-alignment-in-vision-langua.md)
  - [触觉传感](../../wiki/concepts/tactile-sensing.md)

### 2) TaF-Adapter：隐式 latent 对齐 vs 显式力回归

- **链接：** <https://arxiv.org/html/2601.20321#S4>
- **摘录要点：** 对比学习 + vector-quantized shared space；历史帧聚合捕获 stick-slip / 粘弹性；跨传感器泛化优于 explicit calibration。
- **对 wiki 的映射：**
  - [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)
  - [ForceVLA 归档](./forcevla_arxiv_2505_22159.md) — 低维 F/T token vs 高维 VBTS–力对齐

### 3) 力感知操作 benchmark

- **链接：** arXiv §V；项目页 demo 视频
- **摘录要点：** 7 力临界任务；可插拔 TaF-Adapter 到 Diffusion Policy / ACT；果冻切气球等 fragile 场景。
- **对 wiki 的映射：**
  - [接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)
  - [VLA](../../wiki/methods/vla.md)

## 对 wiki 的映射（汇总）

- 实体页：[TaF-VLA（arXiv:2601.20321）](../../wiki/entities/paper-sa-2601-20321-taf-vla-tactile-force-alignment-in-vision-langua.md)
- 概念/方法：[VLA](../../wiki/methods/vla.md)、[视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)、[接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)
- 项目页归档：[`sources/sites/taf-vla-peilin.md`](../sites/taf-vla-peilin.md)

## 当前提炼状态

- [x] 三阶段 pipeline、TaF-Adapter 设计、benchmark 结果、部分开源边界已摘录
- [x] 与 [`sources/sites/taf-vla-peilin.md`](../sites/taf-vla-peilin.md) 互证

## BibTeX

```bibtex
@article{huang2026tafvla,
  title={TaF-VLA: Tactile-Force Alignment in Vision-Language-Action Models for Force-aware Manipulation},
  author={Huang, Yuzhe and Lin, Pei and Li, Wanlin and Li, Daohan and Li, Jiajun and Jiang, Jiaming and Xiao, Chenxi and Jiao, Ziyuan},
  journal={arXiv preprint arXiv:2601.20321},
  year={2026},
  url={https://arxiv.org/abs/2601.20321},
}
```
