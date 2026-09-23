# OmniVTLA: Vision-Tactile-Language-Action Model with Semantic-Aligned Tactile Sensing

> 来源归档（ingest）

- **标题：** OmniVTLA: Vision-Tactile-Language-Action Model with Semantic-Aligned Tactile Sensing
- **类型：** paper / vla / vtla / tactile-pretraining / objtac-dataset / semantic-alignment
- **arXiv abs：** <https://arxiv.org/abs/2508.08706>
- **arXiv HTML：** <https://arxiv.org/html/2508.08706>
- **PDF：** <https://arxiv.org/pdf/2508.08706>
- **项目页 / ObjTac：** <https://readerek.github.io/Objtac.github.io/>
- **数据集门户：** <https://omnisharingdb.paxini.com/>（arXiv 元数据 `\dataset`）
- **机构：** Shanghai Jiao Tong University；Paxini Tech
- **通讯作者：** Zhengxue Cheng（zxcheng@sjtu.edu.cn）
- **入库日期：** 2026-09-23
- **一句话说明：** **语义对齐 VTLA**：双路径触觉编码（预训练 ViT + **SA-ViT**）+ **ObjTac**（56 物体 / 10 类 / **135K** 视–触–文三模态样本）；真机 pick-and-place 夹爪 **96.9%**（**+21.9 pt**）、灵巧手 **100%**（**+6.2 pt**）；peg insertion **83.3%**（**+33.3 pt**）。

## 开源核查（2026-09-23）

| 项 | 状态 |
|----|------|
| 项目页 | <https://readerek.github.io/Objtac.github.io/> — ObjTac 说明、可视化、真机视频 |
| ObjTac 数据集 | **已发布** — 项目页 **Dataset** 按钮 → [Google Drive](https://drive.google.com/drive/folders/1jamNGWYhCk-uVKtrleF55WUpHctmOQBH?usp=drive_link)；arXiv 另列 `omnisharingdb.paxini.com` |
| OmniVTLA 训练/推理代码 | 项目页 **Code — Coming Soon**（弹窗提示） |
| SA-ViT 权重 | **未列** 独立 HF/GitHub 发布 |
| 结论 | **部分开源**（ObjTac 数据集已放；模型代码待发布） |

## 摘要级要点

- **问题：** VLA 偏视–语，触觉难获取且传感器异构；现有 VTLA 常把触觉当低维信号，缺 **与 CLIP/SigLIP 对齐的语义触觉表征**。
- **ObjTac：** Paxini Gen2 **力阵列触觉** + 720P 30 FPS 第一视角视频 + 文本描述；每物体 2–5 次交互、60 Hz 力数据；共 **270k** 力记录 → **135K** 配对样本。
- **SA-ViT：** 在 ObjTac 上对比学习，将触觉与视觉/语言概念对齐（材质、粗糙度、硬度等）。
- **OmniVTLA：** 参数匹配 controlled ablation 下的 **dual-path tactile encoder**（通用 ViT path + SA-ViT path）；端到端接触丰富操作。
- **轨迹：** 触觉 cues 使策略 **「远快近慢」** — 无接触快速接近、接触段平滑减速。
- **与 Awesome Touch 索引：** [`sources/papers/sun_awesome_touch_2508_08706_omnivtla-vision-tactile-language-action.md`](./sun_awesome_touch_2508_08706_omnivtla-vision-tactile-language-action.md) 为清单级摘录。

## 核心论文摘录（MVP）

### 1) 语义对齐 VTLA vs  vanilla VLA

- **链接：** <https://arxiv.org/html/2508.08706#S1> Figure 1
- **摘录要点：** 图像编码器继承 CLIP/SigLIP 式语义对齐；触觉侧需同等 **latent 语义对齐** 而非仅低维力向量拼接。
- **对 wiki 的映射：**
  - [OmniVTLA（Awesome Touch 实体）](../../wiki/entities/paper-sa-2508-08706-omnivtla-vision-tactile-language-action-model-wi.md)
  - [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)

### 2) ObjTac 三模态数据集

- **链接：** <https://arxiv.org/html/2508.08706#S3>；项目页 Dataset Details
- **摘录要点：** 56 物体 × 10 材质类；Paxini Gen2；文本 + 视频 + 力阵列；补充现有 visuo-tactile 数据缺口。
- **对 wiki 的映射：**
  - [触觉传感](../../wiki/concepts/tactile-sensing.md) — 力阵列 / 阵列触觉数据轴

### 3) 真机 pick-and-place 与 peg insertion

- **链接：** arXiv §Experiments；项目页 Real-World Experiments
- **摘录要点：** 夹爪 **96.9%**、灵巧手 **100%**；peg insertion **83.3%**；完成时间更短、轨迹更平滑。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md)
  - [VLA](../../wiki/methods/vla.md)

## 对 wiki 的映射（汇总）

- 实体页：[OmniVTLA（arXiv:2508.08706）](../../wiki/entities/paper-sa-2508-08706-omnivtla-vision-tactile-language-action-model-wi.md)
- 概念/任务：[VLA](../../wiki/methods/vla.md)、[视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)、[接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)
- 项目页归档：[`sources/sites/objtac-omnivtla.md`](../sites/objtac-omnivtla.md)

## 当前提炼状态

- [x] ObjTac 规格、SA-ViT / dual-path、真机结果、数据集/代码开源边界已摘录
- [x] 与 [`sources/sites/objtac-omnivtla.md`](../sites/objtac-omnivtla.md) 互证

## BibTeX

```bibtex
@article{cheng2025omnivtla,
  title={OmniVTLA: Vision-Tactile-Language-Action Model with Semantic-Aligned Tactile Sensing},
  author={Cheng, Zhengxue and Zhang, Yiqian and Tang, Anni and Wang, Keyu and Zhang, Wenkang and Li, Haoyu and Zhang, Hengdi and Song, Li},
  journal={arXiv preprint arXiv:2508.08706},
  year={2025},
  url={https://arxiv.org/abs/2508.08706},
}
```
