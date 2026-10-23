# CompactSplat

> 来源归档（NeurIPS 2026 poster；论文 + 官方项目页 + 代码与权重）

- **标题：** CompactSplat: Spatially Adaptive Gaussian Distribution for Feedforward Scene Reconstruction
- **作者：** Bing He、Jingnan Gao、Yunuo Chen、Zhengxue Cheng、Rong Xie、Li Song、Wenjun Zhang
- **机构：** 上海交通大学（Shanghai Jiao Tong University）
- **发表：** NeurIPS 2026 Poster
- **arXiv：** <https://arxiv.org/abs/2610.13177>
- **项目主页：** <https://hebing-sjtu.github.io/CompactSplat-website/>
- **代码：** <https://github.com/hebing-sjtu/CompactSplat>
- **预训练权重：** <https://huggingface.co/binghe-sjtu/compactsplat>
- **入库日期：** 2026-10-23
- **一句话说明：** 面向稀疏视图前馈场景重建，以纹理引导分区与层级采样按区域分配 3D 高斯，并在推理时调节稀疏度，在画质、原语数量与内存之间折衷。

## 核心论文摘录

### 1) CompactSplat（Bing He 等，2026）
- **链接：** <https://arxiv.org/abs/2610.13177>
- **核心贡献：** 指出现有泛化式 3D Gaussian Splatting（3DGS）前馈重建常按像素均匀分配高斯，导致低纹理区域冗余且平面结构表达不足；提出 Texture-Aware Partitioner（TAP）与 Hierarchical Pyramid Sampler（HPS），让复杂区域用更密的高斯、平坦区域用尺度更大的高斯，并允许推理期调节稀疏度。论文在 RealEstate10K、DL3DV 与 ScanNet 上报告更少原语下的重建质量与高分辨率渲染一致性改进。
- **实现核查：** 官方仓库公开训练 / 评估入口及模型权重；仓库 README 提醒当前默认分区判据与阈值和论文设定仍有差异，复现实验需对照其 reproducibility notes。
- **对 wiki 的映射：**
  - [CompactSplat 论文实体](../../wiki/entities/paper-compactsplat.md)
  - [具身感知六种空间表征](../../wiki/concepts/embodied-perception-six-spatial-representations.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 关联 wiki 页面已添加入口

## 参考来源

- [arXiv 摘要与论文](https://arxiv.org/abs/2610.13177)
- [官方代码仓库与复现说明](https://github.com/hebing-sjtu/CompactSplat)
