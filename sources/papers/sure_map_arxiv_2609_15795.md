# sure_map_arxiv_2609_15795

> 来源归档（ingest）

- **标题：** SURE-Map: Self-Correcting Streaming Geometric Foundation Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.15795>
- **项目页：** <https://mingkai-liu.github.io/projects/sure-map/>
- **代码：** <https://github.com/RCL-Robotics/SURE-map>
- **权重：** <https://huggingface.co/milchstrasse/SURE-Map>（cross-view uncertainty head）；骨干实验用 [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map) 的 `lingbot-map.pt`
- **视频：** <https://www.youtube.com/watch?v=vIKJFzLCtMc>
- **作者：** Mingkai Liu（MBZUAI / PKU）；Hao Zhao（清华，通讯）；Xingxing Zuo（MBZUAI，通讯）
- **入库日期：** 2026-09-24
- **一句话说明：** 在 **流式几何基础模型**（VGGT / LingBot-Map 系）之上引入 **跨视几何不确定性**（联合位姿–深度是否诱导一致跨视像素对应）与 **多时间尺度自校正**（快因果帧 + 稀疏 keyframe 全注意力窗尺度重标定），缓解动态物体/弱纹理下的局部误差累积与长程尺度漂移；长户外基准上报告相对 LingBot-Map 等基线的 **ATE-RMSE** 下降（KITTI 24.00→17.24 m 等），可选 loop-closure 进一步改善。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract）

- **链接：** <https://arxiv.org/abs/2609.15795>
- **核心贡献：** 流式前馈重建每帧仅见有限上下文，易受动态与弱纹理影响；小局部误差会累积为几何畸变与 **长程尺度漂移**。作者主张可靠流式重建需要模型 **可预测且可自校正**；提出 **SURE-Map** 两原则：**跨视几何不确定性** + **多时间尺度自校正**。
- **对 wiki 的映射：**
  - [paper-sure-map](../../wiki/entities/paper-sure-map.md)

### 2) 跨视几何不确定性（Cross-View Geometric Uncertainty）

- **链接：** <https://arxiv.org/abs/2609.15795>；项目页 Method §01
- **核心贡献：** 不同于单视深度/点置信度，该不确定性直接衡量 **联合预测的位姿与深度** 是否产生 **几何一致的跨视像素对应**；用于 **稠密点过滤** 与 **局部平移优化**。
- **对 wiki 的映射：**
  - [paper-sure-map](../../wiki/entities/paper-sure-map.md)
  - [LingBot-Map](../../wiki/methods/lingbot-map.md)（骨干与对照基线）

### 3) 多时间尺度自校正（Multi-Timescale Self-Correction）

- **链接：** <https://arxiv.org/abs/2609.15795>；项目页 Method §02
- **核心贡献：** **快路径** 保持逐帧因果流式效率；**稀疏 keyframe-window** 全注意力推理提供更长程几何证据，**周期性重标定近期轨迹尺度**，抑制仅局部校正无法消除的慢累积尺度误差。
- **对 wiki 的映射：**
  - [paper-sure-map](../../wiki/entities/paper-sure-map.md)
  - [State Estimation](../../wiki/concepts/state-estimation.md)

### 4) 长程位姿基准（实验摘要）

- **链接：** 项目页 Abstract / README Highlight
- **核心贡献（ATE-RMSE，相对 LingBot-Map 等报告值）：** KITTI **24.00→17.24 m**（+ loop closure **15.17 m**）；Oxford Spires **5.11→4.74 m**（**4.63 m**）；VBR **31.37→28.58 m**（**22.12 m**）。
- **对 wiki 的映射：**
  - [paper-sure-map](../../wiki/entities/paper-sure-map.md)

## BibTeX（README Citation 区）

```bibtex
@article{liu2026suremap,
  title={SURE-Map: Self-Correcting Streaming Geometric Foundation Models},
  author={Liu, Mingkai and Zhao, Hao and Zuo, Xingxing},
  journal={arXiv preprint arXiv:2609.15795},
  year={2026}
}
```

## 当前提炼状态

- [x] 摘要与两大机制对齐项目页
- [x] 开源核查（项目页 + GitHub + HF）
- [x] wiki 页面映射确认
