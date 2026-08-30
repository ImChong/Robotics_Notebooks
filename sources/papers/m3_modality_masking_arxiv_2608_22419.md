# M3（双臂 VLA 模态遮蔽）

> 来源归档（ingest）

- **标题：** Robust Bimanual Vision-Language-Action Models via Embarrassingly Simple Modality Masking
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.22419>
- **机构：** 上海创智学院（Shanghai Innovation Institute）；香港城市大学（City University of Hong Kong）
- **作者：** Dongzhou Cheng、Ziang Li、Yixiao Zhou、Haojuan Li、Jinghao Zhang、Lei Lei、Minjing Dong、Jie Gui、Jiaqi Wang
- **项目页：** <https://m3vla.github.io/>
- **入库日期：** 2026-08-30
- **一句话说明：** 训练期随机屏蔽腕相机/语言/动作查询通道，不改推理结构，提升查询式双臂 VLA 的多视角鲁棒性。

## 核心摘录（MVP）

### 1) 查询式 VLA 的注意力分散

- **摘录要点：** 查询式 VLA 延迟低，但复杂双臂任务仍出现动作不连续；作者把原因之一归到多视角与语言融合不稳、注意力被干扰区拉走。
- **对 wiki 的映射：**
  - [M3](../../wiki/entities/paper-m3-modality-masking.md)
  - [VLA](../../wiki/methods/vla.md)

### 2) 训练期结构化遮蔽

- **摘录要点：** 保留 egocentric；左右腕成对遮蔽；语言可整模态隐藏；动作查询元素级遮蔽且至少留一个。推理时全部恢复。相对 token/modality dropout 与视觉增强，结构化遮蔽增益更大。
- **对 wiki 的映射：**
  - [M3](../../wiki/entities/paper-m3-modality-masking.md)
  - [双臂操作](../../wiki/tasks/bimanual-manipulation.md)

### 3) 评测数字

- **摘录要点：** RoboTwin 2.0 十任务、每任务 50 条干净示范。Adapter Clean 平均 **41.0→62.7（+21.7）**；Clean2Rand **+11.4**。OpenVLA-OFT **32.2→53.5（+21.3）**。Agilex 三长时程真机：干净完整任务 **44.4→69.4**，OOD **12.5→61.1**。
- **对 wiki 的映射：**
  - [M3](../../wiki/entities/paper-m3-modality-masking.md)

### 4) 开源状态（截至 2026-08-30）

- **摘录要点：** **未开源**。项目页有方法图、表与真机视频；配套 GitHub 为 Pages 站，未见训练仓。

## 当前提炼状态

- [x] 项目页与 arXiv 摘要对齐
- [x] 开源边界已写入
- [x] wiki 映射：`wiki/entities/paper-m3-modality-masking.md` 新建
