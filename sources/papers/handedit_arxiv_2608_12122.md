# HandEdit（arXiv:2608.12122）

> 来源归档（paper）

- **标题：** HandEdit: A Unified Benchmark for Egocentric Human-to-Robot Dexterous Hand Image Editing
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.12122>
- **PDF：** <https://arxiv.org/pdf/2608.12122>
- **项目页：** <https://handedit.github.io/>
- **代码：** <https://github.com/HandEdit/HandEdit>
- **数据集：** <https://huggingface.co/datasets/HandEdit/HandEdit>
- **入库日期：** 2026-09-20
- **一句话说明：** 200M+ URDF 条件 egocentric 人→灵巧机器人图像编辑实例；Hand-only / Hand-Arm 双轨；横评 11 类编辑器 + embodiment-aware 指标。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-09-20）：GitHub 评测工具链 + HF 数据集；伪 GT 构建以论文 + 发布数据为准。

## 核心摘录

1. **源数据：** EgoDex、ARCTIC、OakInk2、HOI4D、HO-Cap → 300K+ clip、600+ 场景、1.1K+ 物体。
2. **编辑定义：** 替换人手/手–臂为指定 URDF embodiment，保持物体、语义、接触、视角与场景。
3. **伪 GT：** 分割（SAM3）→ inpainting（ProPainter）→ 重定向 → 渲染 → 合成；Hand-Arm 轨固定 virtual base。
4. **评测：** 11 baselines；GPT-Image-2 综合最强；感知指标不足以代表 embodiment 编辑成功。
5. **与通用编辑 benchmark 差异：** 唯一同时支持 ego + dexterous hand + URDF-cond + 26 embodiment + 200M 规模（论文 Table 1）。

**对 wiki 的映射**

- [paper-handedit](../../wiki/entities/paper-handedit.md)
- [handedit-github-io](../sites/handedit-github-io.md)
- [handedit](../repos/handedit.md)
