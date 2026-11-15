# OV-SAM3D 原文通读与实现边界

- **类型：** paper
- **主论文：** Open-Vocabulary SAM3D: Towards Training-free Open-Vocabulary 3D Scene Understanding，arXiv:2405.15580。
- **原文：** <https://arxiv.org/html/2405.15580>；2026-11-15 获取的 HTML 正文与附录。
- **项目页：** <https://hithqd.github.io/projects/OV-SAM3D/>
- **代码：** <https://github.com/HanchenTai/OV-SAM3D>；另见[实现归档](../repos/ov-sam3d.md)。
- **核对范围：** 摘要、§1–5（相关工作、两阶段方法、公式/算法、ScanNet200 与 nuScenes 实验、消融及结论）、附录 A（nuScenes 重建与标签转换）与 B（补充定性结果说明）；正文图注及实验表。另读官方 README、`ovsam3d/tools/generate_coarse_masks.py` 与 `refine_masks.py`，未运行模型。
- **通读：** 已完成
- **核对日期：** 2026-11-15

## 方法与原文证据

1. 输入是点云/网格及已知位姿的 RGB-D 图像。选较大的超点作为 3D 提示，投影到遮挡较少的视角，引导 SAM 分割；反投影后累积超点与提示间的重叠得分表。
2. 用重叠向量内积与交互阈值合并粗 mask，更新合并结果并保存组成关系，避免大物体跨视角无直接重叠时被割裂。
3. 论文以 RAM 生成标签、ChatGPT 去除材料/颜色/房间等非实例标签，再用 CLIP 匹配已有裁剪特征；最终标签取组成 mask 中相似度最高者。公开 `refine_masks.py` 负责几何合并，不能把论文全部语义步骤都写成该脚本内的动作。

## 实验、代价与局限

- ScanNet200 表 1：AP/AP50/AP25 为 9.0/13.6/19.4，OpenMask3D 为 15.4/19.9/23.1。本文优势是无需针对场景训练，不能写成所有室内指标均超过训练式方法。
- nuScenes 表 2：AP/AP50/AP25 为 8.9/16.0/29.1。附录先用 alpha-shape（alpha=0.5）将点云重建为网格，再由框标注推断实例并过滤/合并类别；这不是直接使用原始驾驶数据的原生实例分割协议。
- §4.1 单 GPU 场景耗时：ScanNet 约 4–6 分钟，nuScenes 约 30–60 秒；没有给机载实时控制保证。
- 表 3/4 分别审计反投影、重合并、标签匹配策略与提示数/合并阈值。基础模型识别错误、标定与场景超参数仍影响最终输出。

## 对 wiki 的映射

- [OV-SAM3D](../../wiki/entities/ov-sam3d.md)
