# SceneVerse++（互联网视频驱动的 3D 场景理解数据）

- **标题**: SceneVerse++
- **项目页**: https://sv-pp.github.io/
- **论文**: https://arxiv.org/abs/2604.01907 （*Lifting Unlabeled Internet-level Data for 3D Scene Understanding*，CVPR 2026）
- **代码**: https://github.com/sv-pp/SceneVersepp
- **类型**: dataset / code-release / paper
- **机构**: 通用人工智能国家重点实验室（BIGAI），联合北大、清华、北邮、北理工等
- **收录日期**: 2026-05-07

## 一句话摘要

从互联网无标注长视频中自动重建相机姿态与几何，并流水线生成实例分割与高层语义标注，构建 **6687** 个真实室内场景的端到端 3D 场景理解训练数据，用以补足 ScanNet 以来高质量有标注 3D 数据规模化不足的瓶颈。

## 为何值得保留

- **任务覆盖面**：同时支撑低层几何感知（3D 检测 / 实例分割）与高层空间推理（**3D 空间 VQA**、**视觉–语言导航 VLN**），与具身智能中的空间推理和指令跟随导航直接相关。
- **方法论**：系统讨论「自动数据引擎」各子模块的质量–效率权衡与误差传播，而不只是把流水线当作黑盒。
- **开源**：论文配套项目页、代码与数据发布，便于复现与二次利用。

## 数据管线要点（来自论文公开描述）

1. **视频策展与稀疏几何**：TransNetV2 切镜；过滤低质量/室外等片段；基于视差选关键帧；密集像素匹配 + 全局 BA（风格接近 Mast3R-SFM），得到相机姿态与稀疏点云。
2. **稠密重建**：利用 SfM 稀疏深度作为先验，PriorDA 预测度量稠密深度，TSDF 融合为封闭网格。
3. **实例分割**：CropFormer 等 2D mask 提升到 3D，多视图一致性聚合；Describe Anything + Qwen2-VL 生成实例文本并对齐到 ScanNet 类别。
4. **3D 空间 VQA**：将几何与语义写入 **3D 场景图**，按模板生成计数、相对距离/方向、物体大小、房间尺度等 QA；路线规划类问题结合导航轨迹与 VLM 摘要。
5. **VLN（R2R 风格）**：对室内 tour 类视频轨迹做聚类去冗余、切段、离散化为 R2R 动作空间；VLM 生成三种风格的自然语言指令。

## 规模与验证线索（论文报告）

- **场景数**：8217 段互联网视频 → 6687 个场景；室内、多楼层、长轨迹，覆盖面积与物体多样性相对 ScanNet / ARKitScenes 等的特点在文中对比。
- **VQA**：约 **632K** 条空间 VQA（与 VSI-Bench 格式对齐）；报告对 Qwen2.5-VL 等在 VSI-Bench 上的增益与域差距讨论。
- **VLN**：约 **9631** 条轨迹；报告在 R2R 等设定下微调后导航成功率额外提升（文中给出约 **+14%** 量级的改进叙述）。
- **检测 / 分割**：与 SpatialLM、Mask3D 等结合，展示预训练 + 微调相对合成数据或纯 ScanNet 的差异（含域偏差讨论）。

## 对 Wiki 的映射

- **wiki/entities/sceneverse-pp.md**：数据集与工程实体页。
- **wiki/concepts/3d-spatial-vqa.md**：补全「3D 空间视觉问答」概念，衔接 VLM 空间推理与具身任务。
- **wiki/tasks/vision-language-navigation.md**：补全 VLN 任务定义及与机器人导航的关系。
- **wiki/methods/auto-labeling-pipelines.md**、**wiki/methods/vla.md**：作为大规模自动标注与 VLM 空间能力的数据侧参照。
