# HandEdit 项目页（handedit.github.io）

> 来源归档

- **标题：** HandEdit: A Unified Benchmark for Egocentric Human-to-Robot Dexterous Hand Image Editing
- **类型：** site（项目页 + 评测工具链）
- **URL：** <https://handedit.github.io/>
- **论文：** <https://arxiv.org/abs/2608.12122>
- **代码 / 评测：** <https://github.com/HandEdit/HandEdit>
- **数据集：** <https://huggingface.co/datasets/HandEdit/HandEdit>
- **机构：** 复旦大学（Fudan）· 上海交通大学（SJTU）· 香港大学（HKU）· Inspire Robots 等（与 Bench2Dex 同团队脉络）
- **入库日期：** 2026-09-20
- **一句话说明：** 第一人称 **人→灵巧机器人** 图像编辑数据集与基准：5 源数据集、300K+ clip、200M+ 编辑实例、26 URDF embodiment；Hand-only / Hand-Arm 双轨 + URDF 条件评测；横评 11 类图像编辑器。

## 开源核查（步骤 2.5，2026-09-20）

| 资源 | 状态 | 说明 |
|------|------|------|
| 评测工具链 | **已开源** | [HandEdit/HandEdit](https://github.com/HandEdit/HandEdit)：`build_manifest.py`、`eval.py`、Harmonizer 推理包装 |
| 数据集 | **已发布** | HF [HandEdit/HandEdit](https://huggingface.co/datasets/HandEdit/HandEdit) |
| 伪 GT 构建管线源码 | **部分** | README 详述分割 / inpainting / 重定向 / 渲染 / 合成流程；完整数据构建脚本以 HF 元数据与论文为准 |
| 商用编辑器权重 | **第三方** | GPT-Image-2 等 API/闭源模型仅作 baseline，非本仓发布 |

**判定：已开源（数据集 + 评测）；构建管线为论文 + HF 发布物。** 复现 benchmark 评分可直接 clone 仓库；生成 200M 实例需拉 HF 数据。

## 公开要点（编译自项目页 + README，2026-09-20）

### 问题定义

将 egocentric 帧中可见的 **人手或手–臂区域** 替换为指定 **灵巧机器人 embodiment**，同时保持物体状态、任务语义、接触关系、视角与场景结构——**embodiment-aware image editing**，非通用 style transfer。

### 规模

| 项 | 数值 |
|----|------|
| 源数据集 | EgoDex、ARCTIC、OakInk2、HOI4D、HO-Cap（5） |
| 视频 clip | 300K+ |
| 编辑实例 | 200M+ |
| 目标 URDF | 26（13 hand-only + 13 hand-arm） |
| 场景 / 物体 | 600+ 场景、1.1K+ 物体 |

### 双 benchmark 轨道

- **Hand-only：** 仅替换可见人手为目标 robot hand。
- **Hand-Arm：** 替换手–臂区域为目标 arm–hand；每序列固定 virtual base（27 候选基座、IK/碰撞筛除）。

### 伪 GT 流水线

SAM3 分割 → ProPainter 背景修复 → MANO/3D 手姿 embodiment 重定向 → 机器人渲染 → 合成；Harmonizer 可选后处理（1/10 官方测试集附加分析）。

### 评测指标

- **通用：** PSNR / SSIM / LPIPS / FID（Full / ROI / Background）。
- **VLM 判断**
- **Embodiment-aware：** Removal、Struct Fidelity、ID Fidelity、Interaction Retention

### 主要结论（项目页）

1. **GPT-Image-2** 综合 baseline 最强。
2. VLM 判断有用但不足以单独代表编辑任务成功。
3. 感知质量高 ≠ 编辑任务成功（embodiment / 交互保留需专用指标）。

### 用例：LongCat-Image LoRA

在 HandEdit 对齐对上 LoRA 微调（rank 32、2 epoch），展示 human→Inspire 手替换 while 保留物体与接触。

## 对 wiki 的映射

- [paper-handedit](../../wiki/entities/paper-handedit.md)
- [handedit 仓库](../repos/handedit.md)
- [handedit_arxiv_2608_12122](../papers/handedit_arxiv_2608_12122.md)
- 交叉：[paper-notebook-egodex-learning-dexterous-manipulation-from-larg](../../wiki/entities/paper-notebook-egodex-learning-dexterous-manipulation-from-larg.md)、[macrodata-egocentric-hand-action](../../wiki/methods/macrodata-egocentric-hand-action.md)、[manipulation](../../wiki/tasks/manipulation.md)
