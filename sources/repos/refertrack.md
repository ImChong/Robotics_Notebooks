# ReferTrack（MedlarTea/referTrack）

> 来源归档

- **标题：** ReferTrack
- **类型：** repo（**占位仓库，训练/评测/权重待发布**）
- **来源：** 南方科技大学 RCV Laboratory · 腾讯 Robotics X · 北京大学 · 福田实验室
- **链接：** <https://github.com/MedlarTea/referTrack>
- **论文：** <https://arxiv.org/abs/2607.20061> — 归档见 [`sources/papers/refertrack_arxiv_2607_20061.md`](../papers/refertrack_arxiv_2607_20061.md)
- **项目页：** <https://medlartea.github.io/referTrack/> — 归档见 [`sources/sites/medlartea-refertrack.md`](../sites/medlartea-refertrack.md)
- **视频：** <https://youtu.be/CP7h-tWWABU>
- **许可：** 仓库未附 LICENSE（截至入库日）
- **入库日期：** 2026-08-12
- **一句话说明：** 官方代码仓；README 概述 referring-then-tracking 范式，但 TODO 显示 checkpoint、数据集、训练代码与 data engine 均未发布。
- **沉淀到 wiki：** [`wiki/entities/paper-refertrack.md`](../../wiki/entities/paper-refertrack.md)

---

## 开放程度核查（2026-08-12）

| 项 | 状态 | 依据 |
|----|------|------|
| 模型 checkpoint | **未发布** | README TODO：*Release model checkpoints and evaluation code* |
| 评测代码 | **未发布** | 同上 |
| 数据集 | **未发布** | TODO：*Release the dataset* |
| 训练代码 | **未发布** | TODO：*Release the training code* |
| Data engine | **未发布** | TODO：*Release the data engine* |
| 仓库文件 | `README.md`、`assets/`、`method.pdf` | GitHub 目录树（无 `train.py` / eval 入口） |

**结论：宣称将开源 / 待发布。** 论文实体页不得写「已开源」；`## 源码运行时序图` 按 **不适用** 处理，待正式 release 后按实际入口补 sequenceDiagram。

---

## README 声明的能力（供发布后核对）

| 能力 | README / 论文措辞 |
|------|-------------------|
| Referring-then-tracking | 先图像空间 bbox 接地，再解码跟踪航点 |
| TVBI | 将历史选定框几何注入视觉历史以保留目标运动线索 |
| EVT-Bench 单视角 | 报告 STT / DT / AT 上 SOTA 级单视角结果 |
| Sim2Real | 腿式与人形机器人真机迁移 |

实现依赖（论文）：基于 [OpenTrackVLA](https://github.com/om-ai-lab/OpenTrackVLA)；检测 `YOLO11 + ByteTrack`；LLM 骨干 Qwen3-4B。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-refertrack](../../wiki/entities/paper-refertrack.md) | 论文实体与结论 |
| [qwen-robot-nav](../../wiki/entities/qwen-robot-nav.md) | 同报 EVT-Bench tracking 的导航 VLA 对照 |
| [vision-language-navigation](../../wiki/tasks/vision-language-navigation.md) | 语言条件导航 / 跟踪任务族 |
| [paper-notebook-navila](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md) | 腿式导航 VLA + 真机对照 |
