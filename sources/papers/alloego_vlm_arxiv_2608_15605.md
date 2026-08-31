# AlloEgo-VLM（参照系消歧 VLM）

> 来源归档（ingest）

- **标题：** AlloEgo-VLM: Disambiguating Allocentric and Egocentric Reference Frames in Vision-Language Models
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.15605>
- **机构：** 国立阳明交通大学（NYCU，台湾）
- **代码：** <https://github.com/CKL9001/AlloEgo-VLM>
- **入库日期：** 2026-08-31
- **一句话说明：** AlloEgo-View 数据集 + SFT 集成到现有 VLM，消歧 allocentric / egocentric 空间语言；NVIDIA Isaac Sim 开放物体搜索部署验证。

## 核心摘录（MVP）

### 1) 参照系歧义

- **摘录要点：** 「左/右」可能来自观察者或环境中心视角；现有 VLM 在具身任务中答案不一致。
- **对 wiki 的映射：**
  - [AlloEgo-VLM](../../wiki/entities/paper-alloego-vlm.md) — 问题设定。

### 2) AlloEgo-View + SFT

- **摘录要点：** 图像—问题—视角特定答案三元组；监督微调集成到现有 VLM。
- **对 wiki 的映射：**
  - [AlloEgo-VLM](../../wiki/entities/paper-alloego-vlm.md) — 方法与数据。

### 3) 开源状态（截至 2026-08-31）

- **摘录要点：** **已开源**。`CKL9001/AlloEgo-VLM` 含 `Code/`、`Dataset/` 与论文 PDF。
- **对 wiki 的映射：**
  - [AlloEgo-VLM 仓库](../repos/ckl9001-alloego-vlm.md)

## 当前提炼状态

- [x] arXiv 摘要与仓库已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-alloego-vlm.md` 新建
