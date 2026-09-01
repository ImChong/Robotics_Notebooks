# LightNav-0（通用具身导航 VLM）

> 来源归档（ingest）

- **标题：** LightNav-0: Eliciting VLM Spatial Intelligence for Generalist Embodied Navigation
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.30935>
- **机构：** 光原点（Light Origins）
- **项目页：** <https://www.lightorigins.com/en/blog/lightnav-0>
- **代码：** <https://github.com/lightorigins/LightNav-0>
- **模型：** <https://huggingface.co/LightOriginsHQ/LightNav-0>
- **入库日期：** 2026-09-01
- **一句话说明：** 统一 token 接口对齐预训练 VLM 空间智能：dual-channel pointing 表达任务/场景/本体无关空间意图，RVQ action tokenizer 映射轨迹；10 个公开导航仿真 SOTA + 跨本体零样本真机。

## 核心摘录（MVP）

### 1) 统一空间意图编码

- **摘录要点：** 具身导航需把异构目标与视觉观测转成动作；任务/本体专用头限制泛化。LightNav-0 用 dual-channel pointing + residual VQ action tokenizer 统一空间意图与轨迹。
- **对 wiki 的映射：**
  - [LightNav-0](../../wiki/entities/paper-lightnav-0.md)

### 2) 训练与数据

- **摘录要点：** 视觉历史压缩、ER mid-training、SFT 与 RL；2K+ 场景、4K+ 小时数据；支持指令跟随、开放词汇目标导航、视觉跟踪。
- **对 wiki 的映射：**
  - [LightNav-0](../../wiki/entities/paper-lightnav-0.md)

### 3) 开源状态（截至 2026-09-01）

- **摘录要点：** **已开源** 代码 + Hugging Face 权重。
- **对 wiki 的映射：**
  - [LightNav-0 项目页](../sites/lightnav-0.md)
  - [lightorigins/LightNav-0](../repos/lightorigins-lightnav-0.md)

## 当前提炼状态

- [x] 项目页/仓库已交叉核查
- [x] wiki 映射：`wiki/entities/paper-lightnav-0.md` 新建
