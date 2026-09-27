# Show-Harness: Just a VLM Agent Can Play Robots（arXiv:2609.10522）

> 来源归档

- **arXiv：** <https://arxiv.org/abs/2609.10522>
- **PDF：** <https://arxiv.org/pdf/2609.10522>
- **项目页：** <https://showlab.github.io/Show-Harness/>
- **代码：** <https://github.com/showlab/Show-Harness>
- **模型：** <https://huggingface.co/showlab/Show-Harness-VLMs>
- **数据：** <https://huggingface.co/datasets/showlab/Show-Harness-Data>
- **机构：** Show Lab，新加坡国立大学（NUS）
- **作者：** Yanzhe Chen*、Zechen Bai*、Zhijun Cao*、Wenzheng Zeng*、Kevin Qinghong Lin、Yiqi Lin、Guoqiang Liang、Kevin Yuchen Ma、Qiming Huang、Mike Zheng Shou†
- **开源状态：** **已开源**（步骤 2.5 核查项目页 + GitHub，2026-09-27）
- **入库日期：** 2026-09-10（公众号索引）；2026-09-27（官方元数据与 HF 链补全）
- **一句话说明：** 离散、增量语义动作单元作为 VLM 与机器人之间的接口；本体解释器确定性落地；frontier VLM 零样本与小模型 LoRA 微调共用同一词表；GUMI 用 GUI 采集跨本体演示。

## 核心摘录（对 wiki 的映射）

1. **语义动作空间** — 将控制 reformulate 为无参符号单元（如 `MV_LEFT`、`GRASP`）；解释器供给度量步长，同一词表驱动 Franka、AgileX 与仿真。**→** [paper-show-harness](../../wiki/entities/paper-show-harness.md)「核心原理」
2. **闭环 Harness** — 插件组装多视角与本体感知上下文，VLM 每步输出一个语义单元，解释器执行后反馈进入历史。**→** 流程总览 + 源码运行时序图
3. **双模式** — (1) 闭源 frontier VLM 零样本；(2) 2B 级开源 VLM 在 GUMI 数据上 **数 GPU·小时** LoRA 微调。**→** 工程实践
4. **GUMI** — GUI 按键映射动作单元，人与 GUI agent 同接口采集，无需专用遥操作硬件。**→** `sources/repos/show-harness.md` / `gumi/`
5. **泛化实验（项目页 headline）** — 跨任务 89%/86%（ZS/FT）、跨环境 100%/88%、跨本体 93%/87%；仅仿真训练的 FT 真机 13/20，可训练 VLA 基线 0/20。**→** 实验与评测（以 PDF 表为准）

**对 wiki 的映射：** [paper-show-harness](../../wiki/entities/paper-show-harness.md)

**交叉归档：** [show-harness 项目页](../sites/show-harness.md) · [show-harness 仓库](../repos/show-harness.md)
