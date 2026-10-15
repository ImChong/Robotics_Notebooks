# MorphAct: Robot Policies Morphing with the Worlds They Act In（arXiv:2610.20019）

> 来源归档（ingest）

- **标题：** MorphAct: Robot Policies Morphing with the Worlds They Act In
- **arXiv：** <https://arxiv.org/abs/2610.20019>（cs.RO；提交日期 2026-10-14；PDF：<https://arxiv.org/pdf/2610.20019>；HTML：<https://arxiv.org/html/2610.20019v1>）
- **代码（官方仓库，已核查）：** <https://github.com/MINT-SJTU/MorphAct>（MIT；归档见本页「开源状态」）
- **项目页：** <https://mint-sjtu.github.io/MorphAct.io/>（页面链接 Code 指向上述仓库）
- **模型适配器：** Hugging Face <https://huggingface.co/MINT-SJTU/MorphAct-Metaworld>（Meta-World，三个骨干）
- **作者：** Ziyan Liu、Jiting Liu、Weihong Zhong、Wenda Yu、Yeqiu Chen、Jason Li、Xiaojiang Peng、Shuyuan Hu、Kai Wang、Bo Zhao（arXiv 页未列单位）
- **机构：** 项目页列出 University of Science and Technology of China、Shanghai Jiao Tong University、National University of Singapore、Nanyang Technological University、Tongji University、Shenzhen Technology University（转述自项目页，**未核 PDF 作者单位**）
- **入库日期：** 2026-10-15
- **一句话说明：** 为共享 VLA 策略按情境生成低秩参数更新：视觉语言侧先适配表示，动作侧以适配后的表示为条件生成更新，两侧用原生动作损失端到端联合训练；在 LIBERO 与 Meta-World 上提升 π0 / π0.5 / X-VLA，并报告真机结果。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-15）：** 页面 Code 按钮链接 `github.com/MINT-SJTU/MorphAct`；无单独的模型或数据集链接；未标注 "coming soon"（代码层面）。
- **仓库核查（2026-10-15）：** 仓库 README 含安装、`tools/serve.sh` / `tools/evaluate.sh` 快速开始、训练入口说明；Meta-World 三个骨干的预训练适配器已发布于 Hugging Face，**LIBERO 适配器标注 "coming soon"**。源码包含 `morphact/` 包（生成器、低秩 tokenizer、推理加速）、`openpi/scripts/train_morphact.py`、`openpi/scripts/serve_morphact.py`、`xvla/src/morphact_xvla/` 等。
- **结论：** **部分开源**（代码与 Meta-World 适配器可运行；LIBERO 适配器待发布）。

## 摘录 1：问题与主张（摘要）

- 通用机器人学习中，固定共享参数的策略在不同任务上梯度可能冲突；原子任务诊断显示视觉语言与动作两侧都存在此类冲突。
- 方法：为两侧生成上下文相关的低秩参数更新；动作侧更新以视觉语言表示为条件；两个生成器只用标准动作目标训练，不需要任务标签或专家适配器目标。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-morphact.md`](../../wiki/entities/paper-morphact.md)；与 [VLA](../../wiki/methods/vla.md)、[LoRA](../../wiki/concepts/lora.md) 互链。

## 摘录 2：结果（摘要与项目页）

- **LIBERO（均值成功率）：** X-VLA 93.05 → 97.20；π0 94.15 → 96.65；π0.5 96.85 → 98.60。
- **Meta-World（均值成功率）：** X-VLA 55.54 → 63.11；π0 50.75 → 67.05（最大 +16.30）；π0.5 48.70 → 64.42。
- **消融（X-VLA，LIBERO）：** 完整 97.20；固定条件化 87.50；任务 ID 条件化 88.25；仅 VLM 更新 43.50；仅动作侧更新 81.00；不适配特征 86.00。
- **真机（π0.5 → MorphAct，每任务 50 次）：** 分拣字母 82 → 92；五金分类 8 → 22；送水 78 → 86；柜子收纳 84 → 96（项目页）。摘要称真机相对 π0.5 最多约 14 个百分点提升。

## 摘录 3：工程代价（文章转述，未核 PDF）

- 参数每次重规划生成一次，在动作块内复用；文章称相对共享 LoRA，π0 / π0.5 动作块延迟增加约 18–21%。

## 建议 wiki 动作

- 新建 `wiki/entities/paper-morphact.md`（含流程总览与源码运行时序图）。
- 与 `wiki/methods/vla.md`、`wiki/concepts/lora.md` 互链（本次仅新增实体页，不改已有页）。
