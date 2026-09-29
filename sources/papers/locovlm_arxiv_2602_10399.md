# LocoVLM: Grounding Vision and Language for Adapting Versatile Legged Locomotion Policies

> 来源归档（ingest · arXiv + 项目页）

- **标题：** LocoVLM: Grounding Vision and Language for Adapting Versatile Legged Locomotion Policies
- **类型：** paper
- **原始链接：** https://arxiv.org/abs/2602.10399
- **项目页：** https://locovlm.github.io/
- **作者：** I Made Aswin Nahrendra, Seunghyun Lee, Dongkyu Lee, Hyun Myung
- **机构：** 韩国科学技术院（KAIST）；URobotics Corp.
- **发表：** ICRA 2025 Workshop on Safe Vision-Language Models（SafeVLMs）
- **入库日期：** 2026-09-28
- **一句话说明：** 用离线 LLM 规模化生成「自然语言指令 → 步态周期/相位/速度上限」技能库，机载 BLIP-2 对文本或图像做混合精度检索，驱动风格条件 + 柔顺接触跟踪的腿足策略，推理无需在线查询云上大模型。

## 核心摘录（策展，非全文）

### 1) 问题：几何感知主导，语义与指令难进闭环

- **摘录要点：** 腿足 locomotion 学习仍多依赖环境几何表征，难以响应人类指令与环境高层语义（场景类型、隐喻式指令）。
- **对 wiki 的映射：**
  - [paper-locovlm](../../wiki/entities/paper-locovlm.md) — 问题设定。

### 2) 离线 LLM 技能库 + 机载 VLM 检索

- **摘录要点：** GPT-4o **离线**两阶段：先按「模仿行为 / 场景响应 / 直接指令」生成多样指令，再用 meta-prompt 推理映射到结构化 **motion descriptor**（步态周期、各足相位偏置、速度上限）。推理时用 **BLIP-2** 将文本或相机图像嵌入技能库；**mixed-precision retrieval**（余弦粗筛 + ITM 头重排）与 **text-as-image**（把字符串渲染成图再进图像编码器）将检索准确率提到约 **87%**（100 条人工标注指令集）。
- **对 wiki 的映射：**
  - [paper-locovlm](../../wiki/entities/paper-locovlm.md) — 数据与检索管线。
  - [LLM 机器人控制接口](../../wiki/concepts/llm-robotics-control-interfaces.md) — 大模型停在「顾问/检索层」、不进入力矩环。

### 3) 风格条件策略与柔顺步态跟踪

- **摘录要点：** 低层为 **style-conditioned** locomotion policy，参数化步态周期、相位偏置与速度上限，可表达 pronk/trot/pace/bound/rotary gallop 等；**compliant contact tracking** 在相位合规带内允许偏离目标步态以换扰动鲁棒性。
- **对 wiki 的映射：**
  - [locomotion](../../wiki/tasks/locomotion.md) — 任务语境。
  - [gait-generation](../../wiki/concepts/gait-generation.md) — 步态参数化。

### 4) 系统指标与平台

- **摘录要点：** 报告 **<100 ms** VLM 推理、**50 Hz** 机载控制；**87%** 指令跟随（混合精度 + text-as-image）；**Unitree Go1** 实机（含校园路面/雪地场景图像自适应）；**Unitree H1** 人形 MuJoCo **零样本**复用同一技能库（仅重训人形风格策略，VLM 与库不重做）。
- **对 wiki 的映射：**
  - [paper-locovlm](../../wiki/entities/paper-locovlm.md) — 评测与部署读法。

### 5) 开源状态（截至 2026-09-28）

- **摘录要点：** 项目页无 GitHub；[anahrendra/locovlm](https://github.com/anahrendra/locovlm) 为占位 README → **待发布**。
- **对 wiki 的映射：**
  - [locovlm 项目页](../sites/locovlm.md)
  - [anahrendra-locovlm.md](../repos/anahrendra-locovlm.md)

## 对 wiki 的映射

- [paper-locovlm](../../wiki/entities/paper-locovlm.md)
- [locovlm 项目页](../sites/locovlm.md)

## 参考来源（原始）

- [arXiv:2602.10399](https://arxiv.org/abs/2602.10399)
- [LocoVLM 项目页](https://locovlm.github.io/)
- [Workshop PDF（SafeVLMs）](https://locovlm.github.io/static/images/workshop_final.pdf)
