# ACE-Brain-0.7 论文通读

- **标题：** ACE-Brain-0.7: Learning Execution-Grounded Reasoning for Embodied Agents
- **作者：** Yinhe Wang、Shijun Shi、Feng Pan 等（ACE-Brain Team）
- **机构：** 大晓机器人（ACE Robotics）、南洋理工大学、浙江大学
- **类型：** 论文（arXiv 预印本，cs.RO）
- **首次提交：** 2026-11-19
- **原文：** [arXiv:2611.26519v1 摘要](https://arxiv.org/abs/2611.26519v1) · [PDF](https://arxiv.org/pdf/2611.26519v1) · [HTML 全文](https://arxiv.org/html/2611.26519v1)
- **核对范围：** v1 正文 §1–§6 与附录 A.1–A.4，含作者机构、数据构造、评测口径、消融及失败案例。
- **通读：** 已完成
- **一句话说明：** 从带时间标注的执行记录构造监督，让视觉语言模型用实际执行状态判断进度、失败与下一子任务。
- **机构入口：** [大晓机器人官网](https://www.acerobotics.com/) · [具身基模型介绍](https://www.acerobotics.com/technology/jushenjimoxing)
- **发布报道：** [智能制造网转载大晓机器人微信公众号文章](https://m.gkzhan.com/news/detail/197685.html)（2026-11-24）。
- **代码、权重和数据：** 截至 2026-11-24，论文、官网具身基模型页、[DAXIAORobotics GitHub](https://github.com/DAXIAORobotics)、[ACE-Brain-Team GitHub](https://github.com/ACE-Brain-Team)及 [ACERobotics Hugging Face](https://huggingface.co/ACERobotics) 中暂未核实 0.7 的公开实现、权重或完整训练数据下载入口。

## 核心摘录

- **任务结构与执行状态（§3）：** 任务结构记录子目标、先后依赖和完成条件。执行状态记录某次执行中各子目标的当前状态和证据帧。下一步查询从前置条件已完成且自身未完成的子目标中选择。
  **对 wiki 的映射：** [任务结构与执行状态](../../wiki/entities/paper-ace-brain-0-7.md#核心原理)。
- **监督构造与训练（§4、附录 A.3）：** 从第一视角人类视频、第三视角机器人片段和仿真等构造约 370 万组问答。第一视角数据的证据帧标注与独立步骤时间线交叉核对。Qwen3.5-9B 冻结视觉编码器后监督微调，最终混合约 380 万任务问答、350 万空间锚定样本、20 万进度样本。
  **对 wiki 的映射：** [流程与训练组成](../../wiki/entities/paper-ace-brain-0-7.md#流程总览)。
- **评测与对照（§5、附录 A.2、A.4）：** 六项任务推理均分 76.5，七项空间锚定均分 75.90，200 项进度评测 MAE 0.087。114 组时序配对上，仅任务数据微调的模型将两题均答对的比例从 15.8% 提高到 54.4%。RoboBench Q1/Q2 改用自然语言和模型裁判，分数采用论文自己的协议。
  **对 wiki 的映射：** [实验与评测](../../wiki/entities/paper-ace-brain-0-7.md#实验与评测)。

## 对 wiki 的映射

- [ACE-Brain-0.7 论文实体](../../wiki/entities/paper-ace-brain-0-7.md)
