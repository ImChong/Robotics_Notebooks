# ECoT 项目页（embodied-cot.github.io）

- **类型**：项目静态站点
- **收录日期**：2026-09-09
- **站点**：<https://embodied-cot.github.io/>
- **论文**：<https://arxiv.org/abs/2407.08693>（CoRL 2024）
- **代码：** <https://github.com/MichalZawalski/embodied-CoT>
- **权重 / 数据：** <https://huggingface.co/Embodied-CoT>

## 一句话

**Embodied Chain-of-Thought（ECoT）** 奠基页：VLA 在动作前生成多层次具身推理链；Bridge V2 合成标注管线；相对 OpenVLA **+28%** 绝对成功率。

## 开源核查（2026-09-09）

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — `MichalZawalski/embodied-CoT`（基于 OpenVLA） |
| **权重** | **已发布** — HF `Embodied-CoT/ecot-openvla-7b-bridge`、`ecot-openvla-7b-oxe` |
| **推理 demo** | Colab notebook；TensorRT-LLM 加速（外部 `tensorrt-openvla`） |

## 站点摘录要点

- **推理模块**：高层 TASK / PLAN / SUBTASK + 低层 MOVE / GRIPPER / bbox 等视觉接地特征。
- **数据**：多基础模型子模块从演示提取特征 → 文本推理链；主训练数据为 Bridge V2 ECoT 标注。
- **实验**：14 项真机泛化任务；优于 Octo / OpenVLA / RT-2-X；支持自然语言纠错；未见本体上推理迁移。
- **机构**：UC Berkeley、Stanford、University of Warsaw。

## 对 wiki 的映射

- 主沉淀：[ECoT](../../wiki/entities/paper-ecot.md)
- 原始论文档：[ecot_arxiv_2407_08693.md](../papers/ecot_arxiv_2407_08693.md)
- 代码入口：[embodied-cot.md](../repos/embodied-cot.md)
