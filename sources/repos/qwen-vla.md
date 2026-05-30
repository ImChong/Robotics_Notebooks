# Qwen-VLA

> 来源归档（ingest）

- **标题：** Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments
- **类型：** repo + paper（arXiv）+ 官方博客 / Demo
- **组织：** Qwen Team（阿里巴巴通义）
- **代码：** https://github.com/QwenLM/Qwen-VLA
- **论文：** https://arxiv.org/abs/2605.30280
- **技术报告 / 博客 / Demo：** README 内链（`Technical Report` | `Blog` | `Demo`）
- **入库日期：** 2026-05-30
- **一句话说明：** 基于 **Qwen3.5-4B** 视觉–语言骨干与 **1.15B DiT flow-matching 动作解码器** 的 **统一 VLA 通才**：将操作、导航与轨迹预测纳入同一动作–轨迹预测空间，通过 **本体感知 prompt 条件** 在多平台间切换而无需每平台独立输出头；渐进式训练含大规模动作预训练、多模态持续预训练、SFT 与 RL。
- **沉淀到 wiki：** [Qwen-VLA](../../wiki/entities/qwen-vla.md)

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [VLA](../../wiki/methods/vla.md) | **Qwen3 系 VLM + DiT flow matching** 的工业级 **通才** 实例；同时覆盖 **操作与 VLN** |
| [Xiaomi-Robotics-0](../../wiki/entities/xiaomi-robotics-0.md) | 同为 **Qwen3-VL 4B 级 + DiT flow** 路线，但小米侧重 **异步 chunk 部署**；Qwen-VLA 强调 **单权重跨本体/任务** |
| [StarVLA](../../wiki/methods/star-vla.md) | 同 **Qwen3-VL** 生态；StarVLA 偏 **极简 MLP 头基准**，Qwen-VLA 偏 **大规模通才预训练与导航–操作统一** |
| [VLN](../../wiki/tasks/vision-language-navigation.md) | README 报告 **R2R / RxR** 等导航基准，与操作基准 **同一模型联合评测** |
| [Loco-manipulation](../../wiki/tasks/loco-manipulation.md) | 「操作 + 导航」统一策略的落地语境 |

---

## 设计要点（README / 技术报告归纳）

1. **统一动作–轨迹框架：** 操作（manipulation）、导航（navigation）、以自我为中心的动作建模与 **轨迹预测** 共享一个预测空间，而非为每类任务单独训练 specialist。
2. **本体感知 prompt 条件：** **一套权重** 服务多平台；切换机器人/环境主要通过 **文本 prompt** 描述 embodiment，**无需每平台输出头**。
3. **模型规模（公开 README）：** **Qwen3.5-4B**（VLM）+ **1.15B** DiT **flow-matching** 动作解码器（总参数叙事以官方技术报告为准）。
4. **渐进式训练配方：** 大规模 **动作预训练** → **多模态持续预训练** → **SFT** → **RL**，衔接离散 VLM token 与连续动作轨迹。
5. **主张：** 单一通才在多项仿真与真机评测上 **匹配或超越** 按 benchmark 独立微调的 specialist；强调 **OOD 真机泛化**（如 ALOHA 双臂平台对比 GR00T N1.6、π₀.5）。

---

## 公开基准摘录（README 表格，便于 wiki 编译）

### 仿真：操作 + 导航（联合训练、跨平台零 per-benchmark 适配）

| 模型 | LIBERO | RoboCasa-GR1 | Simpler-WidowX | RoboTwin-Easy | RoboTwin-Hard | R2R OS | R2R SR | RxR SR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen-VLA-Base | 90.8 | 40.4 | 64.3 | 64.3 | 66.4 | 61.7 | 53.8 | 55.1 |
| Qwen-VLA-Instruct | **97.9** | **56.7** | **73.7** | **86.1** | **87.2** | **69.0** | **57.5** | **59.6** |

### OOD 泛化（节选）

| 模型 | SimplerEnv-OOD SR (%) | DOMINO SR (%) | DOMINO MS (%) |
| Qwen-VLA-Instruct | **32.0** | **26.6** | **39.5** |

### 真机 ALOHA（与 per-task specialist 对比；Qwen-VLA 为 **统一通才**）

- **域内平均成功率：** Qwen-VLA-aloha（w/ pretrain）**83.6%** vs π₀.5 **71.6%** vs GR00T N1.6 **28.6%**（README 表）。
- **OOD 平均：** Qwen-VLA-aloha（w/ pretrain）**76.9%** vs π₀.5 **41.5%**（README 表）。

---

## 对 wiki 的映射

- 新建 **`wiki/entities/qwen-vla.md`**：通才架构、训练配方、操作–导航统一与 embodiment prompt 的实体页（含 Mermaid 流程图）。
- 更新 `wiki/methods/vla.md`：在 Qwen3 系路线中补充 **Qwen-VLA** 交叉引用。
- 可选更新 `wiki/overview/vla-open-source-repro-landscape-2025.md` 文首说明：2026 起通义 **Qwen-VLA** 作为 **跨操作–导航通才** 新入口（不改动 2025 策展表主体，避免 star 快照失真）。

---

## 外部参考

```bibtex
@misc{qwenvla,
      title={Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments},
      author={Qiuyue Wang and others},
      year={2026},
      eprint={2605.30280},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.30280},
}
```

- [QwenLM/Qwen-VLA（GitHub）](https://github.com/QwenLM/Qwen-VLA)
- [arXiv:2605.30280](https://arxiv.org/abs/2605.30280)
