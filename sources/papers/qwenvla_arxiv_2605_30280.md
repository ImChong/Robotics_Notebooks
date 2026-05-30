# Qwen-VLA（arXiv:2605.30280）

> 来源归档

- **标题：** Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments
- **类型：** paper（arXiv）
- **链接：** https://arxiv.org/abs/2605.30280
- **代码：** https://github.com/QwenLM/Qwen-VLA
- **入库日期：** 2026-05-30
- **一句话说明：** 提出统一 **视觉–语言–动作** 通才：Qwen3.5-4B VLM + 1.15B DiT flow-matching 动作头，将操作、导航与轨迹预测纳入共享动作–轨迹空间，以 embodiment-aware **文本 prompt** 切换平台而无需 per-embodiment 输出头；渐进训练含动作预训练、多模态 CPT、SFT 与 RL。
- **沉淀到 wiki：** [Qwen-VLA](../../wiki/entities/qwen-vla.md)

---

## 核心贡献（README / 摘要级归纳）

1. **One Generalist Beats Specialists：** 单一模型在多项仿真与真机评测上匹配或超越按 benchmark 独立微调的 specialist。
2. **Unified Action-and-Trajectory Framework：** manipulation、navigation、egocentric action、trajectory prediction 共享预测空间。
3. **Embodiment-Aware Prompt Conditioning：** 多平台共用权重，仅改文本 prompt 描述 embodiment。
4. **Progressive Training Recipe：** 动作预训练 → 多模态持续预训练 → SFT → RL，桥接离散 VLM token 与连续动作轨迹。
5. **Real-World OOD：** 大规模具身预训练支撑未见条件下的真机泛化（README 以 ALOHA 与 GR00T N1.6、π₀.5 对比叙事）。

---

## 对 wiki 的映射

- 主实体页：[qwen-vla.md](../../wiki/entities/qwen-vla.md)（架构、训练、评测与部署语境）
- 交叉更新（本 ingest 已写入正文，不在此表重复链出以免 lint 陈旧预警）：`wiki/methods/vla.md`、`wiki/tasks/vision-language-navigation.md`、`wiki/entities/xiaomi-robotics-0.md`

---

## 摘录要点（维护者可据 PDF 深化）

- **骨干：** Qwen3.5-4B 级视觉–语言模型；**动作：** 1.15B DiT + **flow matching**（与 π₀ / Xiaomi-Robotics-0 同属连续动作生成族）。
- **统一性：** 不显式为每个机器人平台维护独立 policy head；依赖 **prompt 中的 embodiment 描述**。
- **评测广度：** LIBERO、RoboCasa-GR1、Simpler、RoboTwin、R2R、RxR 等；另报 SimplerEnv-OOD、DOMINO 动态操作 OOD。
- **真机：** ALOHA 双臂多任务；强调 **w/ vs w/o 大规模预训练** 对 OOD 的差距。
