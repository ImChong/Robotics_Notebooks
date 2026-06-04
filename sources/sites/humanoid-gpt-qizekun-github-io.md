# qizekun.github.io/Humanoid-GPT（Humanoid-GPT 项目页）

> 来源归档（ingest）

- **标题：** Humanoid-GPT — CVPR 2026
- **类型：** site / project-page
- **官方入口：** <https://qizekun.github.io/Humanoid-GPT/>
- **入库日期：** 2026-06-04
- **一句话说明：** 论文配套站点：强调 **2B 帧语料 + HME 多样性 + GPT 式因果 Transformer tracker** 同时覆盖 **高动态** 与 **零样本泛化**；含与 **SONIC** 四类并排对比视频、训练外真机舞蹈/居家任务演示、scaling 曲线与 **TensorRT <1.5ms** 延迟表。

## 页面公开信息（检索自 2026-06-04）

| 资源 | URL |
|------|-----|
| 项目首页 | <https://qizekun.github.io/Humanoid-GPT/> |
| arXiv | <https://arxiv.org/abs/2606.03985> |
| 代码 | <https://github.com/GalaxyGeneralRobotics/Humanoid-GPT> |

## 与论文一致的公开主张（便于 wiki 溯源）

1. **核心矛盾**：现有 tracker 难以同时 **in-domain 敏捷** 与 **unseen 泛化**。
2. **数据**：聚合 AMASS / LAFAN1 / Motion-X++ / PHUMA / MotionMillion + in-house；**2B** G1-retargeted frames。
3. **HME**：从运动本身量化多样性，支撑 **~300** motion clusters 与 balanced sampling。
4. **结构**：**Transformer + causal attention**；**expert distillation（DAgger）** 合并数百 RL expert。
5. **对比 SONIC**：四类视频（daily / dance / high-dynamic / balance）左侧 SONIC、右侧 Humanoid-GPT。
6. **对比表（#Frames）**：SONIC **100M** vs Humanoid-GPT **2.0B**；唯一同时 **Transformer + Agile + Zero-shot** 的行（站点 Table）。
7. **Scaling**：数据规模与模型容量曲线；Humanoid-GPT-B @ 2B 在 zero-shot 指标上持续改善。
8. **推理优化**：ONNX + TensorRT，**<1.5ms** @ RTX 4090，约 **5×** TWIST。

## 对 wiki 的映射

- [`wiki/entities/paper-humanoid-gpt.md`](../../wiki/entities/paper-humanoid-gpt.md) — 方法栈、实验、与 SONIC 对照及部署归纳
