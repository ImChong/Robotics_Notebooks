# ME-Brain-1.0: Memory, Cognition and Action for Evolving Embodied Intelligence

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.24271>
- **项目页：** <https://machembodied.com/ME-Brain/ME-Brain-1.0.html>
- **代码：** <https://github.com/MachEmbodied/ME-Brain-1.0>（框架占位；动作模型见 [Focus-VLWA](https://github.com/MachEmbodied/Focus-VLWA)）
- **入库日期：** 2026-09-25
- **索引来源：** [具身智能研究室 四篇盘点](../blogs/wechat_li_auto_me_brain_vlm_u0_dex_2026-09-25.md)
- **一句话说明：** 可演进记忆 + 认知核心 + 动作模型闭环；经验写入外部记忆而非在线改权重；Piper 六项任务真机 avg 66.7%。

## 核心摘录

1. **Evolvable Memory** 将执行轨迹整理为事件与技能经验，支持跨任务检索；**Cognitive Core** 分解任务、选技能、检查结果并重规划；**Action Model**（Focus-VLWA）在关键交互时刻生成关节/夹爪动作。
2. **自我演进** 更新外部记忆与技能库，**不**随每次任务自动更新模型权重。
3. Piper 双臂真机六项任务各 10 次：**平均成功率 66.7%**（叠碗 10/10，插充电器 1/10 等）；动作模型单独测 RoboMME **47.88%** avg、RoboDojo **16.03%**。
4. 仓库 TODO（2026-09-22）：Focus-VLWA 训练/推理代码 **已发布**；ME-VLM 认知核、完整 ME-Brain 框架、仿真/真机集成 **待发布**。

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-me-brain-1-0.md`](../../wiki/entities/paper-me-brain-1-0.md)
- 技术地图：[`wiki/overview/li-auto-machembodied-4-papers-technology-map.md`](../../wiki/overview/li-auto-machembodied-4-papers-technology-map.md)
