# ME-Brain-1.0: Memory, Cognition and Action for Evolving Embodied Intelligence

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.24271>（v1：2026-09-21；v2：2026-09-29；本次核对版本 v2）
- **项目页：** <https://machembodied.com/ME-Brain/ME-Brain-1.0.html>
- **代码：** <https://github.com/MachEmbodied/ME-Brain-1.0>（框架占位；动作模型见 [Focus-VLWA](https://github.com/MachEmbodied/Focus-VLWA)）
- **初次入库：** 2026-09-25
- **v2 复核日期：** 2026-10-09
- **作者（arXiv v2）：** Wei He, Hengtao Li, Chenfeng Wang, Zhongrui Yu, Xuhan Zhu, Maokui He, Zide Liu, Xiyue Zhang, Xianwei Mao, Chunpeng Zhou, Jia Shi, Yanze Xin, Jingwen Li, Jingxie Zheng, Sijie Zeng, Fan Lu, Zeyu Zhang, Shuai Guo, Hengxuan Zhang, Pengfei Yu, Jia Shi, Yu Liu, Kun Zhan, Yan Xie
- **机构：** Li Auto Inc.（理想汽车，MachEmbodied）
- **索引来源：** [具身智能研究室 四篇盘点](../blogs/wechat_li_auto_me_brain_vlm_u0_dex_2026-09-25.md)
- **一句话说明：** 可演进记忆 + 认知核心 + 动作模型闭环；经验写入外部记忆而非在线改权重；Piper 六项任务真机 avg 66.7%。

## 核心摘录

1. **Evolvable Memory** 将执行轨迹整理为事件与技能经验，支持跨任务检索；**Cognitive Core** 分解任务、选技能、检查结果并重规划；**Action Model**（Focus-VLWA）在关键交互时刻生成关节/夹爪动作。
2. **自我演进** 更新外部记忆与技能库，**不**随每次任务自动更新模型权重。
3. Piper 双臂真机六项任务各 10 次：**平均成功率 66.7%**（叠碗 10/10，插充电器 1/10 等）；动作模型 RoboMME **47.88%**；RoboDojo 平均 Score **21.51**、SR **16.03%**；Piper 真机 ME-RealBench 平均 SR **66.7%**、Score **69.5**。
4. **论文 v2（2026-09-29）报告的评测：** Cognitive Core 的 ME-VLM 35B-A3B 在 26 项具身基准平均 70.9（+8.2），Agent 基准平均 72.5（+9.6）；Focus-VLWA 在 RoboMME 47.88%、RoboDojo 21.51 Score / 16.03% SR，Piper 六任务平均 66.7% SR / 69.5 Score。
5. 仓库 TODO（2026-09-22，按当前 README 复核）：Focus-VLWA 训练/推理代码 **已发布**；ME-VLM 认知核、完整 ME-Brain 框架、仿真/真机集成 **待发布**。

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-me-brain-1-0.md`](../../wiki/entities/paper-me-brain-1-0.md)
- 技术地图：[`wiki/overview/li-auto-machembodied-4-papers-technology-map.md`](../../wiki/overview/li-auto-machembodied-4-papers-technology-map.md)
