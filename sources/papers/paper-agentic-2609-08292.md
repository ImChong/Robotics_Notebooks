# EvoNav-Bench: Benchmarking Lifelong Navigation in Evolving Environments

- **作者：** Xilin Wang*, Guoxi Zhang*, Hongming Xu, Zhuofan Zhang, Tianxu Wang, Lifeng Fan†
- **机构：** State Key Laboratory of General Artificial Intelligence, BIGAI；Tsinghua University
- **版本 / 载体：** arXiv:2609.08292v2 [cs.RO]（v1: 2026-09-08，v2: 2026-09-14；CC BY 4.0）
- **原文：** [arXiv:2609.08292](https://arxiv.org/abs/2609.08292)（通读版本：[arXiv HTML](https://arxiv.org/html/2609.08292)）
- **项目页 / 代码：** 未确认开源代码（论文原文及 HTML 页面未给出官方项目页、代码仓库或数据集下载链接；文末参考文献 [1] 所列 `https://github.com/matterport/Mask_RCNN` 为基线 UniGoal 感知栈使用的第三方检测库，并非 EvoNav-Bench 官方仓库）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§1 Introduction、§2 Related Work、§3 EvoNav-Bench、§4 Experiments、§5 Conclusion、References 及 Appendix A/B/C，含表 1–4、算法 1 与图 1–6 说明）。
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [EvoNav-Bench: Benchmarking Lifelong Navigation in Evolving Environments](../../wiki/entities/paper-agentic-2609-08292.md)
- **核心问题：** 现有终身导航（LN）基准（如 GOAT-Bench、MultiON、C-Nav、LENL）假设多阶段导航过程中环境完全静态，未能暴露智能体跨阶段复用场景图或视觉快照等持久化记忆时在环境演化下的误导与失效模式。
- **任务与演化生成机制（§3.1–§3.2、Appendix A）：**
  - 基于 ProcTHOR-10k 过滤单房间场景后生成 3–5 阶段连续物体导航任务（训练 7,924 任务 / 31,797 阶段；验证 751 任务 / 3,003 阶段；测试 776 任务 / 3,098 阶段），每阶段提供包含类别、邻接物体、外观描述与 $300\times 300$ RGB 参考图的多模态目标（评测主表使用 `gpt-4o` 生成的 `unified_instruction`）。
  - 阶段内环境静止；阶段切换时以 $p^{\text{evolve}}=0.5$ 触发单房间局部演化，不向智能体通报变化，且屏蔽智能体当前所在过渡房间（`q_current=0`），优先选择参考轨迹已访问房间（$f^{\text{visited}}=3.0$）并抑制连续两阶段修改同一房间（$f^{\text{evolved}}=0.5$）。
  - 三种修改策略：`SALIENT_OBJECT`（0.5，移动大件及其上物体）、`RETAIN_GROUP`（0.4，保留语义组整屋重排）、`FULL_SHUFFLE`（0.1，独立全重排）；经 AI2-THOR 最短路径验证可达性。
  - 全库统计（Fig. 4）：85.9% 回合发生至少一次演化，48.4% 发生 $\ge 2$ 次演化；49.4% 阶段过渡为静态；37.9% 阶段目标在指定前发生过重排（62.1% 未重排）；每回合重排物体比例中位数低于 20%（范围 0% 至 >80%）。
- **评测与主要发现（§4、Appendix B/C）：**
  - 在 10% 测试子集（90 场景、365 阶段）评测，采用阶段级 SR（$r_{\mathrm{succ}}=2.0\,\mathrm{m}$）与 SPL，区分 GT 感知与 Prediction 感知。
  - **表 1 主结果：** GT 下三种已有方法 UniGoal / 3D-Mem / MSGNav 的 SR 为 20.0 / 47.7 / 59.2，均低于三种启发式策略 FU（61.1）/ Stage-Reset（61.9）/ FTU（67.4，SPL 53.3）；Prediction 下感知误差影响凸显，MSGNav（SR 54.8，SPL 38.2）反超 FU（42.2）、Stage-Reset（46.8）与 FTU（49.0）。
  - **表 2 目标重排拆分：** GT 下目标重排时所有方法下降，Stage-Reset 在重排目标优于 FU（60.6 vs 54.1）但在未重排目标劣于 FU（61.4 vs 65.1），体现“重置 vs 复用”权衡；FTU 通过到达后 VLM 核验并删除失效物体条目，在 GT 下重排/未重排目标均取得最高 SR（68.8 / 66.3）。
  - **§4.3 Fixed vs Not Fixed 对照分析：** 在 90 个测试场景强制预建 Stage-2 目标条目后，保留记忆的方法在 Fixed 下优于 Stage-Reset，但在 Not Fixed 下明显下滑（FU 跌破 Stage-Reset），而 Stage-Reset 仅下降 2.2% SR；FTU 因能主动移除失效条目，抗演化退化优于仅做目标核验但不删除旧条目的 MSGNav。
- **原文标注局限（§5）：** 仅模拟物体位置重排，未涵盖物体增删、铰接状态变化及真实人类活动先验；ProcTHOR 视觉真实感有限；所评纠偏策略均为反应式且对预测感知误差敏感。
