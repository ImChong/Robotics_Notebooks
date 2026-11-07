# RoboDoctor：无需重训的机器人策略递归自我改进（arXiv:2610.35604）

> 来源归档（论文原文已核对；不是二手摘要）

- **英文标题：** RoboDoctor: Training-Free Recursive Self-Improvement for Robot Policies
- **中文标题：** RoboDoctor：无需重训的机器人策略递归自我改进
- **作者：** Heyang Yu、Haomin Li、Sihang Li、Jixin Yan、Shiyun Peng、Wanqi Zhong、Tianyu Xin、Yuan Yao、Xiaoyu Tian、Yiming Li
- **机构：** 清华大学（Tsinghua University）、纽约大学（New York University）、西北大学（Northwestern University）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.35604>
- **原文全文（v1 HTML）：** <https://arxiv.org/html/2610.35604v1>
- **原文 PDF：** <https://arxiv.org/pdf/2610.35604>
- **官方技术报告：** <https://robodoctor-rsi.github.io/tech_report.pdf>
- **项目主页：** <https://robodoctor-rsi.github.io/>
- **官方代码：** <https://github.com/THUSI-Lab/RoboDoctor>
- **权重资源：** <https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/robodojo>
- **版本：** arXiv v1（2026-10-26 提交）
- **入库 / 更新：** 2026-11-07
- **核对范围：** arXiv HTML v1 全文：摘要、§1 引言、§2 相关工作、§3.1–3.5 方法与四个公式、§4.1–4.3 评测及 Tables 1–2、§5.1–5.3 Figure 4 三案例、§6 局限、§7 结论、附录 A 运行协议/经验结构、附录 B 全部 54 条任务 Table 4、附录 C Figure 4 的证据 Table 5；对照项目页和官方仓库 README / 可运行源码。
- **通读：** 已完成
- **一句话说明：** 冻结模型权重，把执行中发现的瓶口偏移等问题化成可证实的局部修正、策略控制分配和可复用外部经验。

## 原文方法笔记

1. **Harness RSI**：3 次 Hybrid Warmup 的有效 native score 均值决定后续 Hybrid（≥0.30）或 Direct（<0.30）；后续 controller 固定，不是每一步重新选路。
2. **Intervention RSI**：WAM 给定 H×14 绝对关节/夹爪动作，论文采用 H=32；Doctor 先看 RGB-D、机器人实测状态与无物理副作用的 FK 预览，放行 1–16 步原前缀或执行 1–5 步矫正，再从当前新状态重新预测。
3. **Operation RSI**：像素和深度用于可见表面定位，但不能直接当作可行接触；需结合末端、夹爪时序、接触面、等待臂；实测接触/物体变化与任务完成需分层核查。
4. **Experience**：诊断、行为保留、修正方法、观测结果与适用条件成对保存在外部经验中。Host 做字段/结构校验，不做语义真值证明；复用新场景必须重算相对位置。
5. **瓶口错位 Figure 4(b)**：横向偏移→Doctor 一步横向矫正，**保持已有抓握**→WAM 续接倾倒与撤离；对齐被验证不等于一定完成倒水。
6. **另两例 Figure 4**：Align Blocks 使用桌面支撑与双爪两点推移；Build Tower 从滑脱抓握改端面夹持并以短抬升检查留存。

## 实验记录与准确读法

- 54 RoboDojo-Sim 任务（Generalization 24、Precision 8、Long-Horizon 8、Memory 6、Open 8），五维等权汇总。
- Warmup（iter0–2）：五维总体 49.74，原生成功率 40.83%，162/162 有效；Refinement（iter3–8）：57.10，49.44%，301/324 有效；绝对变化 +7.37 分、+8.61 pp。
- 54 个任务中 24 改善、20 退化、10 持平；Long-Horizon 54.58→52.92。
- **评估边界**：Warmup 已包含 Agent 介入，不能称为无辅助 WAM；Refinement 结果有候选运行按得分筛选，跨运行 agent/checkpoint/budget 未统一；与公开 leaderboard 对比并非受控配对实验；论文承认跨回合经验的独立因果贡献尚未隔离。
- 所有报告结果为**仿真**；Agent 推理、频繁重观测成本与真机迁移尚无充分评测；有界步数不提供碰撞安全保证。

## 论文与正式源码的参数边界

- 正文/附录描述 32 步提案，host 16 步放行上限。README 的官方 RoboDojo PrismWAM 发布适配器也采用 32 步预测 / 16 步再规划；训练权重仍由独立 Hugging Face 发布。
- 附录 A 记录的某预览 backend 默认可出现 24 步 horizon、四帧历史配置，**不能把此预览设置误认为各次归档运行均采用同一参数**；实验运行版本不完全可控。
- 源码记录见 [RoboDoctor 官方仓库归档](../repos/robodoctor.md)。

## 对 wiki 的映射

- [RoboDoctor 唯一论文/项目实体](../../wiki/entities/paper-robodoctor.md)
- [操作任务](../../wiki/tasks/manipulation.md)
