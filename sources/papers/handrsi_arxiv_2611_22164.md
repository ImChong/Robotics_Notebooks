# HandRSI：递归自我改进的第一视角三维手部姿态估计（arXiv:2611.22164）

> 来源归档（paper）；入库日期：2026-11-20

- **英文标题：** HandRSI: Recursive Self-Improvement for Hand Pose Estimation
- **中文标题：** HandRSI：通过递归自我改进优化手部姿态估计
- **作者：** Zhekai Wang、Luqing Luo、Wenhao Yao、Junjie Lu、Huanan Zhu、Feifei Gao、Yuhua Jiang
- **机构：** 灵生机器人（LivSyn Robotics）、华南理工大学、悉尼科技大学、北京科技大学、清华大学（论文列示）
- **论文：** https://arxiv.org/abs/2611.22164
- **HTML v1：** https://arxiv.org/html/2611.22164v1
- **PDF：** https://arxiv.org/pdf/2611.22164
- **日期 / 版本：** 2026-11-17，arXiv v1，Technical Report（预印本）
- **代码：** https://github.com/GinKenneth/Hand_RSI （[代码归档](../repos/handrsi.md)）
- **权重：** https://huggingface.co/HandRSI/HandRSI （[模型归档](../models/handrsi-hf-checkpoints.md)）
- **核对范围：** 论文 HTML v1 的 §1–6（动机、相关工作、Loop A/B/C、十指标门控、数据拆分、评估与消融、速度分层、复现与结论）及附录 A/B 的逐轮结果、严格检测口径和定性案例；对照官方 README、推理入口、轮次控制脚本及 HF 模型卡。未在本机下载权重、运行训练或复现测试。
- **通读：** 已完成

## 关键摘录

1. **对象不是 LLM 权重自训练：** 编码 Agent 改进视频手部估计系统的可执行推理 Harness（Loop A），将有效 Harness 对手部存在性和时间一致性的纠正选择性蒸馏回估计网络（Loop B），并由规划 Agent 基于尝试档案引导下一轮（Loop C）。底座为 ACE-Ego-Hand。
2. **十指标验收：** 误差指标与检测指标统一为截断到 ±5% 的相对收益，使用按序列配对 bootstrap 置信下界；若任一指标严重退化则拒绝。**实际阻塞门主要基于 HOT3D val**，ARCTIC 用于评分报告，不作为门控阻塞条件；sealed test 用于最终报告。
3. **结果：** 5 轮后，HOT3D MPJPE-p 12.97 → 8.37 mm、Jitter 2.82 → 1.34；ARCTIC MPJPE-p 15.58 → 13.85 mm。最终网络仍是 θ2（第 3–5 轮未接受新权重），配合 39 模块的 H5* 推理链。
4. **速度层级：** RTX PRO 5000 单卡测得 Lite/Medium/Heavy 约 41.1/34.0/20.7 FPS；完整 39 模块 Harness 仅 16.2 FPS，不能把 Lite 的 FPS 当作完整系统的速度。
5. **边界与人工介入：** HOI4D 零样本上 3D/抖动改善，但 2D EPE、检测指标部分退化；初始轮曾由操作人员放行违反预设 5% 上限的验证指标（HOT3D FAcc -5.6%），首两轮也参考测试数据做过三项流程决策，不能宣传为从始至终完全自动且测试集绝对未参与开发。
6. **开放与依赖：** 代码（MIT）包含 Loop A/B/C、评估器、Harness、部署分层与结果；权重 CC BY-NC 4.0，MANO 等依赖各自许可。需要 ACE-Ego-Hand、视频骨干、相机内参，部分 Harness 模块依赖 WiLoR。

## 对 wiki 的映射

- [HandRSI 独立论文详情节点](../../wiki/entities/paper-handrsi-recursive-self-improvement-hand-pose.md)
- [递归自我改进纵深路线](../../roadmap/depth-rsi.md)
