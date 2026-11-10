# LingBot-Touch — Robbyant 官方项目主页核验

- **官方主页：** <https://technology.robbyant.com/lingbot-touch>
- **英文题名：** Learning Heterogeneous Tactile Representation for General Dexterous Manipulation
- **技术报告：** <https://github.com/Robbyant/lingbot-touch/tree/main/paper>
- **代码：** <https://github.com/Robbyant/lingbot-touch>
- **模型（HF）：** <https://huggingface.co/collections/robbyant/lingbot-touch>
- **模型（ModelScope）：** <https://www.modelscope.cn/collections/Robbyant/LingBot-Touch>
- **核验日期：** 2026-11-10
- **核对范围：** 官网 Overview、Touch at Scale、01 Real-World Manipulation、02 Simulated Reinforcement Learning、03 Scaling、04 Contact Visualization 与顶部 Tech Report/Code/Hugging Face/ModelScope 链接；全页可读文本已核验。
- **通读：** 已完成
- **代码/权重状态：** 已发布推理代码（Apache-2.0）、Tiny / Small 冻结 encoder 权重；不是预训练全套训练源码/数据集发行包。页面中的演示和曲线是项目方报告，非独立复现。

## 官方资料可直接支持的定量事实

| 维度 | 官网页面表述 |
|---|---|
| 触觉语料 | **4,983 h**，**464,431 episodes**，约 **1.33B frames** |
| 压阻 | 2,197 h，86,906 episodes |
| 磁性 | 1,136 h，294,871 episodes |
| 视触觉 | 1,650 h，82,654 episodes |
| Tiny | HaT-Tiny **5.46M parameters** |
| 真机 | 26 项（Sharpa Wave 20、XHand 6），**每任务 20 trials** |
| 仿真 RL | 12 项；网页描述对比 5 种观测配置，**每任务 3 seeds** |

## 技术路径与解读限制

- 原始触觉 signal + taxel geometry 经 canonical geometry/interface 和模态 tokenizer，送入共享 Haptic Transformer；预训练使用 JEPA latent prediction；冻结特征可用来给下游操纵策略后训练提供触觉表征。
- 官网有四类 policy 的对比图、Scaling 曲线、力/接触探针可视化；页面所能提取的文本没有逐项的数值，故不凭曲线自行填写精确均值、误差条。
- 官网称 Zero additional *encoder* training for cross-sensor transfer；这不等于新机器人策略完全无需 post-training，也不表示任何生传感器数据可直接输入。
- 该项目属于 LingBot 系列，但不是 LingBot-VLA 的直接替代；下游 VLA 负责动作预测，本模型输出触觉 tokens。

## 对 Wiki 的映射

- [LingBot-Touch](../../wiki/entities/paper-lingbot-touch.md)
- [技术报告核验边界](../papers/lingbot-touch-technical-report.md)
- [源码与权重核验](../repos/lingbot-touch-official.md)
