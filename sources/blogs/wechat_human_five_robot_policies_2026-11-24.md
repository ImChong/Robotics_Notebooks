# Robot Policies: 从 Imitation Learning 到 Token VLA

- **类型：** blog / 课程笔记（二手整理）
- **作者：** human five（公众号署名；不等于原讲者）
- **日期：** 2026-11-24 12:38
- **链接：** https://mp.weixin.qq.com/s/wKuYZE3P4yK-EX_v_69uBQ
- **入库日期：** 2026-11-24
- **核对范围：** 用户提供的全文文本（五部分及四组 Q&A）；微信页面抓取失败，原始讲座的讲者、机构、日期、视频及幻灯片尚未定位。另核对下列作者/机构官方入口的对应机制；未逐篇通读所有引用论文或核对原讲座演示。
- **通读：** 未完成

## 内容归纳

1. Physical AI：目标、观测、策略、动作、环境形成反馈回路；区分 state/observation、轨迹概率与累计回报。
2. 动作和控制：关节/任务空间、FK/IK、绝对/相对/增量动作、位置/速度/阻抗/力矩控制；仿真到现实需处理物理、感知和时延差异。
3. IL：BC 的分布偏移、多模态问题、GMM、历史观测、ACT、时间集成、DAgger 和人类接管。
4. RL：rollout、actor-critic、advantage、PPO、AMP、语言模型辅助奖励设计及视频评价。
5. Token VLA：RT-2、Open X-Embodiment、OpenVLA、FAST、OFT、MolmoAct；区分离散 token、连续动作头和分层规划。
6. Q&A：动作块预测长度与执行长度不同；推理吞吐不等于闭环反馈频率。

## 可追溯的一手资料

以下是相关研究的独立原始入口，不是已经定位的讲座出处；链接供逐篇深读。

| 内容 | 论文 / 官方资料 | 核对范围 |
|---|---|---|
| ACT | [论文](https://arxiv.org/abs/2304.13705)、[ALOHA](https://tonyzhaozh.github.io/aloha/)、[代码](https://github.com/tonyzhaozh/act) | 官方项目入口、论文摘要 |
| DAgger | [原论文](https://proceedings.mlr.press/v15/ross11a.html) | 文献身份 |
| HG-DAgger | [原论文](https://arxiv.org/abs/1810.02890) | 文献身份 |
| PPO | [原论文](https://arxiv.org/abs/1707.06347)、[Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html) | 官方算法说明、目标与伪代码入口 |
| AMP | [原始作者论文存档](https://escholarship.org/uc/item/9rh2t6tt) | 文献身份 |
| RT-2 | [原论文](https://arxiv.org/abs/2307.15818)、[DeepMind](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/) | 官方动作 token 及图文/机器人数据联合训练说明 |
| OXE | [DeepMind](https://deepmind.google/blog/scaling-up-learning-across-many-different-robot-types/)、[官方代码与数据入口](https://github.com/google-deepmind/open_x_embodiment) | 官方数据集与 RT-X 模型区别 |
| OpenVLA | [论文](https://arxiv.org/abs/2406.09246)、[项目](https://openvla.github.io/)、[代码](https://github.com/openvla/openvla) | 官方项目架构与开放入口 |
| FAST | [官方 tokenizer](https://huggingface.co/physical-intelligence/fast)、[源码](https://huggingface.co/physical-intelligence/fast/blob/main/processing_action_tokenizer.py) | 官方 tokenizer 与 DCT/BPE 实现线索 |
| OFT | [官方项目](https://openvla-oft.github.io/) | 并行解码、动作块、连续表示、L1 回归 |
| MolmoAct | [论文](https://arxiv.org/abs/2508.07917)、[官方文档](https://docs.allenai.org/models/molmoact)、[代码](https://github.com/allenai/molmoact) | 官方模型文档、论文摘要与空间推理链 |

## 未核实内容

- 讲者使用的 GPT-5 / GPT-5.4 具体版本、Gemini Robotics 2 PPT、实验室多机器人视频、抓取耗时与原始讲座时间戳无直接证据；不作为可复现实验结果。
- 公众号正文中部分数学公式缺失；知识页单独标记标准公式的重建，不能当作讲义原式。
- 本条为跨主题课程笔记，复用既有论文/方法节点；不为各模型再造实体，也不把多个项目代码拼成虚构讲座运行入口。

## 对 wiki 的映射

- [Robot Policies 总览](../../wiki/overview/robot-policies-il-to-token-vla.md)
