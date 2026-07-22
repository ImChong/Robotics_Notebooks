# DiT4DiT（dit4dit.github.io）

> 来源归档（ingest）

- **标题：** DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control（项目主页）
- **类型：** project site
- **官方入口：** <https://dit4dit.github.io/>
- **论文：** <https://arxiv.org/abs/2603.10448>
- **代码：** <https://github.com/Mondo-Robotics/DiT4DiT>
- **机构：** Mondo Robotics；HKUST(GZ)；HKUST
- **入库日期：** 2026-06-10
- **再核日期：** 2026-07-22
- **一句话说明：** 官方项目页：级联双 DiT VAM 方法图、LIBERO / RoboCasa-GR1 仿真表、G1 八项桌面 + 三项全身 loco-manip 真机视频、零样本泛化消融、生成视频计划演示、部署效率表（6 Hz / 2.2B）与 BibTeX。同团队后续 **MotionWAM**（arXiv:2606.09215）继承双 DiT 接口，但 **截至 2026-07-22 仍无** 独立项目页/代码；勿将本页误认为 MotionWAM 开源入口。

## 页面结构（公开站点索引）

| 区块 | 内容要点 |
|------|----------|
| Highlights | 级联 video-action 架构；joint dual flow-matching；高数据效率与零样本泛化（约 15% 预训练数据量级叙述） |
| Method | Cosmos-Predict2.5 Video DiT + hook 隐状态 + Action DiT cross-attn；三时间步示意图 |
| Simulation | LIBERO **98.6%**；RoboCasa-GR1 **56.7%** vs GR00T-N1.6 **47.8%** |
| Real-World | G1 桌面八任务 + 全身三项（+SONIC / +decoupled WBC 演示分区） |
| Generalization | 类别替换、物体替换、数量变化零样本 |
| Efficiency | 6 Hz（A100）；对比 GR00T / Qwen3DiT / mimic-video / Cosmos Policy |

## 对 wiki 的映射

- [DiT4DiT 论文实体](../../wiki/entities/paper-dit4dit-video-action-model.md)
- 技术细节以 [sources/papers/dit4dit_arxiv_2603_10448.md](../papers/dit4dit_arxiv_2603_10448.md) 为准
