# SANA-Robot：高效混合记忆世界动作模型 | Paper archive

- **类型：** paper
- **论文：** SANA-Robot: Efficient World Action Model with Hybrid Memory
- **作者：** Yuyang Zhao、Zekai Li、Yicheng Pan、Jiaming Tang、Zhuoyang Zhang、Jiaxu Xing、Moritz Reuss、Quanquan Peng、Junsong Chen、Enze Xie、Zhijian Liu、Song Han
- **机构：** 英伟达（NVIDIA）、加州大学圣迭戈分校（UC San Diego）、清华大学（Tsinghua University）、麻省理工学院（MIT）、香港大学（HKU）、Inco AI
- **arXiv：** [2611.28943v1](https://arxiv.org/abs/2611.28943v1) · [正文](https://arxiv.org/html/2611.28943v1) · [PDF](https://arxiv.org/pdf/2611.28943v1)
- **项目：** [官方页面](https://nvlabs.github.io/Sana/Robot/)
- **代码：** [NVlabs/Sana robot 分支](https://github.com/NVlabs/Sana/tree/robot) · [源码归档](../repos/sana-robot.md)
- **核对范围：** arXiv v1 正文 §1–11、图 1–12、表 1–9（方法、训练、部署、真机、记忆及世界模拟器）。按表区分得分与成功率，区分请求时延与控制频率。
- **通读：** 已完成
- **入库日期：** 2026-11-25
- **对应节点：** [SANA-Robot](../../wiki/entities/paper-sana-robot-2611-28943.md)

## 来源摘记

- §2：4.5B 单塔扩散 Transformer；32 层中 24 层 Gated DeltaNet 和 8 层局部 softmax。按视觉/动作 token 的噪声门控统一策略、FDM、IDM。AR 记忆只持续写入真实历史视频，不写当前条件、过去动作或生成结果。
- §3–4：23 个来源中包括真机遥操、UMI、仿真和第一视角视频；过滤后 19.4K 小时、133M 个训练块。基础训练的 Policy/FDM/IDM 比为 8:1:1；AR 用块因果清洁/加噪双流 teacher forcing，再以无 padding 的整回合 packing 微调。
- §5：模型每次产生 32 动作；异步以剩余 d 步覆盖推理延迟，同相位记忆使缓存切片维持训练口径。表 2 的 5090 FP8/NVFP4 为 135.1/115.0 ms，Thor 为 333.6/288.0 ms（4 步去噪、三相机，不含每回合一次的 prompt/sink）。
- §6–8：RoboDojo 官方评测 Memory 52.73 分/52.33% 成功；自跑消融的 AR 48.8% 不等于官方数。RoboTwin 2.0 Clean/Randomized 为 91.8/73.3%；LIBERO-Plus 94.0%，原版 LIBERO-Pro 68.09%，LIBERO-Max Base/Dynamic 88.3/76.1%。
- §7：YAM Ultra/Galaxea R1 每项任务分别微调；三杯记忆 AR 75% 对 base 30%，空中曲棍球异步将停顿 181.3 ms 降至约 4 ms，命中 45%→60%；NVFP4 的该任务为 50%，不能笼统称无损。
- §9：FDM 在 DROID 让外部策略读生成画面并预测下一块动作。21 对任务与策略的平均成功率：模拟 58.3%、真机 52.6%、Ctrl-World 32.4%，不是逐对准确率。WorldArena 66.8（4 步学生 66.2，H100 30 帧 0.3 s）。
