# Awaken-HI：以全身人类经验扩展人形智能 / Scaling Humanoid Intelligence with Whole-Body Human Experience

> 一手论文阅读归档（不是全文转载）；以 arXiv v1 原文中的方法、表格和试验协议为依据。

- **标题：** Awaken-HI: Scaling Humanoid Intelligence with Whole-Body Human Experience
- **作者 / 机构：** EngineAI（众擎机器人）；原文 §7 列出 Shuiqi Cheng、Boren Jiang、Weilong Ju、Liyun Li、Xuanyi Li 等贡献者
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.26420>
- **原文版本：** arXiv:2611.26420v1，2026-11-19 UTC 提交（北京时间 2026-11-20）
- **全文 HTML：** <https://arxiv.org/html/2611.26420v1>
- **PDF：** <https://arxiv.org/pdf/2611.26420>
- **项目：** <https://engineai-robotics.github.io/Awaken-HI/>
- **官方仓库：** <https://github.com/engineai-robotics/Awaken-HI>（当前仅资料与演示；非运行代码）
- **数据入口：** <https://huggingface.co/datasets/EngineAI-Embodied/Awaken-Human>（仅公开一条样例，不是 50K h 全量）
- **入库日期：** 2026-11-21
- **核对范围：** 通读 arXiv v1 HTML 全文：§1–2 动机及相关工作、§3 数据采集/标注/重定向、§4 Qwen3-VL 与共享 DiT、多监督目标与两阶段训练、§5 全部 Scaling/真机/泛化/消融、§6–7 结论与贡献者，表 1–4；对照官方项目页、README 和 HF 数据卡的发布状态；未对未发布代码做虚构的运行复现
- **通读：** 已完成
- **一句话：** Awaken-Human 用统一可穿戴协议采集全身第一人称经验，Awaken-HI 用共享动作专家同时学习原始人体运动与重定向 T800 动作，最后以机器人遥操作数据适配真实移动操作。

## 原文精读要点

1. **问题与数据（§1–3）：** 只有 egocentric 视频/手部轨迹不足以监督人形全身行走、姿态与抓取协调；采集 PICO 双目 RGB 和 5 个身体追踪器，构建约 50,000 h、17.02M episodes、5.41B 姿态帧、11.18M 有效语言指令、100+ 技能的人类数据。论文的 50K h 是构建语料的规模，不代表 2026-11-21 的 HF 已全量下载。
2. **标注与对齐（§3）：** 图像去畸变、片段语义标注并人工修正、in-house MANO 21 关键点手重建、Fast-FoundationStereo 深度估计，再用 GMR 衍生管线将人体骨架缩放到 T800、校准末端和双手夹爪。相对原 GMR，平均末端位置误差 16.423→2.422 cm、朝向误差 39.590→15.596°（作者全有效帧统计）。
3. **模型与动作（§4）：** Qwen3-VL 产生视觉语言特征；共享状态编码器与 diffusion transformer（DiT）为 T800 action 40D、原始人运动 161D、相机坐标腕姿 18D 三个分支做 flow matching；机器人状态 37D（根朝向 6 + 关节 29 + 夹爪 2）。40D 动作拆成局部根位移 3 + 根朝向 6 + 29 个关节角目标 + 双夹爪 2。
4. **训练与部署（§4）：** 预训练联合三种 motion loss、当前/未来场景深度辅助损失和 subtask 文本监督；然后仅用 T800 action flow-matching loss 对约 150 h 真机遥操作数据做后训练；部署时从当前图像、指令、机器人状态预测全身动作块，滚动重规划，交给底层控制器。不等同于直接预测关节力矩。
5. **缩放（§5.1）：** 5k、12.5k、25k、50k h 嵌套训练子集，统一 100 h held-out 测量，在 equivalent epochs 对齐 checkpoint。归一化动作推理 MSE 0.06649→0.05926（下降约 11%），validation loss 0.04955→0.04025（下降 18.8%）；拟合幂律指数分别 0.051、0.090。固定真机后训练后，三项家务任务平均成功率从 56.7% 上升至 90.0%，不可当作八项主表的成绩。
6. **主评测（§5.2 / 表2）：** EngineAI T800；各方法同用 150 h 遥操作数据、30k steps、global batch 256、每个任务 10 次试验、相同的 10 组初始配置；推理为单张 RTX 5090。八项独立任务平均 π₀.₅ 57.5%、GR00T N1.7 58.75%、Awaken-HI 93.75%；长程工业上架分别 10/30/90%，家务整理 20/10/80%，长程任务中间不得 reset。
7. **泛化（§5.3 / 表3）：** 四种已见物体与目的地的新配对，Awaken-HI 分别 100/100/80/80%，均值 90%；机器人后训练未见的纸巾和药瓶单物体完成率分别 50%、30%，不是整场景泛化 100%。
8. **消融（§5.4 / 表4）：** 所有变体只使用 5k h 人类预训练。只学重定向动作 Base 的三项家务均值 13.3%，单加 Human Motion 43.3%、单加 Depth 40.0%、单加 Wrist 20.0%，Full 53.3%。Wrist 辅助并非每个任务都增益（衣物悬挂 40→30%），说明贡献非简单线性叠加。

## 关键口径与边界

| 论文数字 | 对应实验协议 | 注意 |
|---|---|---|
| 50,000 h | 人类原始语料规模 | HF 当前仅 1 条约 33.8 s 样例 |
| 93.75% | T800 八项独立任务均值，每项 10 次 | 非开放场景跨机器人总体成功率 |
| 90% / 80% | 连续工业上架 / 家务整理，不能中途重置 | 与八项独立任务区别 |
| 56.7→90.0% | Scaling：三项家务，固定后训练机器人数据 | 非八项主实验 |
| 13.3→53.3% | 5k h 消融 Base vs Full，三项家务 | 不应与最终 50k h 主模型混比 |
| 50% / 30% | 未见纸巾 / 药瓶，逐物体计分 | 不能写成完全未见任务的成功率 |

## 开放状态（2026-11-21）

- 官方 GitHub README 的发布清单：技术报告、数据标为已发布；**代码（含推理）、权重、仿真平台仍未发布**，只有 README、docs、preview 等资源；因此无法核实真实源码运行时序或重做真机测试。
- HF Awaken-Human 数据卡当前仅一条录制样例：33.8 秒 / 1,014 帧 / 8 个时序片段，左眼视频 + tracking + calibration + annotation，数据集页面注明 EngineAI Academic Research Dataset License（非商业学术研究、禁止再分发）。这不构成 50k h 全数据开放的证据。
- 论文实验结果均为作者自报，本次未进行权重推理、硬件复现或统计再估计；固定协议下的结论不外推到陌生环境、机型及安全认证。

## 对 wiki 的映射

- [Awaken-HI 独立论文详情](../../wiki/entities/paper-awaken-hi.md)
- [移动操作纵深路线](../../roadmap/depth-loco-manipulation.md)
- [VLA 纵深路线](../../roadmap/depth-vla.md)
