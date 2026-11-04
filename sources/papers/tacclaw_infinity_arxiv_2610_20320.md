# TacClaw-Infinity: A Large-Scale Vision-Tactile-Language-Action Dataset for Dexterous Manipulation（arXiv:2610.20320）

> 来源归档（ingest）

- **标题：** TacClaw-Infinity: A Large-Scale Vision-Tactile-Language-Action Dataset for Dexterous Manipulation
- **arXiv：** <https://arxiv.org/abs/2610.20320>（PDF：<https://arxiv.org/pdf/2610.20320>；DOI：<https://doi.org/10.48550/arXiv.2610.20320>）
- **代码 / 工具（官方仓库，已核查）：** <https://github.com/tacclaw-infinity/TacClaw-Infinity>（robot replay 与运动学评测工具；归档见本页「开源状态」）
- **项目页：** <https://tacclaw-infinity.github.io/>
- **数据集门户：** Hugging Face <https://huggingface.co/datasets/daimonrobotics/Daimon-Infinity>；ModelScope <https://modelscope.cn/datasets/daimonrobotics/Daimon-Infinity>
- **作者：** Fucai Zhu, Xidong Zhang, Yichi Zhang, Yipan Wang, Fubo Wang, Zibin Wu, Kaiqi Li, Jianhao Jiao, Shihan Lu, Di Hu, Mengdi Xu, Siyu Zhu, Xiaoxiao Long, Yao Yao, Xun Cao, Zhuo Xu, Koichi Hashimoto, Qifeng Chen, Zhe Li, Jianfei Yang, Haofei Lu, Danica Kragic, Shuicheng Yan, Jianghua Duan, Michael Yu Wang, Weihao Yuan
- **机构：** 发布方 Daimon Robotics；项目页列出 16 个单位（Nanjing University、Great Bay University、Tohoku University、SUSTech、HKUST (Guangzhou)、NTU、UCL、Northwest University、Renmin University of China、Tsinghua University、Fudan University、DeepMind、HKUST、KTH、NUS 等，转述自项目页，**未核 PDF**）
- **入库日期：** 2026-10-15
- **一句话说明：** 用无需目标机器人的手持触觉接口 TacClaw 采集同步的视觉、指尖触觉与末端轨迹，经轨迹重建、质量控制、本体感知可执行性筛选与结构化标注形成 VTLA 数据集；论文报告三种机器人上的跨本体迁移。

## 原文核对

- **版本与范围：** [arXiv v2 HTML](https://arxiv.org/html/2610.20320v2)，2026-10-30；核对方法、数据统计、实验、结论和实验表。
- **动作筛选：** TCP 位姿转目标机器人基座后做 IK/FK；位置误差小于 5 mm 且方向误差小于 3° 视为可达。加权雅可比奇异值比低于 0.04 视为低可操纵度。
- **接触标注：** 双指深度图压缩为 8 维，与视觉、本体状态送入 ASFormer 划分四个接触阶段；Qwen3-VL-Plus 生成子任务文本。接触点投影各视图后提示 SAM3 分割目标。
- **实验边界：** 9 人接触阈值实验；分段测试为 15 条双臂示范的 30 条单臂序列；跨本体每任务 1,600 个训练 episode、50 次评测。添加触觉后 π₀.₅ 总平均 50.0% → 74.0%，StarVLA 46.0% → 72.7%。
- **标注质量：** 98.5% 是协议数据中掩码覆盖率而非精度；5% 语言抽查中约 83% 通过。掩码精度与下游影响仍待验证。

## 开源状态（2026-11-05）

- [项目页归档](../sites/tacclaw_infinity.md)与[源码归档](../repos/tacclaw_infinity.md)记录官方入口。数据子集及三种机器人回放和运动学工具公开，策略训练与任务级评测代码仍待发布。
- HF 数据卡称首批 1,000 小时，完整采集规模和公开下载规模不可混同；数据卡注明 CC BY-NC-SA 4.0。

## 对 wiki 的映射

- 更新已有 [TacClaw-Infinity 实体页](../../wiki/entities/paper-tacclaw-infinity.md)，补充数据管线、筛选阈值、实验条件、局限与源码时序。
