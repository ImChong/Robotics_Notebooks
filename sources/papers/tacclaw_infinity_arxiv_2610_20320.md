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
- **核对范围：** [arXiv:2610.20320v2 全文 HTML](https://arxiv.org/html/2610.20320v2)（2026-10-30）；2026-11-16 依次核对 I 引言、II 相关工作及表 I、III 数据采集/质量与可执行性/触觉时空标注、IV 任务与多点采集统计、V-A 至 V-D 实验与表 II–IV、VI 结论、参考文献；论文原始 PDF 未单独阅读，不将项目页或 HF 数据卡算作论文内容。
- **通读：** 已完成
- **一句话说明：** 用无需目标机器人的手持触觉接口 TacClaw 采集同步的视觉、指尖触觉与末端轨迹，经轨迹重建、质量控制、本体感知可执行性筛选与结构化标注形成 VTLA 数据集；论文报告三种机器人上的跨本体迁移。

## 原文核对

- **版本与范围：** [arXiv v2 HTML](https://arxiv.org/html/2610.20320v2)，2026-10-30；核对方法、数据统计、实验、结论和实验表。
- **动作筛选：** TCP 位姿转目标机器人基座后做 IK/FK；位置误差小于 5 mm 且方向误差小于 3° 视为可达。加权雅可比奇异值比低于 0.04 视为低可操纵度。论文明确在后续训练前过滤不可达与低可操纵度轨迹。
- **接触标注：** 双指深度图压缩为 8 维，与视觉、本体状态送入 ASFormer 划分四个接触阶段；Qwen3-VL-Plus 生成子任务文本。接触点投影各视图后提示 SAM3 分割目标。
- **实验边界：** 9 人接触阈值实验；分段测试为 15 条双臂示范的 30 条单臂序列；跨本体每任务 1,600 个训练 episode、50 次评测。添加触觉后 π₀.₅ 总平均 50.0% → 74.0%，StarVLA 46.0% → 72.7%。
- **多点覆盖：** 论文 IV 节只在开放式采集演示中匹配同一时间预算；300 小时条件下多点比单点多 201 个独立动作–物体组合（95% 重采样区间 [60,335]，2000 次）；这是覆盖度，不是下游策略成功率。
- **采样口径：** DM-Tac X 传感器可提供最高 120 Hz 三轴触觉信号，论文 III-B 中的同步多模态 episode 采用 30 Hz；避免混淆传感器原始采样上限与发布帧率。
- **标注质量：** 98.5% 是协议数据中掩码覆盖率而非精度；5% 语言抽查中约 83% 通过。掩码精度与下游影响仍待验证。

## 开源状态（2026-11-05）

- [项目页归档](../sites/tacclaw_infinity.md)与[源码归档](../repos/tacclaw_infinity.md)记录官方入口。数据子集及三种机器人回放和运动学工具公开，策略训练与任务级评测代码仍待发布。
- HF 数据卡称首批 1,000 小时，完整采集规模和公开下载规模不可混同；数据卡注明 CC BY-NC-SA 4.0。

## 2026-11-16 再核对：论文与数据门户的证据边界

- **原论文 v2：** <https://arxiv.org/html/2610.20320v2>（2026-10-30）；核对第 III 节采集与标注、第 IV 节分布式多样性、第 V 节可达性/触觉消融及任务成功率。论文所称 10,000+ 小时、1M+ episodes 为**采集规模**。
- **项目与仓库：** <https://tacclaw-infinity.github.io/> 和 <https://github.com/tacclaw-infinity/TacClaw-Infinity>；公开的可执行工具为三种机器人运动学可执行性评测及回放；策略训练与任务级评测代码仍列待发布。
- **HF 发布门户：** <https://huggingface.co/datasets/daimonrobotics/Daimon-Infinity>；数据卡称首批 1,000 小时，统计图基于约 1,209 小时；分别为**发布说明**与**统计样本**，均不可充当 10,000+ 小时完整公开证明。
- **设备/数据混合：** 该数据卡并列 DM-DataClaw 与外骨骼 DM-DataDex（TacExoGaze），须按 episode 的设备与标定元数据区分，不把 DataDex 全部数据归到 TacClaw 论文验证范围。
- **门户格式检查：** Parquet/视频/元数据目录；门户字段表列 114-D `observation.state` / 111-D `action`、`9930` 占位值掩码、30 fps 同步建议、`uint16` 触觉解码和 CC BY-NC-SA 4.0。不同设备/配置并不保证每维都有效。
- **核对边界：** 本次从 arXiv v2 全文 HTML 逐节核对，PDF 链接存在但当前读取器因文件体积未直接打开；未下载完整数据或重跑论文训练和真机实验。

## 对 wiki 的映射

- 更新已有 [TacClaw-Infinity 实体页](../../wiki/entities/paper-tacclaw-infinity.md)，补充数据管线、筛选阈值、实验条件、局限与源码时序。
