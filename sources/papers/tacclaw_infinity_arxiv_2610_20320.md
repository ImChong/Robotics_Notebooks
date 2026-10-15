# TacClaw-Infinity: A Large-Scale Vision-Tactile-Language-Action Dataset for Dexterous Manipulation（arXiv:2610.20320）

> 来源归档（ingest）

- **标题：** TacClaw-Infinity: A Large-Scale Vision-Tactile-Language-Action Dataset for Dexterous Manipulation
- **arXiv：** <https://arxiv.org/abs/2610.20320>（PDF：<https://arxiv.org/pdf/2610.20320>；DOI：<https://doi.org/10.48550/arXiv.2610.20320>）
- **代码 / 工具（官方仓库，已核查）：** <https://github.com/daimon-robotics/TacClaw-Infinity>（robot replay 与运动学评测工具；归档见本页「开源状态」）
- **项目页：** <https://daimon-robotics.github.io/tacclaw-infinity-page/>
- **数据集门户：** Hugging Face <https://huggingface.co/datasets/daimonrobotics/Daimon-Infinity>；ModelScope <https://modelscope.cn/datasets/daimonrobotics/Daimon-Infinity>
- **作者：** Fucai Zhu, Xidong Zhang, Yichi Zhang, Yipan Wang, Fubo Wang, Zibin Wu, Kaiqi Li, Jianhao Jiao, Shihan Lu, Di Hu, Mengdi Xu, Siyu Zhu, Xiaoxiao Long, Yao Yao, Xun Cao, Zhuo Xu, Koichi Hashimoto, Qifeng Chen, Zhe Li, Jianfei Yang, Haofei Lu, Danica Kragic, Shuicheng Yan, Jianghua Duan, Michael Yu Wang, Weihao Yuan
- **机构：** 发布方 Daimon Robotics；项目页列出 16 个单位（Nanjing University、Great Bay University、Tohoku University、SUSTech、HKUST (Guangzhou)、NTU、UCL、Northwest University、Renmin University of China、Tsinghua University、Fudan University、DeepMind、HKUST、KTH、NUS 等，转述自项目页，**未核 PDF**）
- **入库日期：** 2026-10-15
- **一句话说明：** 用无需目标机器人的手持触觉接口 TacClaw 采集同步的视觉、指尖触觉与末端轨迹，经轨迹重建、质量控制、本体感知可执行性筛选与结构化标注形成 VTLA 数据集；论文报告三种机器人上的跨本体迁移。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-15）：** GitHub、ModelScope、Hugging Face 按钮均标注 "Coming soon"，页面称代码与数据 "will be made public"。
- **仓库与门户核查（2026-10-15）：** 仓库 README 称 **数据子集已公开**（HF / ModelScope `Daimon-Infinity`，README 称首批约 1,000 小时）；已发布 `robot_executability_checks/`（Tianji、DM-Sparky、RealMan RM75-B 的回放与运动学评测）；To-do 中 **数据加载与预处理工具、策略训练与任务级评测代码仍未发布**。HF 页面（转述）标注 CC BY-NC-SA 4.0，并称为首批公开部分。
- **不一致说明：** 项目页的 "Coming soon" 与仓库已有工具及 HF 公开子集并存；以仓库与门户实际内容为准。
- **结论：** **部分开源**（数据子集与机器人回放/运动学工具可用；训练与任务级评测代码未发布）。

## 摘录 1：数据与管线（摘要与项目页）

- 采集：TacClaw 手持接口同步记录视觉、指尖触觉与末端轨迹，采集时无需目标机器人。
- 管线：轨迹重建、质量控制、与本体相关的可执行性过滤、结构化标注（接触分段、子任务描述、空间定位）。
- 规模（项目页）：200+ 设备、1,000+ 任务、10,000+ 小时、1 百万+ 回合、1 十亿+ 帧、20+ 机构、50+ 地点。

## 摘录 2：验证

- 摘要称在数据层面做了验证，并在三种机器人上检验无专属示范的跨本体迁移。具体迁移指标 **未核实**（未读到 PDF 实验表）。

## 建议 wiki 动作

- 新建 `wiki/entities/paper-tacclaw-infinity.md`（流程总览 + 源码运行时序图，对齐 Tianji 评测入口）。
- 仅新增实体页，不改已有页。
