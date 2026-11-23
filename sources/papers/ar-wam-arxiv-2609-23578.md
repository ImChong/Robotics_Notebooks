# AR-WAM：视觉框 + 原子动作词的 Agent-Ready 世界动作模型

> 一手论文归档；映射到 [唯一论文详情节点](../../wiki/entities/paper-ar-wam.md)。

- **标题：** AR-WAM: A Visual-Conditioned Agent-Ready World Action Model for Robotic Manipulation
- **中文标题：** AR-WAM：面向机器人操作的视觉条件化、Agent 就绪世界动作模型
- **作者：** Yicheng Jiang、Zesen Gan、Xiaobo Wang、Tianlun He、Chenxu Zhao、Minghui Wu、Xinyue Wang、Jiaxu Wang、Junhao He、Jianan Wang、Qiming Shao
- **机构：** 香港科技大学、深圳理工大学、深信服、明略科技、香港中文大学 MMLab、不列颠哥伦比亚大学、星尘智能（Astribot）
- **类型：** paper
- **arXiv：** https://arxiv.org/abs/2609.23578
- **原文版本：** arXiv:2609.23578v2（2026-10-02）
- **HTML：** https://arxiv.org/html/2609.23578
- **PDF：** https://arxiv.org/pdf/2609.23578
- **项目页：** https://ar-wam.github.io/
- **代码：** https://github.com/AR-WAM/AR-WAM
- **数据集：** https://modelscope.cn/datasets/Jasonggg15/AR-WAM-RoboTwin-Dataset
- **入库日期：** 2026-11-22
- **核对范围：** 原文 HTML v2 的 Abstract、§I–VI（相关工作、§III-A–E 方法、Algorithm 1、§IV-A–E 评测/消融、§V 局限）及表 I–VI；官方项目主页 Method / Experiments / Agent-Policy Loop / Open-source links；GitHub README、train_rla.py、train.py、models/model_runner.py、dataloader/rla_hdf5_dataloader.py、configs/stage1_robotwin_adaln.yaml。
- **通读：** 已完成

## 原文贡献与准确接口（§III）

1. **空间与动作分离**：外部 Agent 的 detect 从图像确定归一化目标框 g=(x1,y1,x2,y2)；策略只接收框、可学习原子动作 token s（pick/place/handover/click/pull 等）、参与手臂 u、动作风格 w、当前主相机图像和本体状态。语言理解保留在上游 Agent，策略不含语言编码器。
2. **0.5B 架构**：冻结约 0.3B DINOv3 视觉编码器 + 约 0.2B 共享 Transformer；reasoning expert 预测目标 mask、操作结束时双臂末端位姿、结构化任务摘要；world action expert 联合生成 action chunk（流匹配）和未来视觉/语义的紧凑潜预测。框是输入条件，mask 是辅助输出，不要倒置。
3. **两层前瞻**：离线先以两帧端点 DINO 特征训练 latent IDM/FDM，把中间变化压成 foresight gist；训练 AR-WAM 时用其作为未来视觉监督；未来 reasoning token 作为语义前瞻监督（stop-gradient）。gist extractor 不在策略线上推理链中运行，但策略自身仍生成前瞻潜变量。
4. **Agent 闭环**：detect(y)→g，execute(g,s,u,w)→执行轨迹 ξ，query(ξ)→事件反馈 b。触发事件包括末端位移阈值、夹爪闭合、张开、超时；Agent 历史保留已完成子目标与失败，抓空后重定位并按预算重试，策略不保留跨调用递归记忆。无学习的 homing 由系统级传统轨迹规划处理。

## 实验实测与条件（§IV）

- **RoboTwin 2.0：** 10 个双臂任务，各设置每任务 50 次评估。AR-WAM clean 87.2%、randomized 84.2%、两者均值 85.7%；Fast-WAM clean 87.8%（因此 AR-WAM 不是两个设置都第一）；随机场景 AR-WAM 最优。
- **同设定消融：** 视觉框+操作 token 87.2%；改成文字条件 74.2%，**低 13.0 个百分点**；额外拼接文字 84.6%；无 reasoning 79.4%；无 foresight 78.6%；以未来 patch 代替 gist 76.4%；cross-attention 替 AdaLN 73.4%。指标来自论文表 II/III，不是跨论文外推。
- **RMBench：** 长记忆 M(n) 四任务，AR-WAM 搭配本地 Qwen3.8-27B (AR-Q) 63.5%、Kimi-K3 API (AR-K) 52.5%，最强 memory-augmented 对照 Goal2Skill 38.7%；小物体 M(1) 上低于 Cortex 75.2%。
- **真机 Astribot S1：** 三项长程任务分别是按铃 X 次 70%、拿指定瓶两次 65%、鸡蛋放进指定格子 80%，各任务 20 次试验；平均 **71.7%**，π₀.₅ 35.0%、LiLa-WAM 31.7%。每任务先用 100 条真机专家示范对仿真训练策略微调，不是零真机数据部署。
- **推理延迟：** RTX 4090 24GB / bf16 / 同输入下，AR-WAM 每个 32 步 action chunk 的模型推理 14.09 ms（论文表 VI，9 个 ODE 去噪步），π₀.₅ 62.09 ms；**不是端到端机器人控制回路或 Agent API 延迟**。
- **训练数据：** RoboTwin 2.0 每任务 50 条 clean + 500 条 randomized 示范，RMBench 每任务 50 条；训练混合二者。RoboTwin 2.0 使用状态机+检测器发指令，用于隔离上游 Agent 影响，不应当成真实 Agent 规划成绩。

## 局限与可复现范围（§V 与 README）

1. 原子动作词表为预定义桌面技能，细接触如插线可能需要更复杂拆分；输入框无法提供接触面、姿态或精准 3D 几何。
2. 对上游框检测质量敏感；M(1) 小目标与 Swap T 精准空间任务会失败。训练中做框扰动增强，但不能完全抹平这一误差。
3. 官方截至 2026-11-22 已公开训练入口和 RoboTwin 数据集；README 提供 RLA→Stage1→Stage2 的复现步骤，但未提供可直接复现整套 Agent 真机评测的一键部署程序或 AR-WAM 预训策略权重的独立下载入口（以仓库当前 README 为准）。
4. 官方数据集下载+解压合计需约 1.1 TiB 磁盘；8 GPU 示例、DINOv3 权重需另行获取，不能误写为轻量单卡复现。
5. 论文报告是指定任务/硬件/评估种子下的成功率；没有证明通用开放世界任务、力控/高接触任务或任意本体迁移。

## 对 wiki 的映射

- [AR-WAM 独立详情](../../wiki/entities/paper-ar-wam.md)
- [WAM Stage 3：Joint 未来-动作联合预测](../../roadmap/depth-wam.md)
