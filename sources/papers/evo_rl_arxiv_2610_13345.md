# Evo-RL 论文来源归档

## 元数据

- **题目：** Evo-RL: Iterative Robot Reinforcement Learning from Selective Experience
- **方法：** Evo-RL（人类纠正 + 正优势片段选择的迭代真机强化学习）
- **作者：** Kai Ye, Yifan Jin, Qianhang Luo, Hongru Zhang, Jiedong Zhang, Shuyuan Hu, Song-Yuan Wu, Jihai Xu, Jie Zhao, Bo Zhao（通讯：Bo Zhao）
- **单位：** 上海交通大学人工智能学院（SJTU）；Central South University；South China Normal University；Boden AI（页面未逐人对应）
- **预印本：** https://arxiv.org/abs/2610.13345
- **HTML：** https://arxiv.org/html/2610.13345v1
- **版本：** v1，2026-10，cs.RO
- **代码：** https://github.com/MINT-SJTU/Evo-RL（归档见 [evo-rl.md](../repos/evo-rl.md)）
- **数据集：** https://huggingface.co/datasets/MINT-SJTU/RW-RL-Dataset（归档见 [huggingface-co-mint-sjtu-rw-rl-dataset.md](../sites/huggingface-co-mint-sjtu-rw-rl-dataset.md)）
- **入库日期：** 2026-10-12
- **读取范围：** arXiv HTML 正文与表格（WebFetch 抽取）；附录未逐页核对。

## 论文要解决的问题

真机强化学习常把人类纠正与自主 rollout 混在一起训练，但积累的经验价值不均：即便成功轨迹也含犹豫、重复尝试等低效片段。需要更细粒度地选择经验。

## 方法与训练信息

- **基础策略：** 沿用 π0.5（PaliGemma 视觉语言骨干 + 流匹配动作专家，输出动作块），部署用实时动作块（RTC）；初始策略由遥操作示范以流匹配 MSE 微调得到。
- **人类干预：** 策略进入覆盖不足状态时操作员接管；仅保留“成功干预轨迹”里的人类控制帧（control label=1），训练块在连续人类片段内采样，排除策略控制帧。
- **奖励：** 每个非终止帧 −1；接管前一转移再加惩罚 −C_int；终止奖励 0。
- **价值模型：** 任务专属、仅视觉，SigLIP 编码器 + MLP 头，预测 [−1, 0] 上 201 个 bin 的分类分布，two-hot 交叉熵训练，价值取概率加权均值。
- **优势：** n 步估计（窗口内归一化奖励和 + V(t+n) − V(t)，终止帧 V=0）。
- **迭代回路（Algorithm 1）：** 部署策略收集干预并在 Mix(示范, 干预) 上微调；可选地收集自主 rollout、训练价值模型、计算优势、选择高优势片段并在混合集上再微调；每轮两次 SFT。
- **选择：** 在示范与成功自主 rollout 中取正优势非终止起点的前 10%；重叠或相邻的 20 帧窗口合并；选中的自主片段加入训练混合，选中的示范片段提高采样权重。
- **与 RECAP 区别：** RECAP 用二值优势指示符条件化策略；Evo-RL 用估计优势选择经验并设采样权重。

## 数据集 RW-RL-Dataset（论文口径）

超过 1,000 小时真机交互，四种本体：AgiBot G2、Galaxea R1Lite、AgileX PiperX、SO-101；九类场景、30+ 任务模板；episode 标注为遥操作示范、自主 rollout 或人类干预，帧级有人/策略控制标签与子任务区间。

## 真机任务与结果

- **PiperX 螺钉插入**（5 个螺钉–套筒单元，每单元目标 N=10）；**SO-101 T 恤折叠**（五阶段，每阶段目标 N=20，5 分钟限时）。
- **Table I（匹配预算下干预 vs 示范，ASR % / TP）：** 2 h：SFT 0.00/0.00，SFT+干预 46.15/7.20；4 h：27.03/12.00 vs 51.52/20.40；8 h：43.68/50.17 vs 78.00/53.04；16 h SFT 78.26/53.27。8 h 干预混合约等于 16 h SFT。
- **Table II：** SO-101 折叠 SFT 55.56/75.78 → Evo-RL 76.67/90.00。PiperX 插入 SFT 22.73/12.00；SFT+干预 78.72/47.98；SFT+RL 36.99/33.69；RECAP 57.78/35.54；Evo-RL 88.24/60.61；Evo-RL 第二轮 92.16/67.14。
- **其他：** Evo-RL 88.24% 对 RECAP 57.78%，高 30.46 个百分点（摘要写作“30.46%”）；人工控制时间占比 15.30%（SFT）→9.56%（Evo-RL）；微调约 4 h 52 min（1,511 步）对 RECAP 估计约 64 h（102,111 步），均为 8 块 A800 80GB（RECAP 为估计值）；价值训练 Evo-RL 约 2 h 19 min（3 块 A100）。

## 复现与解释边界

- 无专门局限节。试验量小（N=10/20），两任务两机器人；RECAP 对比仅 PiperX；RECAP 成本为估计；价值学习依赖任务尺度 Z 与 C_int，无敏感性分析。
- 摘要称模型检查点公开，但页面未给检查点链接；仓库 README 标 Model “Coming Soon”。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-evo-rl.md
- 汇总页：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md
- 周报来源：[wechat_embodied_station_34_papers_spatial_grounding_2026-10-12.md](../blogs/wechat_embodied_station_34_papers_spatial_grounding_2026-10-12.md)

## 一手来源

- <https://arxiv.org/abs/2610.13345>
- <https://arxiv.org/html/2610.13345v1>
- <https://github.com/MINT-SJTU/Evo-RL>
- <https://huggingface.co/datasets/MINT-SJTU/RW-RL-Dataset>
