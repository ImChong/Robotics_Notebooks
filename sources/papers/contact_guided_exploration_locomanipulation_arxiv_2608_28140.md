# Contact-Guided Exploration for Non-Prehensile Locomanipulation with Multi-Critic RL（arXiv:2608.28140）

> 来源归档：以 arXiv v1 正文、作者项目页和补充材料为主；本次于 2026-10-06 复核。

- **作者：** Simone Tolomei, Mayank Mittal, Franco Angelini, Manolo Garabini, Paolo Salaris, Marco Hutter
- **机构：** Università di Pisa / Centro Piaggio；ETH Zürich；NVIDIA
- **首次提交：** 2026-08-28；**刊物状态：** arXiv 页面标注已被 IEEE Robotics and Automation Letters 接收
- **论文：** https://arxiv.org/abs/2608.28140
- **HTML 正文：** https://arxiv.org/html/2608.28140v1
- **PDF：** https://arxiv.org/pdf/2608.28140
- **项目页：** https://tolomeis.github.io/contact-guided-exp/
- **补充材料：** https://tolomeis.github.io/contact-guided-exp/assets/RAL_Contac_guidance_Supp.pdf
- **代码：** 作者项目页与 arXiv 资源列表未提供本项目专属 GitHub 仓库；Video/Supplementary 有公开链接。
- **对应知识页：** [paper-contact-guided-exploration-locomanipulation.md](../../wiki/entities/paper-contact-guided-exploration-locomanipulation.md)

## 方法摘录

- **问题：** 四足移动操作中的非抓取推/拉，随机探索很少产生有效单边接触；平滑和能耗正则会诱发静止局部最优。
- **接触先验：** grasping proposal 对对象网格生成候选接触点。椅运取 25 个候选点，每回合采样一点并奖励末端接近；箱推使用可见表面的均匀候选点。该先验仅作为探索信号，训练中逐渐衰减。
- **多 Critic PPO：** 任务进度、接触探索、动作正则化组成三个 reward streams；共享 LSTM 特征骨干、独立 value heads；对应优势加权后用于 PPO policy update。
- **权重 schedule：** (w_{task}=0.75)；(w_{exp}:0.1\to0.01)，在 5k–10k step 线性衰减；(w_{reg}:0.15\to0.24)。作者在椅运任务调参，然后不变地复用到其它任务。
- **高层/低层接口：** actor 给出 6 维手臂目标关节角、底座 ((v_x,v_y,\omega_z)) 命令与高度；冻结的预训练 locomotion policy 生成腿部目标。
- **训练设置：** Isaac Lab 4096 环境，physics dt=0.005 s、control dt=0.02 s；修改 RSL-RL PPO；对象质量 2–4 kg、摩擦 0.2–1.5、底座质量 ±5 kg 随机化。椅子训练混合 15 个 IKEA CAD 椅与 100 个程序生成椅。

## 论文结果与口径

- **成功定义：** 物体到目标距离不超过 0.2 m。missed contact 定义为物体位移小于 0.2 m；tipover 标记为物体倾斜超过 35°；并统计 timeout。
- **仿真：** 箱推和椅运主任务的完整方法成功率均超过 90%。结果段给出的最佳值为 94.1% success、4.4% tipover、9.2 s 完成时间。5 个随机种子下，作者方法 success-rate 标准差为 0.98%；固定权重 Multi-Critic、PPO+weight schedule、普通 PPO 分别为 4.2%、1.2%、9.7%。
- **真机：** ALMA 四足移动操作平台在四种未见 IKEA 家具上共 40/58 成功（69.0%）：ADDE 27/37、SANDSBERG 8/14、VIHALS 3/3、LOVBACKEN 2/4。机器人控制使用机载本体感知，物体位姿来自外部 motion capture。
- **额外验证：** 椅子总质量增到 6.5 kg 时成功运输；外力推扰后可以重规划接触并继续。洗碗机仿真中先拉把手、再推门板；相较简单 PPO，手臂关节进入距离位置限位 10% 范围的时间减少 59%。
- **关键消融：** 不含探索奖励成功率为 0%；普通 PPO 椅运 missed-contact 率 9.1%，将权重 schedule 直接施加到标量奖励后降为 4.0%，但方差大；固定权重 Multi-Critic 仍有持续追逐接触点导致的不稳定。单独分值头与衰减机制组合才同时提升接触发现和运输稳定性。

## 复核限制

- 论文主成功率数字来自仿真；真机汇总率只有 69.0%，不可混用。
- 真机物体位姿仍由外部动捕提供。论文指出侧向接近产生的急剧 yaw 命令会损害状态估计/里程计，并造成物体倾倒。
- 作者称需要覆盖更广泛的 domain shifts；固定 schedule 也可能对超参数敏感。
