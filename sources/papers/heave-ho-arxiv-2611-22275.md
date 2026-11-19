# Heave-Ho（arXiv:2611.22275）原论文归档

> 类型：论文（paper）；基于作者官方 arXiv HTML v2 整理，而非转载摘要。

- **英文标题：** Heave-Ho: Decentralized Long-Horizon Multi-Humanoid Collaboration
- **中文标题：** Heave-Ho：以语音同步实现分散式长时程多人形机器人协作
- **作者：** Ting-Hsuan Chen, Siheng Zhao, Han Jiang, Qi Pan, Louison Lu, Jiaheng Luo, Henghui Bao, Yiyu Chen, Quan Nguyen, Yue Wang
- **机构：** University of Southern California（南加州大学）；VinMotion
- **论文：** https://arxiv.org/abs/2611.22275
- **HTML 全文：** https://arxiv.org/html/2611.22275v2
- **PDF：** https://arxiv.org/pdf/2611.22275
- **版本：** arXiv v2，2026-11-18（最初提交 2026-11-17）
- **项目主页：** https://heaveho-humanoid.github.io/
- **官方仓库：** https://github.com/heaveho-humanoid/Heave-Ho（2026-11-19 仅 README 和图片资源，作者计划 2026-12 月底发布代码）
- **入库：** 2026-11-19
- **核对范围：** 通读官方 arXiv HTML v2 的摘要、§1 引言、§2 相关工作、§3.1–3.4 方法、§4 HH-Bench、§5.1–5.5 实验、§6 结论与局限，附录 A 人类参与者、B 基准任务、C 真机结果、D 训练实现细节、E 逐任务统计、F 补充视频说明；核对正文图注与表格（Table 1–7）及相关引文；同时查看官方项目页和 GitHub README。未下载 PDF 图像逐张验证，未运行作者代码（未发布）。
- **通读：** 已完成

## 核心摘录

1. **问题定义（§1、§3.1）：** 严格的分散式执行：每台机器人仅使用自身本体感受、短期参考运动窗口及物体规格 c=(L,W,H,m)，不读队友状态，任务中机器人之间不传网络消息。合作经共享物体的接触力自然耦合。
2. **高层规划（§3.3、附录 D.1）：** 领导者头部相机图像与任务指令输入经 LoRA 微调的 Qwen3-VL-2B-Instruct，输出团队离散技能（含 hold/done）。训练数据来自各任务 24 条真机操作者示范 episode 或 1,000 条仿真脚本 episode；决策发生于每个技能结束时。
3. **语音同步（§3.3）：** 领导者用扬声器播报技能，队友本地麦克风识别；按共享的粗粒度控制 tick 时间格启动，领导者补偿预期声学延迟，减少识别时差导致的失同步。可由人类替换领导者或跟随指令；“无网络”不等于“无通信”。
4. **技能组合（§3.1–3.2）：** lift/put down/步进/转向通过空手静止 S0 与持物静止 S1 两个 hub pose 连接，固定角色映射 ρ_i 把团队技能变成机器人自身坐标的参考。数据来源包括人类双人动作视频恢复与重定向，以及 IK 程序化参考。
5. **底层训练（§3.2、附录 D.2）：** 冻结 SONIC tracking foundation model，训练共享 actor 的 joint-target 残差；集中式 critic 训练时看全体机器人观测，部署时每台只有局部观测。teacher privileged observations → student distilled + adaptation module → MAPPO 继续闭环训练。物理步长 5 ms、控制步长 20 ms（50 Hz），两 G1、4,096 组并行环境。
6. **HH-Bench（§4、附录 B）：** 12 类协作任务，覆盖双机至 10 机、混合机型/人；例如窄门搬运、L 形通道、交接箱子、四机抬桌、楼梯、轿子、钢筋、消防水带、磨盘。仿真成功率按每任务 3 seeds × 512 teams 统计；真机按每组任务 10 trials。
7. **主要结果（§5、附录 C/E）：** 低层 12 项仿真平均成功率 94.41%，最高对照/消融不超过 28.56%（Table 2）；高层 5 项任务总成功 43/50=86%，零样本 GPT-6 (Astra) 21/50=42%，CHORUS 6/50=12%（Table 1）；高层单次推理 RTX 4090 52.5 ms / 5.3 GiB，GPT-6 API 平均 6.46 s。真机十组任务累计 92/100，其中双 G1 窄门 9/10、L 形通道 10/10、交接箱 10/10；G1 与人抬上楼梯 7/10（Table 4）。
8. **泛化边界（§5.5）：** 两个相同 G1 训练的共享策略直接迁移到 4/8/10 台 G1 仿真；4 G1 或 G1+人等真机组合中四个抬桌设置累计 37/40；PM01 以相同训练 recipe 学自己的控制器，并不是把 G1 关节动作直接跨机型复用。
9. **比较限制（§4、§5.3、§6）：** 仿真起点采样变化范围比真机大，成功率不宜直接横比；规划器与 GPT-6 的比较是受限技能接口的 Qwen 微调 vs 零样本 API，不是相同训练预算。论文明确指出噪声环境下语音可靠性及新任务示范采集费时是主要局限。

## 对 wiki 的映射

- [Heave-Ho 唯一论文实体](../../wiki/entities/paper-heave-ho-2611-22275.md)
- [项目页核对归档](../sites/heave-ho.md)
- [官方仓库核对归档](../repos/heave-ho.md)
