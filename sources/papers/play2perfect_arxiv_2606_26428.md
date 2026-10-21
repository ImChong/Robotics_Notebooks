# Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly?（arXiv:2606.26428）

> 来源归档

- **标题：** Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly?
- **arXiv：** <https://arxiv.org/abs/2606.26428>
- **PDF：** <https://arxiv.org/pdf/2606.26428>
- **v1 日期：** 2026-06-24（UTC 22:39:12）；后续 v2 2026-07-09、v3 2026-09-08、v4 2026-10-15（当前版本）。本归档的正文数字取自 v4 HTML 全文。
- **分类：** cs.RO（主）；cs.AI；arXiv 摘要页备注 26 pages、19 figures、4 tables；摘要页未标会议
- **会议：** 项目页首段标注 **CoRL 2026**（arXiv 页面未标，以项目页为准，待会议官方页复核）
- **作者与机构（arXiv 摘要页 + v4 HTML 署名）：** Tyler Ga Wei Lum\*、Kushal Kedia\*（\* equal contribution）、C. Karen Liu、Jeannette Bohg（均为 equal advising 之一）；HTML 署名列斯坦福大学与康奈尔大学，但未逐作者标注归属。致谢感谢 Sharpa 的研究合作与技术支持（点名 Kaifeng Zhang 等 Sharpa 团队成员）。
- **项目页：** <https://play2perfect.github.io>
- **代码：** <https://github.com/kushal2000/play2perfect>（项目页 Code 链接；2026-10-16 复核：仓库公开，MIT 许可）
- **检查点：** 由仓库 `download_checkpoints.py` 从 Hugging Face（`Kushal20/play2perfect-checkpoints`）拉取：Stage-1 play 策略 + 4 个微调后的装配策略
- **开源状态：** **已开源**（训练 / 评测 / 最小部署参考 / 检查点；仅 Isaac Sim 版）
- **入库日期：** 2026-10-16
- **索引来源：** [wechat_sharpa_dexterous_manipulation_2026-10-16.md](../blogs/wechat_sharpa_dexterous_manipulation_2026-10-16.md)

## 一句话说明

先在仿真里让一个目标条件 RL 策略用「与任务无关的自由玩耍」在大量程序化物体上学抓取、手内重定向与 6D 位姿到达，再在由 CAD 派生的稀疏奖励装配环境里微调；系统研究了物体多样性、训练目标、轨迹多样性与目标精度四个预训练设计选择，零样本迁移到 Sharpa 手 + KUKA iiwa 14 真机。

## 文章转述（标注：文章转述）

- **文章转述：** 斯坦福 Jeannette Bohg 与 C. Karen Liu 团队基于 Sharpa Wave 的两项工作之一（另一项是 SimToolReal），与「自下而上」的原子技能路线形成呼应。
- **文章转述：** 让灵巧手先在仿真中进行与任务无关的「自由玩耍」式预训练，再微调到精密装配。
- **文章转述：** 样本效率比从零训练高 33 倍。
- **文章转述：** 在 0.5 毫米间隙的紧配合插入任务上取得 60% 的成功率。

## 核心摘录

1. **（arXiv 摘要）核心论点：** 机器人要完善精密装配，先得学会「玩」；Play2Perfect 是在多样物体与目标上做任务无关预训练、再在精密装配上微调的 RL 框架。玩耍的目的是获得可复用的操作先验（抓取、手内重定向、位姿到达），微调则把探索聚焦到最后的高精度接触阶段。
2. **（arXiv 摘要）研究范围与数字：** 系统研究物体多样性、训练目标、轨迹多样性、目标精度；称该先验比从零 RL 训练样本效率高 33 倍（即便给从零训练稠密多阶段奖励）；零样本 sim-to-real 在仅 0.5 mm 接触间隙的紧插入上成功率 60%，在长程多零件装配与拧螺丝上成功率超过 50%。
3. **（项目页）两阶段与发现：** Stage 1 在自由空间用多样物体玩耍；Stage 2 在接触丰富、精密装配任务上微调并零样本迁移；项目页罗列：1000 个物体优于 100 与 10 个，6D 位姿目标优于仅旋转或仅平移，随机轨迹优于固定的 100/10 条，1 cm 目标优于 5 cm 与 10 cm；总结为「预训练在迫使机器人用手指做手内操作而非固定抓取搬运时迁移最好」；项目页未复述 33 倍、0.5 mm/60% 数字，也未点名 Sharpa 手。
4. **（文章转述）三个数字的出处：** 33 倍样本效率、0.5 mm 间隙 60% 成功率，与论文摘要一致（口径见下节）。

## 论文正文要点（标注：论文正文，自报）

- **预训练：** 单一目标条件 RL 策略同时控制臂与手，操作程序化生成的长方体与胶囊体到随机 6D 目标位姿，无示范；默认 1000 个物体；奖励含平滑项、抓取/抬起项、目标项（1 cm 内成功奖励 1000）；算法 SAPG，演员网络为 LSTM(1024) + MLP，评论家使用特权信息。
- **微调：** 稀疏奖励装配环境由 CAD 经 assembly-by-disassembly 构建，每步含最终目标位姿与中间接触目标（如插入前位姿）。
- **观测与动作：** 140 维观测（本体、当前/目标物体关键点、几何），29 维动作（臂为关节增量、手为绝对目标）；仿真用 Isaac Sim，物理 120 Hz、动作 60 Hz；play 预训练 24,576 并行环境、约 7 天、单卡 RTX A6000；每个装配微调 12,288 环境、约 1 天。
- **33 倍的口径：** 出自简化版 Tight-Insertion（Fixtured，T 形件预先靠在夹具上）：Scratch（稠密奖励）需 100 小时以上才接近满成功率，Play2Perfect 约 4 小时达到同水平；在四个主任务上，两种从零训练 24 小时内均无成功 rollout，Play2Perfect 约 2–5 小时解决全部任务。
- **鲁棒性：** 10 N 外力扰动下 Scratch（稠密）成功率约 20%，更大扰动降到 0%；Play2Perfect 在最大扰动下保持 75% 以上。
- **Play-only vs 微调（Fig 7）：** 仿真中 Play-only 在 40 mm 间隙 75%、4 mm 时近 0%；Play2Perfect 在 4 mm 95%、1 mm 92%、0.2 mm 80%。真机：10 mm 间隙 100% vs 60%；2 mm 90% vs 20%；0.5 mm 60% vs 0%。
- **真机平台与结果（Table 1）：** 22-DoF Sharpa 五指手 + 7-DoF KUKA iiwa 14；FoundationPose 追踪物体位姿；Tight-Insertion 10 mm 10/10、2 mm 9/10、0.5 mm 6/10；Assemble-Beam Step 1 16/20、Step 2（条件）12/16；Screw-Leg 插入 7/10、拧紧（条件）5/7；端到端两步梁装配 12/20。Assemble-Beam 与 Screw-Leg 的零件为 3 倍放大版。60 次真机 rollout 共 18 次失败（超时 6、插入时掉落 5、重定向时掉落 3、位姿追踪灾难性失败 4）。
- **日常插入（Fig 8）：** 同一预训练检查点，iPhone 插头与 YCB 叉子插入在仿真中分别微调约 8.4 与 2.8 小时达到 90%（仅仿真）。
- **作者自述局限：** 学的是短时程装配技能而非完整自主装配流水线，任务排序、主动零件选择与目标位姿由外部给定；每个任务或基准族单独微调；真机依赖物体位姿估计，遮挡或快速运动时会失败；策略除目标位姿外不直接观测夹具与周围几何。

## 对 wiki 的映射

[paper-play2perfect](../../wiki/entities/paper-play2perfect.md)
