# TTCO（arXiv:2610.14896）

> 来源归档（paper；2026-10-16 核验更新，2026-11-06 全文复核）

- **标题：** Test-Time Compute for Humanoid Behavior Foundation Models
- **类型：** paper
- **作者：** Weiji Xie、Junsong Wu、Zibo Zhou、Yueru Chen、Xingyi Wang、Zhenhua Xiong、Weinan Zhang、Chenjia Bai
- **机构：** 中国电信人工智能研究院（TeleAI）、上海交通大学、浙江大学、Gamma Robotics
- **提交日期：** 2026-10-10；v2 更新：2026-10-14
- **arXiv：** <https://arxiv.org/abs/2610.14896>
- **PDF：** <https://arxiv.org/pdf/2610.14896>
- **项目页：** <https://bfmttc.github.io/>
- **官方代码：** <https://github.com/Gamma-Robot/bfm_ttc>
- **版本：** v2（2026-10-14 提交，cs.RO；PDF 内标注日期 2026-10-15）；v1 2026-10-10
- **入库日期：** 2026-10-14；源码核验更新：2026-10-16
- **核对范围：** 2026-11-06 通读 arXiv:2610.14896v2 PDF 全文 37 页：正文 §1–5（含 Limitations、AI use / Reproducibility statement）、参考文献、附录 A（A.1–A.2）、B（B.1–B.4）、C（C.1–C.5.5）、D；核对图 1–8 图注、表 1–25、式 1–44，以及评测协议、超参数与局限。同日复核 arXiv 版本记录与官方仓库 README、`docs/assets.md`、`THIRD_PARTY_NOTICES.md`。
- **通读：** 已完成
- **线索来源：** [具身智能小站 41 篇论文速览](../raw/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)
- **一句话说明：** 在闭环仿真中为冻结的人形行为基础模型搜索输入命令，并利用机器人轨迹生成 BFM 专属动作标签。

## 开源状态

- **已开源（2026-10-16 核验）：** 项目页直接链接 [Gamma-Robot/bfm_ttc](https://github.com/Gamma-Robot/bfm_ttc)，仓库公开提供 TTCO、采样优化器、BFM adapters、仿真运行入口及 surrogate cost model 实验代码。
- **复现条件：** README 要求另行准备 G1 robot assets 与 HEFT-PMG checkpoint；目前未发现项目专属预训练权重或独立数据集下载页。仓库主代码 Apache-2.0，SPIDER sampler 保留 CC BY-NC 4.0 上游许可。
- 归档详见 [项目页核验](../sites/ttco-test-time-compute-humanoid-bfm.md) 与 [源码仓库核验](../repos/bfm_ttc.md)。

## 核心摘录（论文与项目页）

1. **测试时命令优化（TTCO）：** 对目标 motion 采样候选输入，在冻结 BFM 与机器人动力学构成的闭环仿真中评估；选取低代价候选的短段执行，然后重规划。
2. **三种 BFM 的误差改善：** 在 AMASS 的 12,273 条 motion 上评估 HEFT、SONIC、TWIST2。相对将参考 motion 直接作为命令，仿真平均全局身体位置误差分别由 27.1→5.7 cm、61.5→12.7 cm、109.5→17.3 cm，下降 79–84%。
3. **预算配置：** 预算比 K = I·N·H/E（每执行一步消耗的仿真 BFM 转移数）。论文拟合 surrogate cost model 近似计算最优分配：在用于拟合的 10 条 SONIC 动作、6 档预算上平均 regret 4.02%（次优的随机森林 18.44%），K=256 增至 K=8192 时 cost 降低 58.0%；不改参数迁移到 5 条新动作 × 3 个 BFM 时平均 regret 4.60%（随机森林 12.27%），但在 HEFT 上随机森林更低（1.31% vs 8.24%）。
4. **示范重标注：** TTCO 可从记录的机器人轨迹离线生成 BFM 专属命令标签，不需要原 teleoperation body-motion commands；用于训练高层 loco-manipulation policy 与跨 BFM 复用示范。
5. **真机策略评测（表 3）：** 每任务 50 条 SONIC 示范微调 GR00T N1.7-3B，四任务各 60 rollouts。SONIC→SONIC：TTCO 标签 64.2%、原始命令 60.4%、robot-state 2.5%，逐任务有涨有跌；SONIC→HEFT（换执行 BFM）：TTCO 48.8%、原始命令 1.7%、robot-state 12.1%。
6. **HumanoidArena 仿真（表 2、表 25）：** Diffusion Policy、5 任务、3 seeds。同一 BFM 时 TTCO 标签低于原始命令（SONIC 56.7% vs 75.8%；TWIST2 48.0% vs 57.3%），论文称其为原始标签的「部分替代」；跨 BFM 时多数配对优于或接近两条基线，但 TWIST2→HEFT 中 robot-state（31.3%）高于 TTCO（21.4%）。

## 本次全文复核（2026-11-06）

- **代价函数（附录 B.1）：** 跟踪项含关节位置、身体位置与姿态误差（权重 1 / 1 / 0.5，关节速度权重 0）；正则项惩罚解码命令相对参考的形变（0.1 / 0.1 / 0.05）和执行关节速度（0.01）；视野末端加关节与身体位置终端项（各 5）。
- **命令参数化（附录 B.2–B.3、C.1）：** 在名义命令上加偏移，按控制节点（根位置、旋转向量、关节位置，G1 每节点 35 维）线性插值到 H 步；下次规划以上次最优解平移 E 步热启动。比较 CEM、PS、MPPI-CMA、CMA-ES、CBO，CEM 整体有竞争力而被选用。
- **评测设定（§4、附录 C.1–C.2）：** G1 29 关节，物理步 0.005 s、BFM 50 Hz；MuJoCo Warp 做跟踪与优化，loco-manipulation 仿真用 Isaac Sim。AMASS 全量评测用 CEM (I,N,H,E)=(4,80,40,20)，即 K=640。表 1 的 cost 下降 89.0–94.7%，cost 胜率 HEFT 62.27% / SONIC 86.00% / TWIST2 99.67%；关节加速度三者均上升。中位数身体位置误差 13.88→4.80、11.51→4.86、62.68→13.15 cm；误差下降主要来自名义误差最高的四分位（表 4–5）。
- **分配规律（§4.2）：** 固定 O=I·N=8192 时 (1,8192)→(32,256) 使 cost 降 53.42%；H/E 超过 4 后最优 cost 趋平；E 5→10 在 H/E=4 时降 34.43%、在 H/E=10 时升 12.87%。代理模型由 SONIC 10 条动作上的 183 组配置—cost 拟合。
- **标签生成（附录 C.5）：** 用目标 BFM 以记录的机器人动作为参考离线优化，CEM (7,39,37,5)、K=2048，每个候选在 4 个随机化动力学场景取平均。真机为 Unitree G1 + BrainCo Revo2 手 + ZED-M 相机；仅微调 GR00T 动作头，视觉-语言骨干冻结。
- **论文自述局限（§5）：** 实现尚不够快、无法满足实时，因此未做真机闭环命令优化（图 4 真机跟踪为开环执行优化命令）；优化命令可能出现时间不规则并使执行抖动；残余误差仍限制精确接触与全身协调任务。
- **算力（附录 D）：** RTX 3080 Ti 上单次 TTCO 墙钟约 58.1 + 0.009961·C/N 秒（R²=0.890）；§4.1–4.3 约 1,800 / 380 / 1,700 RTX 3080 Ti 等效 GPU 小时，§4.4 约 700 H100 GPU 小时。

## 对 wiki 的映射

- [TTCO 统一论文实体页](../../wiki/entities/paper-ttco-test-time-compute-humanoid-bfm.md)
- [Behavior Foundation Model 概念页](../../wiki/concepts/behavior-foundation-model.md)
- [Loco-Manipulation 任务页](../../wiki/tasks/loco-manipulation.md)

## 参考来源（原始）

- arXiv 摘要及版本记录：<https://arxiv.org/abs/2610.14896>
- 全文 PDF（v2）：<https://arxiv.org/pdf/2610.14896v2>
- [官方项目页](../sites/ttco-test-time-compute-humanoid-bfm.md)
- [官方代码仓库](../repos/bfm_ttc.md)
- 公众号线索：[原始周报](../raw/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)
