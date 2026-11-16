# Improving Robotic Generalist Policies via Flow Reversal Steering（arXiv:2606.13675）

> 来源归档（paper）

- **标题：** Improving Robotic Generalist Policies via Flow Reversal Steering
- **类型：** paper
- **作者：** Andy Tang*（Stanford）、William Chen*（UC Berkeley）、Andrew Wagenmaker（UC Berkeley）、Chelsea Finn（Stanford）、Sergey Levine（UC Berkeley）（* 共同一作）
- **arXiv：** <https://arxiv.org/abs/2606.13675>（v1，2026-06-11，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2606.13675v1>
- **项目页：** <https://flow-reversal-steering.github.io>
- **入库日期：** 2026-11-12
- **一句话说明：** 把「粗糙但合理」的参考动作沿 flow matching 策略的 ODE 反向积分得到噪声，再正向去噪，从而把人或 VLM 的粗略语义指引投射成通才 VLA 先验内的精细动作；并据此提出 DSBC（噪声空间行为克隆）与 DSRL + FRS（用 FRS 轨迹预热噪声空间 RL）。
- **核对范围：** 下载 arXiv v1 PDF 并用 pdftotext 通读正文第 1-6 节（含摘要、图 1-9 与图注，并对图 5、图 8、图 9 看了渲染图）、附录 A-F（相关工作、理想化 FRS 示例、噪声空间分析、人类 / VLM 转向接口与提示、仿真与真机实验细节、超参表 2）；参考文献只扫读，不逐条核对；附录 D.3（动作填充）、表 1（仿真超参）与附录 B/C 的部分子图细节仅浏览，未逐项核对；未运行任何代码，也未找到公开代码仓库。
- **通读：** 已完成

## 开源状态

- 论文首页给出项目页 flow-reversal-steering.github.io；正文与附录中未出现 GitHub 仓库链接，本次核查未找到公开代码仓库（2026-11-12）。

## 核心摘录（均出自论文，带章节）

1. **方法（第 4.1 节）：** flow matching 的去噪是确定性的，把 ODE 反向做欧拉积分 `a_{t-h} ← a_t − v_θ(a_t,t|o)·h` 即得参考动作 a1 对应的噪声 â0；再正向去噪得 â1。有限步数下 â1 只近似重构 a1，偏向 a1 的同时落回通才策略的分布内。主实验用 10 步（h=0.1）。
2. **参考动作来源（第 4.2 节、附录 D）：** 人类用键盘（W/A/S/D/Q/E 三轴平移、夹爪键、交还策略键）或 VLM（Gemini-ER-1.6，输出结构化 JSON：思维链、是否交还基础策略、三轴 +/0/− 方向、幅度 more/less）给出方向性动作；程序化展开为沿该方向直线的动作块。直接执行这些动作无效，但可作为 FRS 的参考。
3. **三种用法（第 4.3 节）：** 零样本在线转向；DSBC（对 FRS 得到的噪声做监督行为克隆，训练辅助噪声策略）；DSRL + FRS（在 DSRL 的 actor 目标上加 BC 辅助项 λ·E[log π]，并用 FRS 轨迹预填回放缓冲）。
4. **仿真设置（第 5.1 节、附录 E）：** LIBERO，基础策略为未在对应切分上训练的 π0.5（LIBERO-90 用 OpenPi 的 π0.5-LIBERO，其余用 Jain 等在 90 上微调的 π0.5）；每任务 50 次评测，Wilson 置信区间；零样本覆盖 Spatial/Object/Goal 全部加 LIBERO-90 中基础策略成功率 ≤40% 的 62 个任务（共 92 任务）。
5. **零样本结果（第 5.2 节）：** 在基础策略成功率 ≤2% 的 42 个 LIBERO 任务中，11 个任务 FRS 带来 ≥10% 的绝对成功率提升；部分加噪与 sample-and-rank 对照分别只改善 4 与 3 个难任务；直接执行 VLM 动作总体反而降低性能。
6. **DSBC（第 5.3 节）：** LIBERO 中平均每任务约 18 条 rollout，训练约 1 GB 显存、不到一分钟；效果追平零样本 VLM 转向，优于同数据上的标准 BC。真机 6 个 DROID 任务每任务 10 条成功的人类 FRS 轨迹，DSBC 平均绝对成功率提升 60%（第 5.5 节），同数据的标准 flow BC 完全失败。
7. **DSRL + FRS（第 5.4 节）：** 15 个 LIBERO-90 任务上预填 20 条 FRS rollout，优于标准 DSRL 与残差 RL（类 PLD）；10 个基础策略近乎全败的任务上只用 1 条 FRS 成功轨迹预热，标准 DSRL 最终约 30%，DSRL + FRS 明显更高（具体数值只在图 7 中，未在正文给出）；对照 Robometer 奖励模型更优。真机毯子（towel）挂杆任务：基础 5% → DSBC 50% → RL 后 80%（附录 F，仅一个任务）。
8. **噪声空间分析（附录 C）：** 分布内动作映射到更小模长的噪声；积分步数增多重构更准但噪声更偏离先验；对 FRS 噪声加 σ=1 或 2 的高斯扰动反而显著提升性能（oracle 实验，图 16）；噪声沿动作块维度平均并重复后性能基本不变，故仿真中噪声策略只预测一个 7D 平均噪声。
9. **oracle 上界（附录 C.3）：** 用在 LIBERO-90 上训练的专家 π0.5（93.9%）当参考动作来源，FRS 在全 LIBERO-90 上达 50%（基础 π0.5-LIBERO 30.8%），挑战子集 33.7%（对比 VLM 转向 11.1%），仍远低于专家本身。
10. **离线 DSBC（第 5.5 节、附录 E.2.2）：** 对 LIBERO-90 全数据集做流反演，平均成功率从 30% 提升到 40%；真机「hang tape」20 条常规遥操作演示，离线 DSBC 优于基础策略与标准 BC（图 9，仅一个任务）。

## 对 wiki 的映射

- [paper-flow-reversal-steering](../../wiki/entities/paper-flow-reversal-steering.md)
