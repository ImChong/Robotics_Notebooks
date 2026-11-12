# Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics（arXiv:2606.12365）

> 来源归档（paper）

- **标题：** Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics
- **作者：** Adam Wei, Nicholas Pfaff, Thomas Cohn, Arif Kerem Dayı, Constantinos Daskalakis, Giannis Daras, Russ Tedrake（MIT）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.12365>（v1，cs.RO，2026-06-10）
- **PDF：** <https://arxiv.org/pdf/2606.12365v1>
- **项目页：** <https://ambient-diffusion-policy.github.io/>（含摘要、方法、实验 rollout 与幻灯片链接；未见代码仓库链接）
- **代码：** 项目页与论文均未给出仓库地址（2026-11-12 核查）
- **入库日期：** 2026-11-12
- **一句话说明：** 把次优 / 分布偏移的机器人动作数据只放进扩散训练的高噪声（t > t_min）与可选的低噪声（t < t_max）区间，高质量数据全程使用；只改 Diffusion Policy 的数据采样器。
- **核对范围：** 通读 arXiv v1 PDF 正文第 1–10 节（引言、相关工作、背景、数据性质、理论、方法、受控实验、OXE 规模实验、局限、结论）；附录读了 D（层级与局部性的实验证据）、E（伪代码与实现细节）、F（实验总览，表 5/6）、G.1 与 G.4（迷宫与积木分拣细节）、H（OXE 数据整理与评测）、I（微调）、J（观测空间偏移的初步尝试）；附录 A 的定理证明只读了定理陈述与思路，未逐行验证推导；附录 B/C（VP 与 Ambient 损失推导）、G.2/G.3 的完整扫参图表只浏览。未运行任何代码，未读项目页幻灯片。
- **通读：** 已完成

## 开源状态

- 论文与项目页均未给出代码仓库（2026-11-12 核查），按「未见开源」处理。

## 核心摘录

1. **问题设定（§3.2）：** 小的目标数据集 D_p（目标机器人、环境、任务的专家演示）加大得多的次优 / 偏移数据集 D_q；目标是改进目标分布上的去噪器估计。D_p 与 D_q 的划分由使用者决定。
2. **方法（§6）：** 阶段一为 D_q 样本标注 t_min（可用分类器区分 p_t 与 q_t，阈值 0.5−τ，式 5）与可选 t_max（可扫参）；阶段二训练时 D_p 样本任意 t 可用，D_q 样本仅在 t∈[0,t_max)∪(t_min,T] 可用，推理与 Diffusion Policy 相同。
3. **数据性质（§4）：** 机器人动作数据的功率谱服从幂律（OXE 拟合指数约 1.47，图 3a），论文据此论证「全局到局部」层级与去噪器局部性。
4. **理论（§5）：** 仅在零均值平稳高斯简化模型下证明定理 1（谱幂律 + 低频一致 → 经噪声快速收缩）与定理 2（最优去噪器低噪声下局部性）；论文自述推广到一般分布留待未来。
5. **受控实验（§7，表 1/2/3）：** 2D 迷宫、7 自由度神经运动规划、平面推 T 的 sim-and-target 代理、积木分拣（任务不匹配）。
6. **OXE 规模实验（§8，图 9/10，附录 H）：** 表格清理与搭塔两个真机任务，D_q 取 Magic Soup++（27 个数据集）或自建 COXE（48 个），每策略各 20 次试验。
7. **局限（§9）：** 尚不处理观测空间分布偏移；分类器标注不总优于昂贵扫参；理论限于高斯；比数据过滤更耗算力；需用户划分 D_p/D_q。

## 对 wiki 的映射

- [paper-ambient-diffusion-policy](../../wiki/entities/paper-ambient-diffusion-policy.md)
