# Immiscible Diffusion Policy 论文来源归档

## 元数据

- **题目：** Immiscible Diffusion Policy: Preserving Multimodal Robot Actions through Label-Free Noise Assignment
- **方法：** Immiscible Diffusion Policy（Immiscible DP）
- **作者：** Xiao Zhang, Yuxin Chen, Zhixuan Liang, Guojian Zhan, Chenran Li, Chenfeng Xu, Masayoshi Tomizuka, Yiheng Li
- **单位：** University of California, Berkeley；Princeton University；The University of Texas at Austin（据 arXiv HTML 标题下作者单位列表；各作者对应关系未核实）
- **预印本：** https://arxiv.org/abs/2610.09369
- **HTML：** https://arxiv.org/html/2610.09369v1
- **版本：** v1，2026-10-07，cs.RO / cs.LG
- **项目页 / 代码：** arXiv 摘要页与 HTML 均未列出项目页、GitHub 或 Hugging Face（核查日期 2026-10-12）。

## 论文要解决的问题

作者观察到：即便示范数据中各模态（例如“从左边抓”和“从右边抓”）数量平衡、mini-batch 内对称，Diffusion Policy 训练出的策略在 rollout 时仍常坍缩到单一模态。论文将原因归结为训练时动作与噪声**独立配对**：不同动作的扩散路径会混合、交叉，去噪网络在交叉区域只能输出平均响应，抑制了模态专属行为。该问题在机器人规划里更严重，因为动作空间稠密且维度低。

## 方法与训练信息

1. 训练期插件，不改策略架构，也不改推理流程；无需模态标签（label-free）。
2. 每个 mini-batch 取 B 个动作块，同时采样 B 个高斯噪声；先按原始独立配对热身，再切换到批内指派。
3. 对展平后的动作块做 mini-batch 逐坐标标准化（均值 μ_B、标准差 σ_B），用匈牙利算法求使总 L2 距离最小的排列 π*，把噪声 ε_π*(i) 分配给动作 a_i。
4. 前向过程 a_t = √ᾱ_t · a + √(1−ᾱ_t) · ε_π*(i)，损失为所匹配噪声与预测噪声之间的 MSE。
5. 激活准则：首个评测检查点任务指标 ≥ 60% 且每种示范模态在 rollout 中频率非零时才打开指派。
6. 基线沿用 Diffusion Policy（状态与 RGB 任务）与 3D Diffusion Policy（点云任务）。批大小 B、匈牙利求解器实现与额外算力开销论文未报告。

## 论文报告的结果

- 仿真任务各 206 条专家示范：Push-T（状态，2 模态）、RoboTwin 2.0 的 Stack Blocks / Place Bread（2 模态）、Place Shoe / Rotate QR Code（4 模态）。真机：Unitree G1 人形的 Fruit-to-Plate 与 Can-to-Bin，各 80 条遥操作示范，左/右手两种模态。
- 两模态仿真任务非主导模态占比提升 6.0×–14.6×：Push-T 2.40% → 35.00%，Stack Blocks 3.33% → 20.00%，Place Bread 4.44% → 33.33%；平均提升 26.05 个百分点（约 8.68×）。
- 任务指标基本保持：Push-T 覆盖率 95.16 → 95.39；Stack Blocks 成功率 96.67 → 96.67；Place Bread 100.00 → 97.78。
- 四模态任务：vanilla 从不生成第 4 种模态，Immiscible 生成了（Place Shoe M(4)=16.03%，Rotate QR Code M(4)=11.10%），其 M(4) 条件成功率分别 85.71 与 40.00；整体成功率有所下降（Place Shoe 100 → 93.33；Rotate QR Code 86.67 → 73.33）。
- 真机：两任务非主导模态由 0 升到 50%，成功率 Fruit-to-Plate 75 → 60，Can-to-Bin 100 → 90（单 seed，各 20 次左右评测）。
- 激活时机消融（Push-T）：第 0 轮激活覆盖率 90.26，第 50 轮 95.39，第 100 轮 95.82，说明过早激活会损害精度。
- 分析：在分离噪声水平下，解析去噪器的模态响应范围 4.29，vanilla 0.24，Immiscible 1.93。

## 复现与解释边界

- 未发布代码或项目页，截至 2026-10-12 无法直接复现；算法本身（批内匈牙利匹配）足够简单可自行实现。
- 真机仅 1 个 seed、每任务 80 条示范，结论是“模态保留”而非“成功率提升”；多数设定下任务成功率略有下降。
- 作者承认被恢复的模态成功率可能偏低；额外训练开销未报告。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-immiscible-diffusion-policy.md
- 方法入口：wiki/methods/diffusion-policy.md
- 任务入口：wiki/tasks/manipulation.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.09369>
- arXiv HTML：<https://arxiv.org/html/2610.09369v1>
