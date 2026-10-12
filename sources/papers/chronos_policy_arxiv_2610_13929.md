# Chronos Policy 论文来源归档

## 元数据

- **题目：** Chronos Policy: Learning Robot Policies on Physical Time
- **方法：** Chronos Policy（时间感知 RoPE + 时序随机化）
- **作者：** Changhao Wang, Jaden Clark, Shuran Song
- **单位：** Stanford University
- **预印本：** https://arxiv.org/abs/2610.13929
- **HTML：** https://arxiv.org/html/2610.13929v1
- **版本：** v1，2026-10，cs.RO
- **项目页：** https://changhaowang.github.io/chronospolicy/（归档见 [chronospolicy-github-io.md](../sites/chronospolicy-github-io.md)）
- **代码：** 截至 2026-10-12 未发布（项目页 Code 标 “Coming soon”，论文未给仓库）
- **入库日期：** 2026-10-12
- **读取范围：** arXiv HTML 正文与表格（经 WebFetch 抽取）；附录细节未逐页核对。

## 论文要解决的问题

多数机器人策略把“序列位置”当成物理时间：观测按固定频率堆叠，动作块按固定间隔执行。一旦相机降帧、各传感器速率不同、传输有延迟或推理变慢，同一序列位置对应的真实时间间隔就变了，策略读到的是错位的时序。论文把策略写成 A_{t_a} ~ π(· | O_{t_o}, t_o, t_a)：显式以观测采集时刻 t_o 与动作执行时刻 t_a 为条件，其中动作查询时刻为 t + d_t + jΔt_a（d_t 为起始延迟，Δt_a 为动作间隔）。

## 方法与训练信息

1. **时间感知 RoPE（主设计）：** 观测 token 用采集时刻，动作 token 用被请求的执行时刻，均相对请求时刻；注意力只依赖时间差 t_j − t_i，token 内容不变。
2. **AddTime（消融）：** 对时间戳取 16 个 sin + 16 个 cos 傅里叶特征，经共享 32→512→512 GELU MLP 加到 token 嵌入，动作 token 在每个去噪步重新加入。
3. **时序随机化：** 按模态随机化观测速率与送达延迟，选取每个请求时刻可用的测量并保留其原始时间戳；同时随机化动作起始延迟（仿真中一半 chunk 零延迟，其余最多 100 ms）。
4. **目标与部署：** 标准扩散训练，预测干净动作，ℓ1 损失；异步 chunk 执行，仿真中 d_t 来自注入延迟，真机上由已完成请求估计，过期动作丢弃。
5. **网络：** 16 层扩散 Transformer，宽度 512、8 头；图像编码器为预训练 SigLIP2 base（256×256，768 维 patch 特征）；非图像 token 自注意力，对图像 patch 交叉注意力；输出 16 个 20 Hz 动作（Pivoting 为 19 维自适应柔顺动作：名义位姿、虚拟目标位姿、刚度）。
6. **训练：** 仿真各任务 50 epoch，每策略 3 个种子，每 5 epoch 评一次并取名义成功率最佳检查点；AdamW，lr 3e-4，betas (0.95, 0.999)，权重衰减 1e-6，预训练编码器 0.1× 学习率，余弦调度，BF16；仿真 DDIM 训练 24 步/推理 8 步，真机 50/16 步；随机裁剪与颜色抖动各 0.3 概率。仿真数据集规模、GPU 型号与总算力原文未报告。

## 论文报告的结果

- **仿真任务：** Conveyor（动态抓放，传送带速度随机）、Pivoting（视觉+本体+力矩的接触丰富任务）、6 个 DexMimicGen 任务（Threading、Assembly、Transport、Coffee、Drawer Cleanup、Pouring）。训练时观测速率 12.5–33.33 Hz、延迟 0–50 ms（原始 100 Hz 数据）；Pivoting 力矩 1–100 Hz、延迟 0–100 ms。每方法每条件 450 回合，三个训练种子汇总。
- **Conveyor（成功率 %）：** Diffusion Policy 名义 73.1，推理延迟 0–200 ms 下 1.8，组合扰动 0.0；Chronos（RoPE+随机化）名义 82.0，0–200 ms 下 76.4，宽传感器范围 45.6，组合扰动 32.7。AddTime 名义 85.8 最高，但各扰动条件均低于 RoPE。
- **Pivoting：** DP 名义 61.6，组合扰动 19.6；DP+随机化名义降到 29.8；Chronos 名义 69.3，组合 38.4（标准差 17.8，较大）。
- **消融：** 仅加随机化在 Conveyor 受益、在 Pivoting 受损（跨传感器歧义）；时间戳与随机化互补；扰动条件平均值 Conveyor 48.8%→63.1%，Pivoting 33.3%→55.3%（文中数字）。
- **DexMimicGen：** 名义下与 DP 相当（如 Assembly 75.3 vs 68.7，Coffee 47.3 vs 51.3）；组合扰动下 Coffee 2.7→60.0、Pouring 2.7→77.3、Assembly 0.0→57.3；五种扰动平均 DP 59.7% vs Chronos 73.4%。
- **真机：** 三任务 Dynamic Cup（252 训练/13 验证示范）、Peg Insertion（220/11）、Pivoting（322/17）；基线为异步 Diffusion Policy 与 DP+Real-Time Chunking；打分：成功 1、部分 0.5、失败 0。Dynamic Cup（名义/观测扰动/推理延迟/组合）DP 50/35/40/30，DP+RTC 60/40/50/25，Chronos 80/65/65/55；Peg Insertion Chronos 100/75/75/65；Pivoting Chronos 75/75/77.5/75。每场景仅 10 或 20 次试验。真机型号原文提取中未报告。

## 复现与解释边界

- 代码、权重、数据均未发布；论文与项目页互相矛盾之处：项目页仍标 Paper “Coming soon”，而 arXiv 已有 v1。
- 固定动作频率假设仍在；可变控制频率为后续工作。真机试验次数少，部分仿真标准差大。
- 真机评分含半分，不能与二值成功率直接比较。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-chronos-policy-physical-time.md
- 汇总页：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md
- 周报来源：[wechat_embodied_station_34_papers_spatial_grounding_2026-10-12.md](../blogs/wechat_embodied_station_34_papers_spatial_grounding_2026-10-12.md)

## 一手来源

- <https://arxiv.org/abs/2610.13929>
- <https://arxiv.org/html/2610.13929v1>
- <https://changhaowang.github.io/chronospolicy/>
