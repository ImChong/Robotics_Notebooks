# Stop Wandering: Efficient Vision-Language Navigation via Metacognitive Reasoning

- **作者：** Xueying Li、Feng Lyu\*、Hao Wu\*、Mingliu Liu、Jia-Nan Liu、Guozi Liu
- **机构：** Central South University、Nanjing University、State Grid Hubei Electric Power Research Institute、Dongguan University of Technology
- **版本 / 载体：** arXiv:2604.02318v1 [cs.RO]（2026-04-02；10 页，6 图）
- **原文：** <https://arxiv.org/html/2604.02318>（摘要页 <https://arxiv.org/abs/2604.02318>）
- **项目页 / 代码：** 未确认公开开源代码（论文摘要末尾注明需联系作者邮箱 `xueyingli@csu.deu.cn` 获取代码，正文及元数据均未给出 GitHub 仓库或独立项目主页链接）
- **核对范围：** 已通读 arXiv HTML v1 全文（Abstract、§1–§5、Algorithm 1、Eq. 1–5、Table 1–6、Figure 1–6 说明文字及 References）
- **通读：** 已完成

## 原文核对笔记

1. **核心问题与动机（§1–§2）：** 现有免训练视觉语言导航（VLN）智能体多依赖基于瞬时感知或静态空间记忆的贪心前沿选择（greedy frontier selection），缺少对自身探索进展的监控与策略反思能力，容易出现局部往复震荡（local oscillation）、重复访问已探索区域以及被局部相似干扰物误导，同时每步高频调用 VLM 带来高延迟与高查询开销。
2. **MetaNav 三大模块设计（§3.1–§3.4，Algorithm 1）：**
   - **D1 空间记忆构建（Spatial Memory Construction，§3.2）：** 对每帧 RGB-D 观测 $I^t$ 用开放词汇检测/分割（实现细节中为 YOLOv8x-World + SAM-L）与冻结 VLM 提取物体元组 $(c_j, s_j, p_j)$ 形成局部语义观测 $\mathcal{K}^t$；利用相机内外参通过加权累积 TSDF 融合构建全局体素网格，将空间划分为自由 $\mathcal{V}_{free}^t$、占据 $\mathcal{V}_{occ}^t$ 与未知 $\mathcal{V}_{unk}^t$（Eq. 1），提取边界前沿体素 $\mathcal{F}^t$（Eq. 2），并按空间邻近性与视觉相似性融合更新全局物体级持久语义地图 $\mathcal{M}_{sem}^t$。
   - **D2 历史感知启发式规划（History-aware Heuristic Planning，§3.3）：**
     - 先在 $\mathcal{M}_{sem}^t$ 与近期关键帧中检测目标是否已定位（Known-target Navigation），若定位则直接切换至目标到达策略；
     - 否则聚合前沿区域并按固定重规划间隔 $N_{replan}$ 批量调用 VLM 给出语义得分 $s_{sem}(f_i)\in[0,1]$，结合归一化 2D 几何距离代价 $c_{geo}(f_i)$ 与基于历史轨迹点集 $\{p_\tau\}_{\tau=1}^{t-1}$ 的时间衰减高斯排斥场惩罚 $p_{ep}(f_i)$（Eq. 4），按统一效用函数 $\mathcal{U}(f_i)=\alpha s_{sem}(f_i)-\beta c_{geo}(f_i)-\gamma p_{ep}(f_i)$（Eq. 3）选择 $f^*$，并在接下来 $N_{replan}$ 步内执行底层运动而不重复查询 VLM（主实验 $N_{replan}=3$）。
   - **D3 反思与纠偏（Reflection and Correction，§3.4）：**
     - 双层情景记忆：短期滑动窗口 $\mathcal{B}_{st}$（容量 $K$，主实验 $K=5$）记录 $e_\tau=\langle p_\tau, a_\tau, r_\tau\rangle$，满容后由 LLM 压缩为长期摘要 $\mathcal{S}_{lt}$ 并清空短期缓存；同时维护全局足迹集合 $P_{visited}$ 供 Eq. 4 计算。
     - 基于信息增益的停滞检测：计算未知体积减少量 $g_t=|\mathcal{V}_{unk}^{t-1}|-|\mathcal{V}_{unk}^t|$，若连续 $N_{stag}$ 步满足 $g_\tau < \epsilon_{gain}$ 且距上次反思超过冷却期 $T_{cool}$（Eq. 5），触发 LLM 反思生成含 **Avoid**、**Try**、**Evidence** 三部分的反思摘要 $\mathcal{R}_t$，拼接进后续重规划的 VLM 提示词中。
3. **实验设定与量化结果（§4.1–§4.4，Table 1–6）：**
   - 配置：单回合上限 50 步、$1280\times 1280$ RGB-D（$120^\circ$ FOV、有效深度 $1.7\text{ m}$、相机高 $1.5\text{ m}$、下倾 $30^\circ$、单步最大移动 $1.0\text{ m}$）；YOLOv8x-World + SAM-L 感知；GPT-4o 担任决策 VLM 与反思 LLM，VLM 输入图像降采样至 $360\times 360$；Table 6 统计硬件为 RTX 3080 GPU。
   - **GOAT-Bench Val Unseen（Table 1）：** MetaNav 取得 **71.4% SR / 51.8% SPL**，高于 3D-Mem（69.1% / 48.9%）、Explore-EQA（61.5% / 45.3%）、CG w/ Frontier Snapshots（55.0% / 37.9%）及有监督基线 MTU3D（47.2% / 27.7%）。
   - **HM3D-OVON Val Unseen（Table 2）：** MetaNav 取得 **46.1% SR / 29.8% SPL**，SR 高于 NavFoM（45.2% SR / 31.9% SPL，但 NavFoM 的 SPL 更高）与 Dynam3D（42.7% / 22.4%）、VLFM（35.2% / 19.6%）、TANGO（35.5% / 19.5%）。
   - **A-EQA（Table 3）：** MetaNav 取得 **58.3% LLM-Match / 45.5% LLM-SPL**，高于 3D-Mem（52.6% / 42.0%）与 MTU3D + GPT-4o（51.1% / 42.6%）。
   - **消融与效率（Table 4–6，Fig. 4–5）：** 去掉 D3 反思（w/o Reflection）SR 从 71.4% 降至 66.3%；去掉 D2 统一评分退化为贪心 VLM 降至 64.9%；去掉 D1 空间记忆降至 58.6% SR / 40.0% SPL。单次 VLM 推理平均 5.40 s、LLM 反思与摘要 5.75 s，而点云投影 54.0 ms、记忆融合 359.9 ms、效用打分 2.0 ms；GOAT-Bench 单回合总 VLM/LLM 查询从 3D-Mem 的 31.6 次降至 25.1 次（基础导航 21.3 + 反思 1.9 + 摘要 1.8），降幅 20.7%。
4. **对 wiki 的映射：** [MetaNav / Stop Wandering 论文实体页](../../wiki/entities/paper-agentic-2604-02318.md)
