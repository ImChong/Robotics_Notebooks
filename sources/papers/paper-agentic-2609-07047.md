# MEMOBench: A Process Level Memory Benchmark for Robotic Manipulation

- **作者：** Haiyang Sun, Haoxiao Wang, Junming Chen, Weicheng Fang, Zihao Su, Jingkun Yi, Wenyou Yi, Hao Chen, Zhou Zhao
- **机构：** Zhejiang University（浙江大学，Hangzhou, China）
- **版本 / 载体：** arXiv:2609.07047v1 [cs.RO]（Submitted on 7 Sep 2026；仓库说明标注 `[EMNLP 2026]`）
- **原文：** [arXiv HTML 全文](https://arxiv.org/html/2609.07047) · [arXiv 摘要页](https://arxiv.org/abs/2609.07047)
- **项目页 / 代码：** **已开源** — 官方代码仓库 [Collab-Gen/MEMOBench](https://github.com/Collab-Gen/MEMOBench)（`main` 分支，含 `src/eval.py`、`src/get_object_pool.py`、30 个任务的 `src/bddl/` 配置与 checkpoint 元数据及 vendored `src/third_party/LIBERO`）；1,500 条示范数据集（`MEMOBench.zip`，340.56 GiB）与物体/场景资产（`assets.zip`）发布于 [HuggingFace Datasets (SunSeaLucky/MEMOBench)](https://huggingface.co/datasets/SunSeaLucky/MEMOBench)。注：README 克隆示例中的 `https://github.com/SunSeaLucky/MEMOBench` 经核查目前不可访问（404），实际可访问的官方仓为 `Collab-Gen/MEMOBench`；策略训练与记忆对齐训练脚本未包含在该仓内，评测端通过 `openpi_client` WebSocket 对接外部策略服务端。
- **核对范围：** 通读 `arXiv:2609.07047v1` 全文（Abstract、Section 1–6、Limitations、Ethics Statement、References 及 Appendix A–H，含 Table 1–11 与 Figure 1–7 文字说明），并核对官方 GitHub 仓库 `Collab-Gen/MEMOBench` README 与文件树样本。
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [MEMOBench 实体页](../../wiki/entities/paper-agentic-2609-07047.md)

### 1. 研究动机与双层记忆分类学

- 现有记忆操作基准（MemoryBench、MIKASA-Robo、LIBERO-Mem、RoboMME、RMBench、RoboCerebra、RoboMemArena）主要依赖最终任务成功率（SR）或子目标/关键帧标注，将「记忆遗忘/状态过时」与「底层运动执行失败」混为一谈。
- MEMOBench 提出双层分类学：
  1. **任务记忆类型层（Task Type）**：Temporal（`when`，10 任务）、Spatial（`where`，10 任务）、Object（`what`，5 任务）、Procedural（`how`，5 任务）。
  2. **过程记忆操作层（Memory Operation）**：在每个可执行 checkpoint 标注互斥操作标签：
     - **Storage**（$\varnothing \to \mathcal{M}$）：编码新观察到的任务信息（35 模板 / 1,750 实例）。
     - **Update**（$\mathcal{M} \to \mathcal{M}'$）：外部状态改变后修订已有记忆条目，含删除/覆盖（32 模板 / 1,600 实例）。
     - **Compression**（$\mathcal{M} \to \hat{\mathcal{M}}$）：将累积历史信息整合为紧凑表征（17 模板 / 850 实例）。
- **盲标一致性研究（Appendix C, Table 8–9）**：3 名未参与原始标注的实验室内成员对 84 个 checkpoint 模板盲标，Fleiss' $\kappa = 0.799$，三评委完全一致率 $70/84$（83.3%），多数投票与发布标签一致率 $71/84$（84.5%）；主要分歧集中在 Update 与 Compression 边界（8 例）以及 Storage 与 Update 边界（6 例）。

### 2. 任务套件、数据规模与可执行 Checkpoint

- 基于 LIBERO / robosuite / MuJoCo 栈构建，覆盖 30 个历史依赖家居操作任务、29 类物体与 5 类固定装置（fixtures）。
- 每个任务 50 条专家示范，共 1,500 条示范、927,444 个控制步（平均 618.30 步、中位数 489.00 步、P95 1,110.10 步、最长 2,012 步），HDF5 总容量 340.56 GiB；单条轨迹包含动作 $(T, 7)$、第三人称与腕部 RGB $(T, 256, 256, 3)$ 以及状态 $(T, D)$（$D \in \{38, 47, 51, 58, 71, 77, 84, 97, 149\}$）。
- 84 个 checkpoint 模板扩展为 4,200 个示范级 checkpoint 实例；每个 checkpoint 包含 3 级粒度语言（coarse / intermediate / fine-grained）、操作标签（S/U/C）和可由仿真器直接求值的 BDDL predicate（不读取模型隐状态）。
- 定义过程指标 **MSR**（Memory Storage Rate）、**MUR**（Memory Update Rate）、**MCR**（Memory Compression Rate），以及顺序前缀指示器 $\bar{v}_i(c_k) = \prod_{j=1}^k v_i(c_j)$ 防止乱序满足虚高。

### 3. 记忆对齐监督与实验关键数据

- 在 RoboMME $\pi_{0.5}$ 记忆模块（$L=512 = 32 \text{ frames} \times 16 \text{ tokens/frame}, d_m=2048$，冻结 SigLIP 文本编码器缓存 `simple_subgoal` 嵌入 $d_t=1152$）上验证三种仅在训练期使用的正则化损失 $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{action}} + \lambda \mathcal{L}_{\mathrm{align}}$：
  - **Semantic Memory Alignment (SMA)**（$\lambda=0.1$）：掩码池化记忆与文本嵌入余弦距离。
  - **Contrastive Memory Alignment (CMA)**（$\lambda=0.1$）：双向 InfoNCE（初始 $\tau=0.07$，对批内余弦相似度 $>0.99$ 的相同 subgoal 视为正样本）。
  - **Framewise Memory Alignment (FMA)**（$\lambda=0.05$）：逐帧池化特征匹配后续最近邻 `simple_subgoal` 并施加线性近因权重 $\alpha_{b,f} = 0.5 + 0.5\frac{r_{b,f}}{\max(N_b-1,1)}$。
- **主实验（Table 2，1,500 held-out rollouts，replan $k \in \{5, 20, 35, 50\}$）**：
  - 未对齐最佳基线为 $\pi_{0.5}$ + RoboMME Memory Module（$k=5$）：Avg SR **31.9%**（MSR 58.1%、MUR 16.5%、MCR 7.8%）；标准 $\pi_{0.5}$ 在 $k=5$ 时 Avg SR **30.4%**（Object SR 99.6%，但 Temporal 2.2%、Spatial 0.0%、Procedural 19.6%）。
  - MemoryVLA 在 $k=20$ 达到自身峰值 Avg SR **29.2%**，并取得基线中最高的 Avg MUR（**49.9%**）与 Procedural SR（**55.6%**）；HiF-VLA 峰值 Avg SR **21.0%**（$k=5$），在 $k=50$ 时 Avg MCR 达 32.1% 但 Avg SR 仅 3.5%。
  - 应用 CMA 后（$k=5$），$\pi_{0.5}$ + RoboMME 记忆模块 Avg SR 提升至 **34.7%**（Temporal SR $9.8\% \to 14.4\%$，Procedural SR $16.0\% \to 23.6\%$）；跨 4 个 replan 步长的均值中 CMA 与 FMA 均达 29.3%（基线 27.9%），且 FMA 在稀疏重规划（$k=50$）下最稳健（27.3% vs 21.8%）。
  - **Oracle Memory Decomposition（Table 4）**：将真值记忆文本注入提示词后，$\pi_{0.5}$ 在 $k=5$ 的 Avg SR 从 30.4% 升至 51.9%，弥合了 45.5% 的 MUR 缺口与 54.9% 的 MCR 缺口，而仅弥合 20.2% 的 MSR 缺口；但 Procedural SR 反降至 10.0%，说明程序性控制执行仍是瓶颈。
  - **特权监督参考（Appendix G, Table 10）**：MemER（Qwen2.5-VL-7B + $\pi_{0.5}$）使用 checkpoint 关键帧与子目标监督后 Avg SR 达 56.4%，但 Spatial MUR（52.8%）和 Procedural SR（19.6%）仍远未饱和。
- **许可差异核查记录**：论文 Ethics Statement 写道计划以 Apache-2.0 发布代码、CC BY 4.0 发布数据集与标注；GitHub 仓库 `Collab-Gen/MEMOBench` 当前实际 `README.md` 与 `LICENSE` 标注 MEMOBench 采用 `CC BY-NC 4.0`（非商业研究许可），`src/third_party/LIBERO` 保持 MIT 许可。
