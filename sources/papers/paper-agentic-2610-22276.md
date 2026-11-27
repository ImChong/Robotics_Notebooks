# MJev-Spatial: System One Spatial Intelligence via Native Attention Matching

- **作者：** Luozheng Qin, Yuhan Zhu, Changhai Zhou, Qian Qiao, Jun Gao, Yuhua Zhou, Harry Zhang, Xiangyu Zeng, Yuxuan Liu, Yuxin Lu, Kaijie Chen, Andrew Chen, Pony Ma
- **机构：** 复旦大学（Fudan University，通讯邮箱 `chzhou25@m.fudan.edu.cn`）；MindLab（通讯邮箱 `andrew@mindlab.ltd`、`pony@mindlab.ltd`）
- **版本 / 载体：** arXiv:2610.22276v2 [cs.CV]（2026-10-15 提交 v1，2026-10-26 修订 v2，37 页，7 图，27 表）
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2610.22276)、[arXiv HTML 全文](https://arxiv.org/html/2610.22276)
- **项目页 / 代码：** **已开源**（代码 + 模型权重；仓库 [Fr0zenCrane/MJev-Spatial](https://github.com/Fr0zenCrane/MJev-Spatial)，Apache-2.0；权重 [mindlab-research/MJev-Spatial-4B](https://huggingface.co/mindlab-research/MJev-Spatial-4B)）— 仓库核查归档见 [`sources/repos/paper-agentic-2610-22276.md`](../repos/paper-agentic-2610-22276.md)
- **核对范围：** 通读论文 Abstract、§1–§5 正文与 Appendix A–H 全部章节，核对 Table 1–27（含 12 基准主评测 Table 2、B200 时延评测 Table 11、读出机制消融 Table 3/12–17、离散化消融 Table 4/23–25、视觉细化与多点解码消融 Table 5/18–22、训练吞吐与显存分析 Table 6/26–27）及仓库 `README.md` 与 `files_sample`
- **通读：** 已完成

## 原文核对笔记

### 1. 研究动机与核心范式（§1–§2）

- **痛点：** 空间智能（空间关系判断、距离/尺寸/角度/计数估计、2D 指向落点）需要达到交互帧率，但现有 VLM（如 SpatialVLM、SpatialRGPT、RoboPoint、RoboRefer）仍走自回归（AR）逐 token 文本生成与正则解析，既把空间答案与冗余格式模板耦合，又在解码与解析上付出高时延代价。
- **核心思路：** 借鉴 Jev 的「对显式声明答案空间直接决策」思路，提出面向空间理解的 **System 1 多模态决策模型 MJev-Spatial**，将**选择题（choice）、数值估计（scalar：长度/计数/角度）、点定位（point）**统一为**有限候选集分类问题**，单次分类前向完成打分，不生成答案文本 token。
- **对 wiki 的映射：** 升格 [`wiki/entities/paper-agentic-2610-22276.md`](../../wiki/entities/paper-agentic-2610-22276.md)，重点拆解 System 1 候选分类范式与自回归 VLM 的本质差异。

### 2. 统一分类框架与粗到精层次设计（§3.1, Appendix A.1, A.4, G）

- **统一概率决策：** 第 $t$ 阶段确定性控制器构造合法候选集 $\mathcal{C}_t=\{c_{t,1},\dots,c_{t,K_t}\}$（每阶段最多 32 个槽位，填充槽置 $-\infty$），模型在候选集上做 softmax 分类并取 $\arg\max$。
- **任务结构（Table 1）：**
  1. **Choice（1 阶段）：** 直接在输入给定选项文本上单轮分类（训练时随机打乱选项顺序，评测保留原基准协议）。
  2. **Numerical（1–2 阶段，分物理量独立码本）：**
     - **长度（m）：** $(0, 16384\text{ m}]$ 按 $s=0.01\text{ m}$ 的 $\log(1+x/s)$ 均匀划分 30 个正粗区间 + `exact-zero` + `overflow` 共 32 根类；正区间内再线性细分 $F_j\in\{4,8,16,32\}$ 格满足 $\frac{e_{j+1}-e_j}{2F_j}\le\max(0.01\text{ m},0.05 e_j)$（前 6 区间各 4 格、后 24 区间各 8 格，合计 216 个正叶格），两阶段均用 $\arg\max$ 并返回精细格中点（Oracle 量化误差上界 $\max(1\text{ cm}, 5\% y)$）。
     - **计数（Count）：** 整数 $n\in[0, 1023]$ 拆成 32 粗组 $\lfloor n/32\rfloor$ 与 32 精细整数 $n\bmod 32$，精确重构 $32a_1+a_2$。
     - **角度（deg）：** $(0, 360^\circ]$ 分 12 个 $30^\circ$ 正区间 + zero/overflow（14 根类），每区间 32 精细格。
  3. **Point（3 阶段视觉细化）：** 每轮做 $3\times 3$ 九宫格分类（固定候选序：top-center 等 9 个方位词）；前两轮选中矩形 $B_t$ 后，从**原始图像像素** $B_0=[0,0,1,1]$ 重新裁剪并经冻结视觉编码器+merger 编码后追加至上下文，最终输出 $B_3$ 中心点（等效 $27\times 27$ 网格，单轴最大正确格中心误差 $1/54$）。
- **无训练多点扩展（Appendix A.4）：** 针对 Point-Bench 计数，在同一权重上展开 3 层条件概率树（完整 91 决策节点 / 729 格；根剪枝丢弃 $p(a_1)<0.01\max_j p(j)$ 的分支，均值降至 51.10 节点），保留全局峰值 10% 以上单元格并做 8 连通域聚类，每连通域输出峰值格中心。

### 3. 模型架构与零新增参数原生 Q/K 读出（§3.2, Appendix A.2–A.3）

- **骨干：** `Qwen3.5-4B`（32 层语言模型，隐层宽度 $D=2560$；L4/L8/…/L32 为 Full Attention，其余 24 层为 Gated DeltaNet；Full Attention 含 $H=16$ 查询头、$G=4$ 键值头、头宽 $d=256$、前 64 维部分多模态 RoPE）。
- **序列格式：** `[\mathrm{state}, I_1, \dots, I_m, \mathrm{question}, q, \mathrm{opt}, c_1, \mathrm{end}, \dots, \mathrm{opt}, c_K, \mathrm{end}, \mathrm{decide}]`，所有特殊标记均复用已有词表词元，不新增 embedding。
- **读出机制：** 运行 L1–L31 后，在 L32 输入端抽取决策位状态 $u_t$ 与各候选结束位状态 $v_{t,i}$；直接复用 L32 预训练输入 RMSNorm $N_{\mathrm{in}}$、头归一化 $N_Q, N_K$、部分多模态 RoPE $R_p$ 及 $W_Q\in\mathbb{R}^{4096\times 2560}$（取 Q+gate 投影的前半 query 维）与 $W_K\in\mathbb{R}^{1024\times 2560}$ 计算 GQA 分组内积并跨 16 个查询头求均值：$z_{t,i}=\frac{1}{16}\sum_{h=0}^{15}\frac{Q_{t,h}^\top K_{t,i,\lfloor h/4\rfloor}}{\sqrt{256}}$。
- **裁剪与参数：** 丢弃 L32 的 V 投影、output gate、输出投影、post-attn norm、SwiGLU FFN（单 FFN 即 70,778,880 参数）与最终 LM norm，保留 4,111,374,336 可训练语言参数（比完整语言侧 KEV 少 95,688,192）；视觉编码器与 merger 保持 BF16 冻结。

### 4. 12.3M 数据管线、联合阶段监督与高效批调度（§3.3, Appendix B–C, H）

- **训练语料（Table 7–8）：** 从 Molmo2-ER 收集的标注转化并清洗出 **12,304,058 条 QA**（Choice 7.1M / 57.65%、Point 3.5M / 28.34%、Length 1.6M / 12.71%、Angle 120K / 0.98%、Count 38K / 0.31%），来源涵盖 RefSpatial（11M）、SenseNova-SI（484K）、VST-P（388K）、VSI-590K-image（255K）、SAT（168K）。执行 2.1M 图像校验、45,820 条重复输入与 3,430 条冲突标签剔除，以及基于 64-bit dHash 的 201 条评测近重复过滤，超过 4,096 token 的样本剔除。
- **单次前向联合监督：** 利用真值路径与真值裁剪图像，把同一问题的多阶段序列按因果顺序串联，通过 Teacher Forcing 在一次语言前向中计算全部阶段交叉熵损失并按阶段数平均（保证 3 阶段点定位与 1 阶段选择题权重相等）。
- **调度优化：** 全局 batch 512（8 卡 × microbatch 8 × 8 步梯度累积，24,032 updates 单遍训练；语言计算 BF16，参数/梯度/Adam/读出 FP32；AdamW 学习率 backbone $5\times 10^{-6}$、readout $2\times 10^{-4}$）。禁止跨 QA 序列打包以防污染 Gated DeltaNet 状态，改用**全局长度分组 + 最长工作量优先跨卡负载均衡**（$C(B)=|B|\max L_q + 0.25\sum V_q$），在 8×L20D 受控 72K 实验中将吞吐从 99.06 QA/s 提至 129.54 QA/s（+30.78%），padding 从 35.17% 降至 1.77%。

### 5. 核心评测、消融与局限（§4–§5, Appendix D–H）

- **12 基准主结果（Table 2 & Table 11）：**
  - 综合均分 **65.64%**（8 模型第一，7/12 基准第一：CV-Bench 88.82、RoboSpatial VQA 79.06、MMSI-Bench 45.60、RefSpatial 58.50、Where2Place 67.96、Point-Bench 64.38、ERQA 46.75），全部 12 个基准均超过未改动的基座 Qwen3.5-4B（均分 54.80%）；VST-300 开发集长度 MAE 为 **0.103 m**。
  - NVIDIA B200（batch=1）平均单问端到端时延 **160.6 ms**，比最快的 AR 基线 Spatial-SSRL-Qwen3VL-4B（397.4 ms）快 **2.47×（2.5×）**，比基座 Qwen3.5-4B（894.1 ms）快 **5.57×（5.6×）**。
- **关键消融结论：**
  1. 原生 Q/K 读出（L32 mean）参数最少、RefSpatial（51.00%）和长度 MAE（0.3341 m）最优，但线性头在 Where2Place / RoboSpatial pointing 上略高；L4–L12 浅层读出几何定位很差，L32+L28 等权融合会恶化长度 MAE；16 个头高度同向冗余（余弦相似度 0.994–0.999）但单头移除仍无一满足近零影响阈值。
  2. 216 格（4/8 精细格）尽管 Oracle 量化 MAE 更高（0.1540 vs 0.0391 m），但学习预测 MAE 更低（0.3202 vs 0.3277 m），且 argmax 优于条件期望（0.3202 vs 0.3300 m）。
  3. 视觉裁剪重编码不可替代：推理去裁剪使 RefSpatial 从 51.00% 跌至 9.00%；仅原图上下文重训只恢复至 46.50%，ViT 特征重放 47.00%，KV 重放 45.50%，直接视觉 token 候选（任务路由）54.63% Point-Bench（低于分层裁剪 56.74%）。
  4. 概率树相比 continue/stop 循环把 Point-Bench 计数成功率从 19.39% 提至 38.78%，根剪枝提速 1.79×（4.716s → 2.638s）且仅损失 0.588% 概率质量。
- **原文/消融暴露的局限：** 依赖预先声明的离散候选集（不支持自由开放式解释或未支持物理量）；多点计数显著弱于部分 AR 模型（72K 试点中 Point-Bench 计数 38.78% vs AR 66.84%，依赖 10% 峰值连通域后处理阈值）；MMSI 多图度量与相机-物体关系仍偏弱；训练期依赖单点/单标量主监督且近重复过滤不能保证物理场景或视频相邻帧级独立性；无真机闭环控制实验。
