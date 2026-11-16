# arXiv:2607.29613 — World Critic Model (WCM)

> 原始论文归档（完整核对版）

- **标题：** World Critic Model
- **作者：** Senyu Fei\*, Zihao Wu\*, Yujie Wang, Siyu Lin, Haohong Chen, Chengming Xu, Yanwei Fu, Hang Xu, Jieji Qiu, Changsheng Lu, Yujia Zhang†（同济大学 Tongji University、上海创智学院 Shanghai Innovation Institute、复旦大学 Fudan University）
- **链接：** <https://arxiv.org/abs/2607.29613>（v1，2026-07-31；核对 `2607.29613.txt` 与 `2607.29613.html`）
- **项目页：** <https://sylvestf.github.io/wcm-homepage/>
- **代码仓库：** <https://github.com/sylvestf/WCM>（MIT 协议；WCM Critic 数据/训练/评测/视频渲染已开源，全量 RL 策略微调管线与剩余权重逐步发布）
- **模型/数据集合：** <https://huggingface.co/collections/Sylvest/wcm>（已发布 `Sylvest/pick-place-wcm`、`Sylvest/pick-place-wcm-ckpt`、`Sylvest/libero_plus_lerobot`）
- **通读：** 已完成
- **核对范围：** 已通读 `2607.29613.txt` 与 `2607.29613.html` 全文（Sec. 1–6、Eq. 1–10、Fig. 1–6、Table 1–4 及 Appendix A–F）；并核对 `sylvestf/WCM` 源码与 `sylvestf.github.io/wcm-homepage/`。

---

## 核心问题与洞察（Sec. 1–2）

现有 VLA 强化学习后训练的 Critic 普遍存在两类结构性错配：
1. **无记忆单帧估计 vs 部分可观测控制（POMDP）：** 多数 PPO/Value-based Critic 仅根据单帧视觉观测估计 $V(o_t)$，无法区分视觉静态近似状态间的阶段进度（如“刚靠近”与“刚松手回退”），且难以在未见过背景下保持稳定表征。
2. **仅拟合标量回报导致表征坍缩：** 在长程稀疏奖励（$r_t \in \{-1, 0, -C_{\text{fail}}\}$）下，若仅靠单步/蒙特卡洛回报 MSE 监督 Critic，隐藏特征会迅速退化为只记住训练集背景与任务 ID 的标量映射器，一旦遇到光照/布局/物体 OOD 偏移便出现价值虚高或失效。

**WCM 核心设计（Eq. 1–10）：** 将 Critic 重构为**带因果历史记忆的轻量 LeJEPA 式联合世界建模评论员**（约 107.2M 可学习参数）：
1. 视觉帧经可训练 ViT 编码器压为隐状态序列 $z_{t-K+1:t} = \text{enc}_\varepsilon(o_{t-K+1:t})$（多视角经 `view_attention` 池化），与冻结 CLIP 文本 token $u_\ell = A_{\text{lang}}(\text{CLIP}(\ell))$ 做交叉注意力融合后送入 Action-Free Causal Transformer 骨干得到上下文特征 $h_t$。
2. **结构级动作隔离：** 价值头 $V_\theta(h_t)$ **在计算图消费任何动作 $a_t$ 之前**输出标量预测，严防模型走捷径通过“记住训练动作分布”作弊估分。
3. **动作条件门控潜动力学：** 仅在价值分支之后，将 $h_t$、当步动作 $a_t$ 与当前隐表征 $z_t$ 送入 FiLM 门控残差动力学头 $\hat{z}_{t+1} = D_{\text{world}}(h_t, a_t, z_t) = z_t + \Delta(h_t, a_t)$，在隐空间预测下一时刻表征（目标取 $\operatorname{sg}(z_{t+1})$），联合可选的 `SIGReg` 各向同性高斯正则化避免表征坍缩（Eq. 10：$\mathcal{L} = \mathcal{L}_{\text{value}} + \lambda \mathcal{L}_{\text{pred}} + \eta \mathcal{L}_{\text{SIGReg}}$；详见 Appendix D.1 对在线 RL 停用 VLM 潜空间 SIGReg 的计算开销说明）。

---

## 奖励设计与全量基准定量结果（Sec. 3.3 & Sec. 4, Tables 1–4）

### 回报构造（Eq. 8）
- 成功终止帧 $r_t = 0$，失败终止帧 $r_t = -C_{\text{fail}}$（真机实验中 $C_{\text{fail}} = 300$），中间过程每步 $r_t = -1$；对累积 Monte Carlo 回报采用全局 Min-Max 归一化到 $[-1, 1]$ 后用 MSE 监督 $\mathcal{L}_{\text{value}}$。

### 1. ManiSkill Pick-and-Place（25 任务，IND 与 3 维 OOD — Table 1）
- **OpenVLA-OFT：** SFT (`IND 28.1 / OOD avg 18.3`) → +GRPO (`94.1 / 60.6`) → +PPO (`97.7 / 77.1`) → **+WCM (`99.0±0.4 / 77.9±0.8`)**；甚至从未微调的 Zero-Shot 底座 (`IND 0.8 / OOD avg 0.8`) 出发，+WCM 可直接升至 **`98.7±0.3 / 73.5±1.8`**。
- **$\pi_0$：** SFT (`38.4 / 18.1`) → +FlowSDE (`78.8 / 39.3`) → +FlowNoise (`77.8 / 36.9`) → +$\pi$-stepNFT (`79.2 / 50.4`) → **+WCM (`84.4±1.2 / 51.5±1.5`)**。
- **$\pi_{0.5}$：** SFT (`47.0 / 26.4`) → +FlowSDE (`90.9 / 49.3`) → +FlowNoise (`89.7 / 53.4`) → +$\pi$-stepNFT (`85.4 / 59.5`) → **+WCM (`91.9±0.4 / 64.4±1.4`)**（vision 78.1、semantic 58.5、execution 56.5）。

### 2. LIBERO-Plus 极低数据量测试（One-SFT = 每任务仅 1 条演示，Table 2）
- **$\pi_0$：** Full-SFT (`71.2±1.5`) / One-SFT (`39.1±2.1`) → **One-SFT+WCM (`72.8±1.9`, +33.7)**。
- **$\pi_{0.5}$：** Full-SFT (`72.9±1.8`) / One-SFT (`38.0±1.6`) → **One-SFT+WCM (`73.7±1.4`, +35.7)**（七维度中 light 91.6、env 90.3、noise 86.3 显著跃升）。
- **OpenVLA-OFT：** Full-SFT (`71.7±1.3`) / One-SFT (`29.3±1.5`) → **One-SFT+WCM (`74.0±1.8`, +44.7)**。

### 3. MetaWorld & CALVIN（Sec. 4.2, Fig. 4）
- **MetaWorld（50 任务）：** OpenVLA-OFT 与 Flow-matching 基线引入 WCM 后，相比 SFT、GRPO/PPO 及 FlowNoise/FlowSDE 均录得最高平均成功率（文本与 Fig. 4 图表标注，具体条形图在 HTML `metaworld_calvin.svg` 中绘制）。
- **CALVIN（ABC→D）：** 引入 WCM Critic 在连续长程平均完成长度上优于单帧价值基线。

### 4. 真机 WidowX-250S 七项任务（Table 3，前 50 条评测轨迹统计）
真机以 10 Hz、chunk=5（0.5 s）运行，初期各任务 100 条演示起步，经 8 轮迭代每轮每任务采集 50 条带人工纠偏的 rollout（累计 25 万 transitions）：
- **OpenVLA-OFT（SFT → +AWR → +WCM）：**
  - Carrot: `24/50 → 29/50 → 32/50` | Banana: `11/50 → 23/50 → 26/50` | Pepper: `19/50 → 24/50 → 26/50`
  - Cloth Folding: `15/50 → 29/50 → 38/50` | Towel Folding: `16/50 → 35/50 → 40/50`
  - Stovetop Cleaning: `1/50 → 10/50 → 15/50` | Conveyor Belt Sushi Picking: `9/50 → 17/50 → 22/50`
- **$\pi_{0.5}$（SFT → +RECAP → +WCM）：**
  - Carrot: `25/50 → 33/50 → 44/50` | Banana: `31/50 → 37/50 → 38/50` | Pepper: `34/50 → 40/50 → 43/50`
  - Cloth Folding: `21/50 → 32/50 → 38/50` | Towel Folding: `24/50 → 33/50 → 35/50`
  - Stovetop Cleaning: `4/50 → 27/50 → 33/50` | Conveyor Belt Sushi Picking: `13/50 → 18/50 → 24/50`

### 5. Sim-to-Real 迁移对比（Table 4，Sim SFT / Real SFT 对比 + Sim RL）
- Sim SFT (`Real 0/25` on carrot/banana/pepper) 经 WCM Sim RL 后实现 zero-shot 真机抓取 (`7/25, 7/25, 6/25`)；而 Real SFT (`13/25, 2/25, 4/25`) 叠加上 Sim RL 后达到 `11/25, 8/25, 9/25`（总分 `28/75`，提升显著）。
