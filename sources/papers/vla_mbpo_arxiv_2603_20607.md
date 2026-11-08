# VLA-MBPO（arXiv:2603.20607）

> 来源归档（paper）

- **标题：** Towards Practical World Model-based Reinforcement Learning for Vision-Language-Action Models
- **类型：** paper
- **作者：** Zhilong Zhang, Haoxiang Ren, Yihao Sun（共同一作）, Yifei Sheng, Haonan Wang, Haoxin Lin, Zhichao Wu, Pierre-Luc Bacon, Yang Yu（通讯）；南京大学 / Mila / 蒙特利尔大学
- **arXiv：** <https://arxiv.org/abs/2603.20607>（v1，2026-03-21，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2603.20607>
- **代码：** PDF 全文未给出代码或项目页链接（全文检索 github / project page 无命中）；未另行核查项目页
- **入库日期：** 2026-11-08
- **出处：** PDF 首页标 "Preprint. March 24, 2026"，未写会议名；ICML 2026 取自 [RCL 清单摘录](rcl_awesome_wam_2603_20607_towards-practical-world-model-based-rein.md)，本归档未能从论文本身核实
- **一句话说明：** 用统一多模态模型（UMM，基座 Bagel）做世界模型，同时预测多视角画面与任务成败，配合交错视角解码与 chunk 级分支 rollout，在想象数据上用 Flow-Noise PPO 微调 VLA（π0.5）。

- **核对范围：** 逐页读完 arXiv v1 PDF 全部 27 页：摘要、§1 引言、§2 预备知识、§3 方法（3.1–3.3）、§4 理论（Thm 4.1 / 4.2 及案例）、§5 实验（Table 1–3、Fig 3–6）、§6 相关工作、§7 结论与局限、附录 A–B（引理与定理证明，只核对结论式与推导走向，未逐步复算每一行不等式）、附录 C（真机硬件与任务协议）、附录 D（掩码、提示词、Table 4–5）、附录 E（定性与失败案例）、附录 F（算力）。图表以文本抽取加渲染页面核对，Fig 5 柱高为目测读数。未读：参考文献所引的其他论文；未运行任何代码（论文未提供）。
- **通读：** 已完成

## 核心摘录

1. **问题设定（§1、§2）：** 真机上微调 VLA 的 RL 成本高且有安全风险；用世界模型做虚拟环境有三类难点——像素级建模、多视角一致、稀疏奖励下的误差累积。形式化为语言条件 MDP（L-MDP），奖励 r(s, l) ∈ {0,1}，策略输出长度 k 的动作块。
2. **世界模型（§3.1）：** 动作离散为整数 token（范围示例 [0,256]）直接送入 UMM 词表，无需改结构；基座为 Bagel；UMM 直接预测 chunk 末帧，跳过中间帧（Fig 2），与视频世界模型相比 rollout 更快；chunk 级奖励 r_θ = Σ γ^(i-1) r(s_{t+i}, l)。
3. **交错视角解码（§3.1，式 3）：** 先生成 head 视角下一帧，再以其为条件生成 wrist 视角（而非两视角并行独立生成）；实现靠注意力掩码（Fig 10）。
4. **chunk 级分支 rollout（§3.2）：** 从离线数据集中任意观测起步、只滚少量 chunk；世界模型按 chunk 预测，有效 horizon 再缩 1/k。借鉴 Park et al. 2025，作者称其「未在像素级 VLA 微调中验证过」。
5. **算法（§3.3，Algorithm 1）：** 真机采集 → 世界模型微调 → 在世界模型内用 Flow-Noise（π_RL 的 PPO 变体）更新 VLA，另挂 MLP 价值头；优势估计为 chunk 级 GAE（式 4）。不使用保守正则、拒绝采样或 Q 模型；全任务共用一组超参（Table 5）。
6. **理论（§4）：** Thm 4.1 给出步级世界模型 + 全 horizon rollout 的价值差上界；Thm 4.2 给出 chunk 级世界模型 + n-chunk 分支 rollout 的上界。γ=0.99、k=10、n=2 的案例：约 4183 ε_π + 18916 ε_m → 约 1710 ε_π + 400 ε_m（系数省略了 r_max 因子，笔者按式复算与文中数字吻合）。
7. **世界模型评测（§5.1，Table 1）：** LIBERO-Object，每任务 50 条训练轨迹，10 条留出，rollout 40 步。UMM-World 对 Ctrl-World：head 视角 LPIPS 0.094 对 0.150，PSNR 23.29 对 21.95；wrist 视角 LPIPS 0.254 对 0.435，PSNR 18.76 对 13.87；推理时间 10 对 21（表内无单位，正文称快约 2 倍）；奖励 ACC 98.4 / F1 0.861，对 Qwen3-VL-8B 的 97.0 / 0.841。去掉交错解码（w/o IVD）wrist LPIPS 退到 0.454，去掉预训练（w/o PT）则整体大幅变差。
8. **LIBERO 策略结果（§5.2，Table 2）：** 起点为 π0.5 单轨迹 SFT；行为数据为每任务 50 条 episode。Spatial / Object / Goal / Long / Avg：π0.5(SFT) 78.2 / 88.6 / 85.8 / 54.6 / 76.8；π_RL 86.0 / 92.4 / 90.8 / 61.2 / 82.6；IDQL 79.0 / 92.4 / 86.4 / 52.2 / 77.5；BC(WM) 80.6 / 89.8 / 85.0 / 48.6 / 76.0；VLA-MBPO 87.8 / 96.6 / 92.8 / 66.8 / 85.9（相对 SFT +9.6 / +8.0 / +6.8 / +12.2 / +9.1）。96.6 是 Object 子集成绩，不是平均。
9. **真机（§5.3、附录 C）：** ARX-X5 双臂（Plug In、Fold Towel）与 Galaxy-R1 全身（Pick Cup、Insert Pen、Wipe Board），每任务评测 50 次（30 见过 + 20 未见）。SFT 用遥操作示教（50 / 60 / 100 条不等），RL 阶段再加 50 条 π0.5(SFT) 自采轨迹。结果只有 Fig 5 柱状图（无数值标注），对比项只有 π0.5(SFT)、IDQL、VLA-MBPO；目测 VLA-MBPO 约 34 / 56 / 42 / 54 / 58（Plug In / Fold Towel / Insert Pen / Pick Cup / Wipe Board），SFT 约 6 / 38 / 28 / 30 / 28。
10. **消融（§5.4，Table 3、Fig 6）：** LIBERO-Long 上分支长度 1 / 2 / 4 chunk 为 63.9 / 66.8 / 62.9，全 horizon 为 52.8；每轮想象样本量越大成功率越高（Fig 6 单调）。
11. **局限（§7、附录 E）：** 样本生成仍需大量算力（8×H100；世界模型训练约 7-8 小时，策略优化约 4-6 小时，附录 F）；UMM 未在带动作标注的机器人数据上预训练，下游仍需少量数据微调世界模型。失败案例（附录 E.2）：部分可观测（机械臂移出 head 视角、转身后首次出现的区域无法凭空生成）、大幅物理运动（Wipe Board 出现 motion collapse，Fold Towel 出现不合物理的织物形态）。

## 与二手说法的核对备注

- 论文没有把「接触（抓取滑动、碰撞、摩擦）预测不准」写成局限；其失败分析是部分可观测与大位移 / 非刚体形变，且 §5.3 正文反称 Plug In 体现了接触密集操作上的提升。
- 「预测越长越偏离」在论文中的证据是 §3.2 / §4.1 的误差累积论述、Table 3（全 horizon 52.8 低于分支 rollout）与 Fig 14（标题为 large prediction horizon，实际展示的是大位移 / 大形变）；没有给出「误差随预测步数变化」的专门曲线。
- 真机实验未与在线 RL 基线（π_RL）对比，因而「减少真机交互」的量化依据主要是 LIBERO 中同等数据预算对比（§5.2）和每任务仅 50 条自采轨迹的设定。

## 对 wiki 的映射

- [paper-rcl-2603-20607-towards-practical-world-model-based-reinforcemen](../../wiki/entities/paper-rcl-2603-20607-towards-practical-world-model-based-reinforcemen.md)
- [RCL 清单摘录](rcl_awesome_wam_2603_20607_towards-practical-world-model-based-rein.md)
