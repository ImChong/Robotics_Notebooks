# World-Language-Action Model（WLA-0，arXiv:2606.05979）

> 来源归档（paper，全文通读）

- **标题：** World-Language-Action Model for Unified World Modeling, Language Reasoning, and Action Synthesis
- **作者：** Yi Yang, Zhihong Liu, Siqi Kou（共同一作）等；通讯作者 Zhijie Deng；单位 SJTU、SII 等
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.05979>（v1，cs.RO，2026-06-04）
- **PDF：** <https://arxiv.org/pdf/2606.05979v1>
- **代码：** https://github.com/SJTU-DENG-Lab/WLA（论文首页 Code 字段给出，本次未访问仓库）
- **入库日期：** 2026-11-12
- **核对范围：** 下载 arXiv v1 PDF 并 pdftotext 全文，读完摘要、§1 引言、§2 相关工作、§3 方法（3.1、3.2）、§4 实验（4.1–4.4 及表 1–3）、§5 结论、§6 局限、附录 A（加速）、B（仿真基准，含表 4、表 7 逐任务）、C（真机，表 5）、D（视频学新任务，表 6）；图 3(c)(d) 的柱状数值只读到文字描述与表 5，未逐点核图；参考文献仅扫读；未运行任何代码、未访问代码仓库。
- **通读：** 已完成

## 论文要点（作者自述，按章节）

1. **问题与主张（摘要、§1）：** 现有 WAM 多用双向扩散 Transformer 只预测视觉下一状态，作者认为下一状态应同时含「文本意图」和「物理动态」，提出 WLA：自回归（AR）Transformer 骨干 + World Expert + Action Expert。
2. **方法（§3.1）：** 骨干由预训练 VLM 初始化；文本子任务窗口 S_t = f(o_{t-h}, o_t, ℓ, M)（式 3.1），记忆 M 递归追加已预测子任务；追加 meta-queries Q，经因果注意力得到物理动态表征 h_t（式 3.2）；World Expert 由 h_t 与 o_t 预测未来帧 o_{t+n} 的 VAE 特征（式 3.3）；Action Expert 由 (h_t, q_t) 生成 n 步动作块（式 3.4）；总损失 L = L_act + α L_wm + β L_lang（式 3.5）。只预测目标帧而非整段视频，依据附录 B（表 4）。
3. **推理（§3.2）：** 高效模式默认关闭 World Expert；TTS 模式采样 K 个候选动作块，World Expert 想象未来静态帧，价值模型打分选最高者。价值标签 v_t = y·γ^(T-t)，用微调后 WLA-0 的 rollout 训练。
4. **实现（§4.1）：** WLA-0 总参数 3.4B，RynnBrain-2B（2.1B）骨干 + SANA-600M（含 VAE 共 900M）World Expert + 390M flow-matching Action Expert；64 个 meta-queries；动作块 LIBERO 为 8、其余为 32；α=0.1，β=0.005；AdamW，基础学习率 5e-5。
5. **RoboTwin 2.0（§4.2，表 1、附录表 7）：** 50 任务，2500 条干净 + 25000 条强随机化轨迹训练 100k 步，批量 256，每任务 100 次评测；WLA-0 Clean 92.94、Rand. 90.02；去掉 L_wm 为 90.98 / 89.34；π0.5 为 82.74 / 76.76；Lingbot-VA 92.90 / 91.50（附录 B.1：Lingbot-VA 为 seen instructions 设定，其余为 unseen）；Fast-WAM 91.88 / 91.78。该基准上关闭了语言子任务预测。
6. **LIBERO（§4.2，表 1）：** 四套件单模型训练 100k 步，每任务 50 次评测；WLA-0 平均 98.6，TTS（6 候选、想象步长 2）98.9；Lingbot-VA 98.5，Fast-WAM 97.6。
7. **RMBench（§4.2，表 2）：** 仅 M(n) 子集四个任务，每任务单独训练 30k 步、批量 448；WLA-0 平均 56.5%（Battery Try 45、Blocks Ranking Try 23、Cover Blocks 84、Press Button 74），Mem-0 28.5%，Fast-WAM 13.3%，π0.5 5.5%，X-VLA 7.3%；去掉 L_lang 降至 17.3%（正文写 17.25%）。
8. **真机（§4.3，图 3、附录表 5）：** AgileX Piper 双臂，四个长程任务，每任务 60 条示教，每设定 10 次试验；与 π0.5、Motus 比，WLA-0 从头训练（无预训练）。标准设定平均成功数 WLA-0 7.5、π0.5 7、Motus 6；OOD 物体 5.25 / 4.5 / 3.75；OOD 场景 4.75 / 4.75 / 4。Dispose Trash 标准设定 6 / 4 / 1。Stack Cup 上延迟较 Motus 低约 40 倍（同步执行，10 步 flow-matching，动作块 32）。
9. **从视频学新任务（§4.4，表 3、6，附录 D）：** RoboTwin 2.0 的 45 个已见任务 + 5 个未见任务；基线 13.0/11.6（Clean/Rand.），加入未见任务同本体视频 34.4/30.0，跨本体（ARX-X5）视频 28.8/27.4；加入真人第一视角视频（表 6）反而降到 7.8/7.8，作者推测是真人视频与仿真的域差。
10. **加速（附录 A）：** CUDA Graph、Triton 算子融合、预计算与缓存，延迟从约 116 ms 降到 40 ms 以内（RTX 5090）。
11. **作者自述局限（§6）：** 真机只有少量双臂任务、单一平台；视频学新任务实验用的是仿真机器人视频。

## 阅读者备注（非论文原文）

- 摘要与图 1(d) 的 RoboTwin「91.5」是 Clean 与 Rand. 两栏的平均（92.94 与 90.02 的均值约 91.48，推算），不是 Clean 单栏。
- 摘要称 WLA-0「state-of-the-art」，但表 1 中 Rand. 栏 WLA-0（90.02）低于 Fast-WAM（91.78）与 Lingbot-VA（91.50）。
- 真机每设定仅 10 次试验，Unscrew Cap 标准设定 WLA-0 为 7、π0.5 为 9，差异多处在 1 到 2 次之内，统计意义有限（推测）。

## 对 wiki 的映射

- [paper-sa-2606-05979-world-language-action-model-for-unified-world-mo](../../wiki/entities/paper-sa-2606-05979-world-language-action-model-for-unified-world-mo.md)
- 早期策展摘录：[sun_awesome_ego_2606_05979_world-language-action-model-for-unified](sun_awesome_ego_2606_05979_world-language-action-model-for-unified.md)
