# Light-WAM: Efficient World Action Models with State-Fusion Action Decoding（arXiv:2606.08242）

> 来源归档（paper，原文通读）

- **标题：** Light-WAM: Efficient World Action Models with State-Fusion Action Decoding
- **作者：** Ziang Li、Dongzhou Cheng 等（武汉大学、上海创智学院等）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.08242>（v1，2026-06-06，cs.CV）
- **PDF：** <https://arxiv.org/pdf/2606.08242>
- **代码：** https://github.com/L1ziang/Light-WAM
- **入库日期：** 2026-11-12
- **核对范围：** 读了 arXiv v1 PDF 全文：摘要、§1 引言、§2 相关工作、§3.1-3.4 方法（式 1-12）、§4.1-4.7 实验（Table 1-5，Fig.3、Fig.4 真机柱状图已看图读数）、§5 结论与局限、附录 A（算法细节）、B（训练细节、Table 6 参数构成）、C（Table 7 RoboTwin 2.0 逐任务）、Algorithm 1/2。未做：Fig.1 与 Fig.2 的图内文字在 PDF 文本层是乱码，未逐项读图（以正文与 Table 为准）；附录 D / Fig.5 真机 rollout 仅为图，未细读；未读 GitHub 代码（只静态读论文）；参考文献未逐条核对。
- **通读：** 已完成
- **一句话说明：** 基于 Wan2.1-T2V-1.3B 的轻量 World Action Model：future-video 监督放在下采样 latent 空间，动作由 StateFusionActionExpert 对多层 adapted state 做 learned-query 池化后一次前向直接输出 action chunk。

## 核心摘录（均出自论文，括号为位置）

1. **问题（摘要、§1）：** 现有 WAM 多依赖大型生成式架构，训练成本与推理延迟高，难作为高效闭环策略部署；引用 Fast-WAM [12] 称测试时生成未来视频并非必需，视频预测的收益可能主要来自训练期表征学习。
2. **方法（§3）：** 冻结 Wan2.1-T2V-1.3B，加 LoRA（作用于所有 block 的自注意力、交叉注意力、FFN 投影，附录 A）与稀疏 WAM adapter（残差瓶颈 MLP，瓶颈 256 维，层 {8,16,24}）；视频分支在下采样 latent（默认 2x 空间下采样）上做 flow matching 损失 L_video（式 6）；动作分支输入原分辨率当前观测 latent，不 rollout 未来视频；StateFusionActionExpert 每层 16 个 learned query、8 头注意力，对 query 取平均并 LayerNorm（式 9），融合后加步 embedding 经输出头一次性输出 K 步动作（式 10-11）；总损失 L = L_video + λ L_action（式 12）。
3. **规模（§4.1、Table 6）：** 总参数 1.99B，可训练 0.44B（441.03M：LoRA 87.49M、WAM adapter 2.37M、视频头 0.10M、StateFusionActionExpert 351.03M、proprio encoder 0.04M）；冻结部分含 Wan 骨干 1418.90M 与 VAE 126.89M。
4. **LIBERO（§4.2、Table 1）：** Spatial 98.2 / Object 99.6 / Goal 97.8 / Long 93.0，平均 97.2；无具身预训练（w/o EPT）中第 1、全部对比方法中第 3（LingBot-VA 98.5、Motus 97.7 更高）；Long 上 Motus 97.6、LingBot-VA 98.5 高于它。检查点按套件选取（附录 B：Spatial/Goal 60K、Object 12.5K、Long 80K 步）。
5. **RoboTwin 2.0（§4.3、Table 2、Table 7）：** 50 任务单策略，2,500 条 clean + 25,000 条 randomized 演示训练；Clean 76.4、Randomized 76.3、平均 76.4（460K 步）；Fast-WAM 91.9/91.8，LingBot-VA（w/o EPT）80.6，Motus（w/o EPT）74.9，π0 62.2，X-VLA 72.9，π0.5 79.8，作者自述高于 π0 与 X-VLA、与 Motus(w/o EPT) 相当，但不及 Fast-WAM 与最强具身预训练 WAM。
6. **训练效率（§4.4、Table 3，4×H100，全局 batch 64）：** 相对 Fast-WAM，总加载参数 6.73B→1.99B、可训练 6.02B→0.44B，单卡峰值显存 70.7→43.1GiB，吞吐 0.49→2.08 steps/s（4.25x）。逐项：换 StateFusion 头 1.14x（batch 8/卡 + 梯度累积 2），再加 latent 缓存 1.76x，再加 2x 下采样 4.25x。
7. **推理效率（§4.4、Table 4，单卡 RTX 4090 48GB，每次动作查询，含 VAE 编码与策略前向，不含仿真与 I/O）：** Light-WAM 总延迟 72.03ms（VAE 12.7ms、视觉分支 56.5ms、动作分支 2.1ms），峰值显存 4.1GiB；Fast-WAM 404.62ms / 12.7GiB；LingBot-VA 3214.14ms；Motus 2148.68ms；π0.5 76ms（> 8GiB，引自 [37]，非本文自测）。
8. **消融（§4.5、Table 5，仅 LIBERO-Spatial）：** 不下采样 99.0（默认 98.2）；5 个 adapter 层 98.0；8 个 query 95.4（默认 16）。
9. **真机（§4.7、Fig.4）：** IMETA Y1 双臂，3 个任务，每任务 50 条演示，对比 π0.5；柱状图读数 π0.5 为 80 / 93 / 60，Light-WAM 为 67 / 87 / 53（T1 清理纸团进垃圾桶、T2 叠碗后放入篮子、T3 交接水瓶放入篮子）。论文未给每任务试验次数。
10. **局限（§5）：** 更难的多任务设定里更大的 WAM 与具身预训练策略成功率更高；未在 LIBERO-Plus 等泛化与鲁棒性基准上训练或测试；后续计划加入数据增强与鲁棒性训练。

## 对 wiki 的映射

- [paper-rcl-2606-08242-light-wam-efficient-world-action-models-with-sta](../../wiki/entities/paper-rcl-2606-08242-light-wam-efficient-world-action-models-with-sta.md)
- 清单索引档：[rcl_awesome_wam_2606_08242_light-wam-efficient-world-action-models](rcl_awesome_wam_2606_08242_light-wam-efficient-world-action-models.md)
