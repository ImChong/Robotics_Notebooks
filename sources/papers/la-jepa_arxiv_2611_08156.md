# La-JEPA（arXiv:2611.08156）

> 来源归档（paper）

- **标题：** La-JEPA: Compact Robot Policies with Latent World Modeling
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.08156>
- **PDF：** <https://arxiv.org/pdf/2611.08156>（v1，2026-11-05，cs.RO）
- **代码：** https://github.com/mt-clemente/lajepa（论文首页与 Reproducibility Statement 自述；本次未能打开仓库核对内容，开源程度待核实）
- **入库日期：** 2026-11-09
- **机构（论文首页）：** Huawei Technologies Ltd.（Mateo Clémente、Rui Heng Yang、Tongtong Cao、Zhanguang Zhang、Yingxue Zhang）；University of Toronto（George Z. Xue，标注为 Huawei 实习生）；McGill University（Amin Abyaneh，标注为 Huawei 实习生）；Mila（Liheng Ma）
- **一句话说明：** 在冻结 V-JEPA 2.1 ViT-L 特征空间中从零训练一对紧凑 DiT（World DiT + Action DiT），训练时联合预测未来特征与动作，推理时只生成动作，无需生成式预训练骨干。
- **核对范围：** 主文第 1 至 7 节、AI Use / Ethics / Reproducibility 声明及参考文献列表全文；附录 A（设计研究，含 Table 6 至 10）、附录 B（训练与评测协议，含 Table 11 至 14）、附录 C（延迟与参数统计，含 Table 15、16）、附录 D（实机设置与 Table 17）、附录 E.1、E.3、E.4 通读。附录 E.2（评测设置扫描）与 E.5（逐任务 RoboTwin 表 18 至 19）只读了前几行表与文字说明，未逐行核对。图 2 目视核对，其余图表通过 pdftotext 抽取文字与图注阅读，未逐图目视。未运行任何代码，未打开代码仓库。
- **通读：** 已完成

## 开源状态

- 论文称代码（特征预计算、训练、LIBERO / LIBERO-Plus / RoboTwin 2.0 评测与 S/B/L 配置）发布于上述 GitHub 地址；本次环境访问 GitHub 被拦截，**未核实**。论文称不发布新数据集（Ethics Statement）。

## 核心摘录

1. **动机与定位（摘要、§1）：** WAM 常改造预训练视频生成器、VLA 常基于预训练 VLM，策略架构与推理成本被「为生成而设计」的大 Transformer 绑定；作者问：能否不继承生成骨干，用从零训练的紧凑 WAM 达到同等性能。作者称「据我们所知」此前没有工作在预训练 JEPA 特征空间中从零训练紧凑 WAM 策略（§1，作者自述）。
2. **架构（§3.1）：** V-JEPA 2.1 ViT-L 逐相机编码，得到观测特征 z_ctx 与未来 32 帧的目标 z_fut，是模型唯一视觉输入与预测目标；FLAN-T5-large 冻结编码指令；World DiT 与 Action DiT 均从零初始化，每层交互；注意力掩码令动作 token 只看观测与动作，不看未来特征。
3. **训练 / 推理（§3.2）：** 未来特征与动作联合 flow matching，损失 L = MSE(v_f, z_fut − ε_f) + MSE(v_a, A − ε_a)（式 2）；推理丢弃未来 token，观测 KV 缓存，仅 Action DiT 做 Euler 去噪；LIBERO 每次执行 10 个动作，RoboTwin 24 个（均沿用 Fast-WAM）。
4. **设计研究（§4，图 3，表 6）：** 联合未来预测 vs 仅动作：RoboTwin clean +7.0、randomized +3.7 个百分点，LIBERO-Plus 三 seed 平均 +2.1（表 7）；V-JEPA vs Wan2.2 VAE 潜变量：RoboTwin clean +15.9 / randomized +16.0，LIBERO-Plus +6.0；相机 dropout、本体 dropout、图像增强加编码器 LoRA 均有增益（§4.2）。
5. **主结果（§5.1，表 1、2、3）：** 无具身预训练策略中，RoboTwin 2.0 clean / randomized 均最高（S/B/L 的 clean 85.5/86.2/86.0，randomized 33.9/35.4/32.1）；LIBERO-Plus 修正指令后 S/B/L 为 86.3/87.3/88.0，NowWAM 为 86.2；LIBERO 97.9 至 98.1。
6. **LIBERO-Plus 指令泄漏（§5.2，表 4、13、14）：** 官方评测器在 10,030 个变体中的 8,493 个把扰动参数拼到指令后；修正后 Fast-WAM +19.3、LeapBot-WA +8.1、Cosmos Policy +4.1、La-JEPA +3.9 至 6.3。
7. **成本与实机（§5.4、§5.5，表 5、15）：** 优化构建下 10 步去噪延迟 S/B/L 为 17.78/25.16/59.72 ms，Fast-WAM 88.09 ms，Faster-WAM 50.96 ms，π0.5 71.89 ms；四项 Franka 实机任务平均成功次数（满分 20）La-JEPA 17.50，π0.5 17.25，Fast-WAM 16.00，Faster-WAM 16.25。
8. **论文自述局限（§6）：** 未系统调学习率 / 权重衰减 / dropout / LoRA 配置 / 两个 DiT 相对容量；RoboTwin 随规模非单调，原因未隔离；仅用目标演示训练，具身预训练的扩展性未知；randomized RoboTwin 仍低于最强具身预训练方法；实机仅四任务一套机器人。

## 对 wiki 的映射

- [paper-la-jepa](../../wiki/entities/paper-la-jepa.md)
