# ForeAct（arXiv:2611.11500）

> 来源归档（paper）

- **标题：** ForeAct: Geometry-Guided Action Learning from World Representations
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.11500>
- **PDF：** <https://arxiv.org/pdf/2611.11500v1>（v1，2026-11-08，cs.CV）
- **代码：** https://github.com/Bulrush666/ForeAct
- **权重：** <https://huggingface.co/Bulru1234/ForeAct>（README 列出 bridge 50K 步与 LIBERO 100K 步两个检查点）
- **作者机构：** 香港科技大学（广州）；京东探索研究院（JD Explore Academy）
- **入库日期：** 2026-11-11
- **一句话说明：** 把预训练视频世界模型（Cosmos-Predict2-2B）当作单次前向的表征骨干，训练时用可学习几何 token 预测下一帧相对深度作辅助监督，推理时移除几何读出头；动作头仍是流匹配 Action DiT。

## 核对范围与通读

- **核对范围：** 论文 PDF v1 全部 9 页：摘要、引言、相关工作、方法（III-A 至 III-C）、实验（IV-A 至 IV-E，含 Table I-III、Fig. 4-7）、结论、参考文献；该版本没有附录或补充材料，论文也没有单独的局限节。另静态读了仓库 README（raw.githubusercontent.com，main 分支，2026-11-11）；GitHub API 与代码文件未读，未运行任何代码。
- **通读：** 已完成

## 核心摘录（均出自论文）

1. **方法（III-A/B，式 1-4）：** 输入单帧 RGB 与语言指令，不要求本体状态；冻结 VAE 把当前帧编码为干净 latent，与 8 个可学习几何 token 拼接，在扩散时刻 t_diff=0 过一次 Cosmos 骨干；输出视觉 token 送入 Action DiT，几何 token 经读出头解码为下一帧相对深度。
2. **训练目标（III-A，式 3）：** L = L_act + λ_geo(λ_lat L_lat + λ_pix L_pix + λ_grad L_grad)，λ_geo=0.01、λ_lat=1、λ_pix=0.1、λ_grad=0.05；动作为条件流匹配；深度对共享骨干的梯度另乘 0.01。深度目标由 MoGe-3 离线生成，经冻结 VAE 编码，仅训练用。
3. **实现（IV-A）：** Cosmos-Predict2-2B Video2World 初始化；16 层 DiT-B 动作专家，隐维 1024；LIBERO 训练 100K 步（4 GPU，全局 batch 48，动作 horizon 8），SimplerEnv-Bridge 50K 步（8 GPU，batch 96，horizon 16）；骨干学习率 1e-5，动作专家与读出头 1e-4。
4. **LIBERO（Table I）：** Spatial 98.4 / Object 98.8 / Goal 97.2 / Long 91.8，平均 96.6；表中标 Pre-Train ✗（无额外机器人动作预训练）。对比：Fast-WAM 97.6、Cosmos Policy 98.5、GeoPredict 96.5、DepthVLA 94.9。
5. **SimplerEnv-Bridge（Table II）：** ST 87.5 / CP 54.2 / SB 12.5 / EB 100.0，平均 63.5；GR00T N1.5 为 61.9。
6. **真机（IV-C，Fig. 5）：** Franka Panda + 固定 RealSense D435i RGB；6 个任务，每任务 15 次；对比 π0 与 Fast-WAM；ForeAct 67/90（74.4%），Fast-WAM 58（64.4%），π0 52（57.8%）；六个任务中五个最好。
7. **延迟（IV-D，Fig. 6）：** 单张 H20、batch 1，LIBERO 上 168 ms、约 2.1B 参数；文字称与 Cosmos Policy 相当成功率下延迟约其三分之一，且比 Fast-WAM 更快。Fig. 6 为散点图，仅正文给出 168 ms 这一具体数。
8. **消融（IV-E，Table III、Fig. 7，LIBERO 平均）：** Diffusion WAM 93.2；Action Only（干净 latent）94.3；仅几何 token 无深度监督 94.7；对视觉特征直接深度监督 93.4；单向注意力 94.9；完整 96.6。几何 token 数 1/8/32/64 为 94.3/96.6/95.2/94.7；下一帧深度 96.6，当前帧深度 95.2，t+4 为 95.5，t+8 为 94.9。

## 仓库 README 要点（静态读）

- README 称动作头为 flow-matching，预测连续动作块；8 个几何 token 训练时受下一帧相对深度监督。
- 结果表：LIBERO 4 套件 2000 episode 96.55%，SimplerEnv-Bridge 4 任务 96 episode 63.54%；评测记录在 `configs/evaluation/`。
- 评测须匹配检查点、denoising steps、seed、执行 horizon 与后处理，说明动作头推理仍含去噪步（具体步数未在论文正文给出，README 也只指向配置文件，本次未读该文件）。
- 需自备 Cosmos-Predict2 权重、LeRobot 数据与 MoGe-3 深度缓存；README 未写真机训练或部署代码（未核实仓库其余文件）。

## 对 wiki 的映射

- [paper-foreact](../../wiki/entities/paper-foreact.md)
