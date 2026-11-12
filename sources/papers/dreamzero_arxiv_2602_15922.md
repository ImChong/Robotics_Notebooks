# DreamZero: World Action Models are Zero-shot Policies（arXiv:2602.15922）

> 来源归档（paper · 原文通读）

- **标题：** World Action Models are Zero-shot Policies（方法名 DreamZero）
- **类型：** paper
- **作者：** Seonghyeon Ye、Yunhao Ge、Kaiyuan Zheng 等（NVIDIA；项目负责人 Ye、Zhu、Jang 等，见首页）
- **arXiv：** <https://arxiv.org/abs/2602.15922>（v1，2026-02-17）
- **PDF：** <https://arxiv.org/pdf/2602.15922v1>
- **项目页：** <https://dreamzero0.github.io>
- **代码：** <https://github.com/dreamzero0/dreamzero>（论文 §1 贡献列表给出）
- **入库日期：** 2026-11-12
- **核对范围：** arXiv v1 PDF 全文（pdftotext），含正文 §1–§6、附录 A–H（含 Algorithm 1/2、评测任务表的文字部分）；图只读了图注与可提取的数字标注，未逐图看像素，Fig.8/9/10 的柱状数值有部分仅靠文字描述交叉确认；参考文献列表只扫读；论文正文提到的「Table 9」（Table 2 的基线来源）在 PDF 文本中未找到对应表格；未运行任何代码，未核对开源仓库内容与 OpenReview 版本。
- **通读：** 已完成

## 核心摘录（均出自论文，括号为章节）

1. **WAM 定义（§1, §2.2, §3.1）：** 在预训练图像到视频扩散骨干（Wan2.1-I2V-14B-480P，§4.1）上联合预测未来视频与动作；联合分布分解为自回归视频预测与逆动力学 IDM（式 1）。作者用 WAM 而非 VAM 的理由：视频只是世界建模目标之一，将来可换触觉、力反馈或学到的隐表示（§2.2）。
2. **架构（§3.1, 附录 C）：** 仅新增状态编码器、动作编码器与解码器；多视角拼成单帧；自回归只用于视频模态；每个 chunk K=2 个潜帧，默认 M=4 个 chunk；AgiBot 动作 30 Hz、H=48，DROID 动作 15 Hz、H=24，每 chunk 均覆盖 1.6 秒；最大上下文 8 个潜帧，约 33 帧原始帧，6.6 秒。
3. **训练（§3.1, §4.1）：** flow matching，视频与动作共享去噪时间步，teacher forcing（以干净历史 chunk 为条件）；训练 100K 步、全局 batch 128（AgiBot 与 DROID 均如此）；更新全部 DiT 块与状态/动作编解码器，冻结文本编码器、图像编码器与 VAE；LoRA 试过，结果偏差（脚注 7）。
4. **推理（§3.1, Alg.2）：** 联合去噪视频与动作，KV cache；每次执行动作后用真实观测替换预测帧写回 cache，作者称此举消除自回归视频生成的误差累积。
5. **实时化（§3.2, Table 1, 附录 D）：** 朴素实现单 GPU 约 5.7 s/chunk（16 步扩散）；异步执行；CFG 并行（两卡，单步延迟降 47%）、DiT 缓存（16 步降到约 4 步）、torch.compile 加 CUDA Graphs、NVFP4 量化（Blackwell）、cuDNN 注意力；累计加速 H100 上 9.6x，GB200 上 16.6x；加 DreamZero-Flash 后 GB200 上 38x，延迟约 150 ms。Discussion 称达到 7 Hz 用了 2 块 GB200。
6. **DreamZero-Flash（§3.2.5, 附录 D.3, Table 3）：** 训练时视频时间步取 1−η，η~Beta(7,1)（E[t_video]=0.125），动作时间步仍均匀；Flash 作为主模型训练之后的最后一阶段。桌面清理任务上：4 步 83%±6.1%（350 ms），1 步 52%±10.2%（150 ms），Flash 1 步 74%±10.1%（150 ms）。
7. **数据（§4.1, 附录 E）：** AgiBot G1 约 500 小时遥操作，22 个环境，7193 条 episode，平均 4.4 分钟、约 42 个子任务；任务表内某任务采满 50 个 episode 即下架以强制多样；DROID 用于 Franka 版本。
8. **基线（§4）：** GR00T N1.6 与 π0.5，各有 from-scratch（仅 VLM 权重）与 from-pretrained（官方跨本体权重，再在同数据上继续训）两种；预算按总 batch 与梯度步数对齐。DROID 评测中 π0.5-DROID 为公开权重，GR00T N1.6-DROID 为作者内部训练。
9. **评测协议（§4.1）：** AgiBot：已见任务 10 个、未见任务 10 个，各 8 次/任务（4 台机器人），每个检查点共 80 次；DROID：20 个已见、20 个未见动词任务，各 2 次，共 80 次；评测场地与训练数据采集地点不同；按部分完成度 0 到 1 打分。
10. **结果（§5.1）：** AgiBot 已见任务（新环境、新物体）平均任务进度 DreamZero 62.2%，最强预训练 VLA 基线 27.4%，from-scratch VLA 接近 0（Fig.8）。AgiBot 未见任务 DreamZero 39.5%，预训练 VLA 基线 16.3%，from-scratch 低于 1%（Fig.9）。DROID-Franka 未见动词任务 DreamZero 任务进度 49%、成功率 22.5%，GR00T N1.6 为 31% 与 12.5%，π0.5 为 33% 与 7.5%（Fig.9）。后训练三任务（叠衣、装水果、清桌）上 DreamZero 与预训练 VLA 持平或更好，水果装箱优势明显（Fig.10，§5.1 Q3）。
11. **跨本体（§5.1 Q4/Q5, Table 2）：** 对 9 个未见任务（排除拉车），每种来源 72 条多视角轨迹（每任务 8 条，YAM 机器人 20 分钟、人类 12 分钟），只用视频目标、无动作标签，与 AgiBot 预训练数据 1:1 混合，在 DreamZero-AgiBot 检查点上继续训 10K 步。基线 38.3%±7.6%，加人类视频 54.3%±10.4%，加 YAM 视频 55.4%±9.5%。小样本新本体适配：55 条轨迹、11 个任务、约 30 分钟的 YAM 玩耍数据后训练（11 条简短全局语言标注）。
12. **消融（§5.2, Table 4）：** 仅在 PnP Easy、50K 步、batch 32 下。数据多样 vs 重复（均约 500 小时）：50%±6.3% vs 33%±4.2%；5B vs 14B：21% vs 50%；把 VLA 放大到 5B/14B 规模同等对比，均为 0%；双向 vs 自回归：任务进度均为 50%，自回归动作更平滑、推理快 3–4 倍。
13. **作者承认的局限（§6, 附录 H, 脚注）：** 尚未证明 WAM 缩放律；7 Hz 需 2 块 GB200，而 VLA 在消费级 GPU 可超 20 Hz；视觉记忆仅约 6 秒；未显式评测依赖长时记忆的任务（脚注 2）；亚厘米精度任务（插钥匙等）是行为克隆的通病；失败多源于视频生成错误，机器人忠实执行错误的视频计划（§5.1，附录 H, Fig.16）；人类视频仅 12 分钟实验室数据；跨本体改进是「早期信号」，成功率仍中等；每个本体分别预训练，多本体训练留作未来工作（§4）。

## 对 wiki 的映射

- [paper-notebook-dreamzero-world-action-models-are-zero-shot-poli](../../wiki/entities/paper-notebook-dreamzero-world-action-models-are-zero-shot-poli.md)
- 此前的策展锚点（笔记摘要，非全文）：[humanoid_pnb_dreamzero-world-action-models-are-zero-shot-poli](humanoid_pnb_dreamzero-world-action-models-are-zero-shot-poli.md)
