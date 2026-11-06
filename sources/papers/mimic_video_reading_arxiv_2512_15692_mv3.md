# mimic-video 论文通读（VAM 视角）

> 来源归档（原文通读 · 支撑 `wiki/methods/mimic-video.md`）

- **标题：** mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs
- **作者：** Jonas Pai, Liam Achenbach, Victoriano Montesinos, Benedek Forrai, Oier Mees, Elvis Nava（mimic robotics / Microsoft Zurich / ETH / UC Berkeley）
- **类型：** paper / video-action-model / imitation-learning
- **原文：** <https://arxiv.org/abs/2512.15692>（2026-11 通过 arXiv PDF v2 获取全文，pdftotext 提取）
- **一句话说明：** 把预训练视频扩散骨干的中间潜表示作为「视觉计划」，由流匹配动作 DiT 充当逆动力学模型输出动作块，骨干可冻结。
- **核对范围：** 通读摘要、§I 引言、§II 相关工作、§III oracle 案例研究、§IV 方法（流匹配预备、视频模型、动作解码器、动作采样、训练、Algorithm 1/2）、§V 实验（SIMPLER-Bridge、LIBERO、真机双臂灵巧手、样本效率与收敛、τv 权衡）、结论与局限，以及附录（超参表、数据预处理、视频模型层选取、π0.5-style 基线经验、τv 去噪分析）。表 I–III 的逐项数值与参考文献列表仅浏览，未逐项核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **动机（§I）：** VLA 的图文预训练是静态的，物理动力学全靠后训练阶段从昂贵遥操作数据里学；把视频压成语言计划 / affordance / 关键点会形成信息瓶颈；已有视频方法常需在每个控制步合成未来帧。
- **Oracle 案例（§III）：** 在真值未来视频的潜特征上训练动作解码器，成功率近乎完美；用预测潜特征时，机器人域微调的视频模型优于现成视频模型，结论是策略性能随视频模型质量缩放。
- **架构（§IV）：** 两个条件流匹配模型：Cosmos-Predict2 2B 潜空间 DiT（上下文前缀 5 帧，T5 语言交叉注意）；动作 DiT 编码本体与动作块，随机以 mask token 替换本体 token，每层含对视频第 k 层隐状态的交叉注意、自注意、MLP，AdaLN 注入 (τv, τa) 的低秩双线性编码。
- **采样（§IV-E，Algorithm 1）：** 视频流积分到 τv，取前 k 层激活条件化动作解码器，动作再完整去噪；τv=1 时骨干只需一次前向。
- **训练（§IV-F，Algorithm 2）：** 阶段一 LoRA 微调视频骨干；阶段二冻结骨干从零训动作解码器，每步独立采样 τv（logit-normal）与 τa。
- **实验（§V）：** SIMPLER-Bridge 平均成功率最高；LIBERO 上从零训练的解码器超过多数基于通用模型微调的方法并明显优于 π0.5-style 基线；真机双臂（Panda + 16 自由度 mimic 手，包裹分拣与卷尺收纳）仅用单个工作区相机，超过单任务 DiT-Block Policy；动作数据分别约 1h33m（512 条）与 2h14m（480 条），视频骨干用约 200 小时语料微调。
- **样本效率与收敛（§V-B）：** 仅用 10% 数据即达到 VLM 条件解码器的最高成功率；每任务一条演示（减少 98%）仍有 77% 平均成功率；收敛更快且渐近成功率更高。
- **τv 权衡（§V-C）：** SIMPLER 上最佳自主性能出现在 τv=1；用带噪真值潜特征扫描时动作重建误差最低点约在 τv≈0.4，趋近 0（完全重建）时误差急升。
- **附录：** 最佳源层 k=19（视频模型），π0.5-style 基线交叉注意第 11 层；解释为「噪声即增强」缓解训练 / 推理分布不一致，且接近 τv=0 时隐状态趋于恒等映射而信息量下降。
- **局限（结论）：** 仅单视角视频骨干；尚未训练统一的大规模跨本体模型；真机任务集有限。
