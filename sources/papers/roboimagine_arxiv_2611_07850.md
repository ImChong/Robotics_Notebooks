# RoboImagine（arXiv:2611.07850）

> 来源归档（paper）

- **标题：** Learning Robotic World Models with Semantic and Visual Feedback
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.07850>（v1，cs.RO，2026-11-05）
- **PDF：** <https://arxiv.org/pdf/2611.07850>
- **代码：** https://github.com/Alexander-wu/RoboImagine（论文附录 D 给出；权重与演示输入见 https://huggingface.co/easylearning/RoboImagine）
- **入库日期：** 2026-11-09
- **作者 / 机构（以论文首页脚注为准）：** Hao Wu、Yizhou Zhao、Laszlo A. Jeni（CMU）；Yuqi Li、Yingli Tian（The City College of New York）；Kun Wang、Yu Cheng、Bo An（NTU）；Xiao Luo（UW-Madison）；Ming-Hsuan Yang（UC Merced，通讯作者）
- **一句话说明：** 对自回归（GPT 式）机器人世界模型做后训练：用分组多样化束搜索为同一输入生成多个候选未来，用冻结多模态评判器的成对语义偏好加像素 / 感知视觉一致性打分，融合后做组相对优势的策略优化。
- **核对范围：** 通读正文 §1–§5（摘要、引言、相关工作、方法 §3.1–3.5、实验 §4.1–4.5、结论与局限）、表 1–5、图 1–14 的图注与可读数值，以及补充材料附录 A（实现）、B（搜索 / 记忆 / 反馈）、C（评测与对照）、D（发布资源）；附录 E（Lemma 1 / Corollary 1 / Proposition 1 的证明）读了到命题 1 的证明，其后的局部改进分析只扫读；附录 F（评判器提示词全文）只确认存在、未逐字读。图中的图像内容（定性对比）未逐帧核验。未打开 GitHub / Hugging Face 页面，开源范围仅按论文附录 D 转述。
- **通读：** 已完成

## 开源状态

- 论文附录 D 称仓库含模型实现、检查点推理接口与实验源码，Hugging Face 含各数据集的主 / SFT 预测器、视觉编解码器、文本适配器、预处理配置与演示输入；**不含每个消融检查点与完整数据缓存**，全量复现还需原始数据集与评判器资产。（以论文自述为准，入库日 2026-11-09 未打开链接核对。）

## 核心摘录

1. **摘要（原文）：** RoboImagine 通过候选未来比较，用语义与视觉反馈后训练自回归机器人世界模型；RT-1 上在匹配初始化、更新次数与评测协议下，相对继续监督微调 PSNR 提高 1.26 dB、LPIPS 降低 11.93%。
2. **§3.1：** 视觉编码用 iVideoGPT 式上下文条件双分支编码器加 FSQ，每帧 80 个动态 token、初始上下文 1280 个 token；12 层 LLaMA 式解码器，T5 文本经交叉注意力注入（附录 A.1：宽度 768、词表 9008）。
3. **§3.2：** GDBS 用 G=4 组、每组 B=2 束、多样性惩罚 λ=0.1，每组保留 1 个完整候选；每分支保留固定初始上下文 + 动态锚点 + 最近 W=6 个状态 - 动作对。
4. **§3.3：** 冻结 4B mJev 评判器对候选两两比较（8 帧、双向顺序），语义奖励为对其余候选的平均反对称偏好差；视觉奖励为 -(L1 像素误差 + LPIPS)；各分支组内标准化后按 (0.25, 0.75) 融合。
5. **§4.3 / 图 9：** 500 次更新，继续 SFT 为 1.06 GPU 小时、RoboImagine 为 28.15 GPU 小时；论文明说不证明等算力下的效率优势。
6. **结论节（自述局限）：** 证据仅限离线视频预测，不证明闭环规划或真机控制改进。

## 对 wiki 的映射

- [paper-roboimagine](../../wiki/entities/paper-roboimagine.md)
