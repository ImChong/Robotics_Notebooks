# 事件相机动作识别多帧预训练论文来源归档

## 元数据

- **题目：** Event-based Action Recognition by Pre-Training on Multi-Frame Representations
- **作者：** Emma Norrby, Tomas Wilkinson, Per Nyblom, David Gustafsson
- **单位：** Swedish Defence Research Agency（FOI），Linköping, Sweden
- **预印本：** https://arxiv.org/abs/2610.13856 （v1，2026-10-09，cs.CV）
- **HTML：** https://arxiv.org/html/2610.13856v1
- **代码：** 论文给出 https://github.com/nrthby/event-action-mae ；截至 2026-10-12 核查仓库仅有标题 README（见 `sources/sites/event-action-mae-github.md`），未发布代码

## 论文要解决的问题

事件相机适合快速运动并保护隐私，但把事件累积成单帧表示在长而复杂的动作序列上判别力下降。论文用多帧表示加掩码自编码预训练改善长时段动作识别。研究对象是识别而非控制。

## 方法与训练信息

- **多帧表示：** 事件序列均分为 t 段，每段单独生成帧式表示并分别 token 化。表示有两种：CSTR（三通道：正事件平均时间戳、无极性二值事件直方图、负事件平均时间戳）与事件直方图（正/负事件计数分通道）；均以最大值归一化。
- **预训练：** 时空 MAE，ViT 骨干；默认 ViT-Tiny（深度 12、嵌入 192、3 头，约 5.5M 参数），另测 ViT-Small（21.7M）与 ViT-Base（85.8M）。输入每帧 224×224、patch 16、每帧 196 个 token、可学习位置编码；掩码比 75%。
- **token 选择：** 以 patch 内事件绝对值之和为活跃度，去掉固定比例最不活跃的 patch。
- **稀疏感知掩码：** 打分 s = u + c_s·v（u 均匀随机，v 归一化活跃度），得分最高者保持可见；c_s=0 退化为随机掩码。
- **多掩码词元：** n_t 个可学习掩码词元，由两层 MLP（隐层 2048）依据被掩 patch 的位置嵌入产生 softmax 权重混合初始解码词元；仅预训练使用，无推理开销。解码器为轻量版，具体配置原文未给。
- **优化：** AdamW，batch 128；预训练 1500 epoch（15 epoch 预热至 5e-4 后余弦衰减），微调 100 epoch、学习率 1e-3，从零训练 200 epoch；随机裁剪缩放，计算前滤除热像素；2×NVIDIA RTX A5000。
- **数据：** THU-EACT-50 CHL（50 类，无官方验证集，留出 20% 训练数据）与 DailyDVS-200（200 类）；预训练仅用训练集，测试集报告。

## 论文报告的结果

Top-1（%），线性探测 LP / 微调 FT：

- CSTR，4 帧，100% token：THU 52.0 / 65.8，DailyDVS 26.4 / 55.6；8 帧：56.8 / 66.4，31.0 / 61.7；8 帧 + 50% token：59.6 / 67.3，30.0 / 60.3。
- 用 DailyDVS-200 预训练后微调 THU：CSTR 4 帧 75.9，8 帧 **78.7**；8 帧 50% 78.5。
- 不预训练（4 帧）：THU FT 19.5，DailyDVS FT 20.7。
- 模型规模：8 帧 50%，ViT-Small 68.2 / 64.6，ViT-Base 68.6 / **66.9**（THU FT / DailyDVS FT）。
- 基线（THU / DailyDVS）：EV-ACT 58.5、EventMamba 59.4、EventBind+POKER 63.7、SMV-EAR 66.7 / 54.7、HoloEv-Net Base 69.9 / 54.6、Darkshake – / 52.0。摘要的 78.7% 与 66.9% 对应上面最佳项，78.7% 依赖 DailyDVS 预训练。
- 消融（THU LP，CSTR）：单帧 30.5 → 多帧 47.4 → 加稀疏掩码 51.7 → 加多掩码词元 52.0（直方图 24.1 → 47.1 → 49.4 → 52.4）。
- 掩码词元数（4 帧，LP）：1 个 51.7；2/4/8/12/16 个为 49.3/45.0/46.9/37.3/47.6；6 个 52.0。**正文称多于一个词元“持续改善”，表中并不清晰支持。**
- 帧数与动作时长（DailyDVS）：总体 1/4/8 帧为 33.1/55.6/61.7；Tie shoe laces（8.73 s）4 帧 79.2、8 帧 75.0，Sneeze（2.88 s）4 帧 50.0、8 帧 53.9。

## 复现与解释边界

- 仅两个数据集；THU-EACT-50 CHL 无官方验证划分。
- 稀疏偏置 c_s 和掩码词元数没有通用推荐值；多帧输入的计算复杂度需结合架构处理。
- 不同时长动作可能需要不同时间分辨率（论文提出作为未来工作）。
- 附录与 Figure 3/4 的数值未读到，只引用了文字描述。
- 代码待发布，无法复现。

## 一手来源

- <https://arxiv.org/abs/2610.13856>
- <https://arxiv.org/html/2610.13856v1>
