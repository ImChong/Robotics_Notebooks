# STARE（arXiv:2611.06089）

> 来源归档（paper）

- **标题：** STARE: Scene Text-Aware multimodal Retrieval for Everyday objects
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.06089>
- **PDF：** <https://arxiv.org/pdf/2611.06089>
- **项目页：** <https://stare-projectpage.pages.dev/>
- **代码：** <https://github.com/keio-smilab24/STARE>
- **入库日期：** 2026-11-06
- **线索来源：** [只在虚拟房间练习，人形机器人走进陌生房间能做什么？](../blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)（具身智能小站，2026-11-06）
- **核对范围：** 用 pdftotext 抽取 arXiv v1 PDF 全文（正文 §1–§5 与参考文献 [1]–[42]）后通读，含方法、实验、消融与硬件实验。arXiv 版本不含补充材料（正文中 "Section ??" 指向的损失函数、评测指标细节、失败案例、误差分析、MFIE 消融、recall@20 均未见），这些未核对；图 1–4 为图片，仅读了图注。
- **通读：** 已完成
- **一句话说明：** STARE 把指令里的品牌或品名与场景文字、物体外观一起检索，再让机器人去抓。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-06）：仓库可达（git ls-remote）。项目页 HTTP 200。

## 核心摘录

1. **任务 STMR（§1、§3.1）。** 机器人预探索时收集图像（可含场景文字），给定自由形式的操作或导航指令，从图像集合中检索并排序含目标物体的图像。方法不假设场景文字存在，在无文字的 LTRRIE 上同样适用。
2. **三模块（§3.2）。** MFIE（指令编码）、STVE（场景文字感知图像编码）、STRR（场景文字重排）。预探索阶段对候选图像做 OCR、CVP、STVE 并存储嵌入；查询时只做指令侧处理与 STRR。
3. **MFIE（§3.3，式 2）。** 一次 LLM 调用同时完成查询扩展（把品名改写成外观描述）与命名实体抽取；扩展前后的文本分别过 Stella 与 CLIP 文本编码器，拼接后经 SSM（Mamba）得到 h_txt。
4. **STVE 与 CVP（§3.4，式 3）。** 图像用多个编码器（CLIP、DINOv2）做重叠分块的细粒度特征；CVP 用第三方 OCR 的框，在框顶叠加非字母数字的跨语言标记（实验用片假名），连同 OCR 文本交给 MLLM 生成叙述式图像描述，再经语言编码器得 l_N，与 DINOv2 中间层视觉特征做交叉注意力，最后经 SSM 得 h_img。
5. **STRR（§3.5，式 4）。** 相似度 s = cos(h_txt, h_img) + λρ_i，ρ_i 为指令命名实体 token 与图像场景文字 token 的交集大小，除以候选集中的最大交集（最大为 0 时取 0）；λ 可学习。文字做大小写与符号归一化后做字符串匹配。
6. **基准（§4.1）。** GoGetIt 共 27,280 条样本、3,657 张含场景文字的图像（RefText 子集 23,323 条，新标注的 Instruction 子集 3,957 条、129 位标注者）；TextCaps-test 400 条指令（10 位标注者）、105 张图像；另用无场景文字的 LTRRIE（5,108 条指令、3,569 张图像）。作者估计 RefText 约 50% 可能与 BLIP-2、Long-CLIP 预训练数据重叠，故这两者与 RelaX-Former 的 GoGetIt 分数未报告。
7. **训练与效率（§4.1–4.2，表 1）。** AdamW，学习率 1e-4，批大小 128，20 个 epoch；约 4000 万可训练参数、50 亿次乘加；单张 RTX 4090 训练约 1.2 小时；单条指令对 100 个预存嵌入的相似度计算约 39.8 ms。MFIE 与 STVE 用 GPT-4o，OCR 用 Google Cloud Vision API；损失为带 STRR 分数的 Double Relaxed Contrastive loss。
8. **离线对比（表 2，recall@10）。** STARE 在 GoGetIt-RefText / GoGetIt-Instruction / TextCaps-test / LTRRIE 上为 94.4 / 92.8 / 95.4 / 86.6，分别比最佳基线（81.2 / 84.6 / 91.8 / 84.3）高 13.2 / 8.2 / 3.6 / 2.3 个点，p < 0.01；基线含 CLIP、Long-CLIP、BLIP-2、BEiT-3、Meta CLIP 2、NLMap、ReMEmbR、RelaX-Former、ViSTA。
9. **消融（表 3，GoGetIt-Instruction recall@10）。** 去掉 STRR 为 85.9（低 6.9），STRR 改用整条指令而非命名实体为 87.7（低 5.1）；CVP 改用数字、英文、希腊字母分别为 92.3 / 92.2 / 92.1（片假名 92.8）；去掉 CVP 为 91.9，去掉全部 OCR 相关特征（OCR、CVP、STRR）为 82.3；去掉分块 92.2、标准分块 92.4。
10. **真机（§4.6，表 4）。** 家用服务机器人 DSR（Point Grey Chameleon 头部相机），YCB 物体加带印刷标签的日用品，零样本；12 种布置共 100 个回合（每种 5–10 个），预探索沿 15 个路点拍照。成功判据：目标图像进入前五且成功抓取。STARE 的 recall@5 为 88%、SR 为 80%，比最佳基线（ViSTA 81% / 73%）高 7 个点。查询延迟含 LLM 调用平均 2.0 s，对比 GPT-4o 端到端 21.3 s；显存查询时 7.23 GB、嵌入时最多 7.40 GB。
11. **作者陈述的边界。** 现有 VLA 难以处理数千张输入图像（延迟大、超 API token 上限）；OCR 受遮挡与朝向影响；仅用 MLLM 会幻觉出不存在的文字，因此用 OCR 文本与 CVP 约束。

## 对 wiki 的映射

- [paper-stare](../../wiki/entities/paper-stare.md)
- [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-06-20-papers-technology-map.md)
