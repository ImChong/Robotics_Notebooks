# MultiMediate '26：跨领域、年龄与文化的人-机/人-人互动投入度估计挑战（arXiv:2611.09913）

> 来源归档（paper）

- **标题：** Engagement Estimation in Human-Robot and Human-Human Interactions across Domain, Age and Culture: The MultiMediate '26 Challenge
- **类型：** 原始论文（挑战赛概述；ACM MM '26，2026-11-10 至 14 日，巴西里约；DOI 10.1145/3767308.3837718；arXiv:2611.09913v1，cs.HC，论文首页 arXiv 戳记日期为 25 Aug 2026）
- **作者与机构（论文首页）：** Marius Funk、Daksitha Withanage Don、Elisabeth André（奥格斯堡大学）；Michal Balazia、François Brémond（INRIA 蔚蓝海岸大学）；Shogo Okada（JAIST）；Victor Oei、Andreas Bulling（斯图加特大学）；Philipp Müller（马克斯·普朗克智能系统研究所）
- **arXiv：** <https://arxiv.org/abs/2611.09913>
- **PDF：** <https://arxiv.org/pdf/2611.09913v1>
- **代码：** https://github.com/hcmlab/MultiMediate_26
- **项目页：** <https://multimediate-challenge.org/Dataset/>
- **核查日期：** 2026-11-10
- **核对范围：** 下载 arXiv PDF v1（共 6 页，首页标题与编号已核对一致），通读摘要、§1 引言、§2 挑战描述（任务定义、数据集、训练数据政策）、§3 特征提取、§4 基线、§5 结果（表 1–4 逐表核对）、§6 结论与参考文献列表；论文无附录。图 1 仅为数据集示例照片，只读了图注。GitHub README 只读了前部（基线结果表、CDD_G 说明开头），未读其余部分与代码；项目页仅确认可访问（HTTP 200），页面内容未读；未运行任何代码。获奖团队的方法细节只来自本文第 5.2 节的转述，未读其各自论文。
- **通读：** 已完成

## 核心摘录（均出自论文）

1. **定位：** 此前 MultiMediate 聚焦成人对话里「逐时刻连续投入度数值」的跨文化估计（摘要、§1）。今年新增 PInSoRo 数据集（英语儿童-儿童、儿童-机器人自由游戏），按「任务投入」「社交投入」做类别标注，以扩展投入度类型、年龄段与互动类型（摘要、§1）。
2. **数据集（§2.2，表 1）：** NoXi（英/法/德：训练 38、验证 10、测试 16 场）、NoXi Add.（印尼语 4、西班牙语 4、意大利语 2、阿拉伯语 2，仅测试）、NoXi-J（日语、汉语）、MPIIGI（德语，验证 6、测试 6，无训练集）、PInSoRo 儿童-儿童 CC（20/7/6）、儿童-机器人 CR（12/5/6）；共涉及 9 种语言。PInSoRo 平均每场约 39 分钟，二人、成人监督、围绕触摸屏桌（§2.2）。
3. **PInSoRo 标签（§2.2）：** 任务投入 4 类（goal-oriented、aimless、adult-seeking、no play）；社交投入 5 类（solitary、onlooker、parallel、associative、cooperative，取自 Parten 游戏阶段）。验证/测试集每场至少 2 名标注者，另为 CR 补标 10 场；测试只在所有标注者一致的帧上评估。
4. **指标（§2.1，式 1）：** 连续标签用 CCC，类别标签用 Cohen's kappa，总分 Global = 2/3·CCC_Avg + 1/3·kappa_Avg。
5. **基线（§3–§5）：** 特征由 DISCOVER 管线提取；PInSoRo 不公开视频与转写（隐私）。基线为 3 层 ReLU MLP（沿用 MM'25，KerasTuner Hyperband 调参，Adam）。回归基线选 eGeMAPS v2：CCC_Avg 0.45（NoXi 0.55、NoXi-J 0.31、MPIIGI 0.45、NoXi Add. 0.49，表 2）；类别基线 CC 取 XLM-RoBERTa、CR 取 Swin Transformer，kappa_Avg 0.16（表 3、4）；Global 0.36。作者说明所选基线「应视为该架构的上界」（§5.1）。
6. **挑战结果（§5.2，表 4）：** 13 个参赛组。HFUT-LMC（SEST，多模态 LLM）Global 0.65 夺冠，kappa_Avg 0.52；USTC_IAT_United 的 CCC_Avg 最高 0.74；NUDT_LDM-AF NoXi 0.82、NoXi Add. 0.80；mltac NoXi-J 0.66（正文）。作者称 NoXi 较去年冠军仅提高 0.03，「进展似乎趋于平台」，类别子任务表现差异大、提升空间显著。
7. **其他任务（§5.2）：** 身体行为识别准确率 0.65 → 0.67，眼神接触检测 0.82 → 0.83，反馈信号检测只有 1 份提交且未改进。

## 对 wiki 的映射

- [paper-multimediate-26-engagement-estimation](../../wiki/entities/paper-multimediate-26-engagement-estimation.md)
