# DigitCode: Symbolic Tokenization of Hand Motion by Anatomical Units（arXiv:2608.03127）

> 来源归档（ingest）

- **标题：** DigitCode: Symbolic Tokenization of Hand Motion by Anatomical Units
- **缩写 / 框架：** **DigitCode**（A / F / H）；评测床 **HandTok**；基线 **HL-26**（Hand Labanotation）
- **类型：** paper / hand-motion / tokenization / retargeting
- **arXiv：** <https://arxiv.org/abs/2608.03127>
- **项目页：** <https://digitcode-demo.github.io/>（归档见 [`sources/sites/digitcode-demo.md`](../sites/digitcode-demo.md)）
- **会议：** AAAI 2027（项目页：匿名审稿中）
- **作者：** Haoyu Gu、Haotian Lu、Jingrun Du、Xiao-Ping Zhang（通讯）
- **入库日期：** 2026-08-15
- **一句话说明：** 在 Hand Labanotation 的 \(T\times 40\) 骨向网格上，把 token 跨度改到骨 / 指 / 整手层级；杠杆在解剖单元而非量化器家族。DigitCode-H 把 held-out 角误差从 **14.71° 降到 3.26°**（近同码率），并给出无训练的逐指编辑、畸形手修复与机器人重定向接口。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-15）：** [digitcode-demo.github.io](https://digitcode-demo.github.io/) 有交互演示与结果表；文案写 **PDF 与 HandTok「审稿结束后再挂」**。
- **结论：** **演示页已发布，代码 / 评测床宣称将开源。** 源码运行时序图标 **不适用**。

## 摘录 1：问题与主张

- 连续表示（关节角 / MANO）准但不可索引、不可局部编辑、不标解剖合法。
- HL-26 已证明手可符号化，但单位被记谱法钉在「每骨一符」。
- **主张：** 固定单元时，无训练 k-means 与强学习量化器可互换；**改单元**才移动率–失真前沿。

## 摘录 2：三步改码

| 变体 | 单元 | 操作 | InterHand2.6M 角误差 |
|------|------|------|----------------------|
| HL-26 | 骨 | 固定 26 向立方体字母 | 14.71° / 4.70 bit |
| DigitCode-A | 骨 | 球面 k-means 适配字母 | 8.45° |
| DigitCode-F | 指 | 四骨联合量化 | **5.50° / 2.0 bit** |
| DigitCode-H | 指+骨残差 | 粗指码 + 每骨残差 | **3.26° / 4.75 bit**（可到 1.86° / 6.75 bit） |

手指内互信息 39% vs 跨指 29%；时间轴 79% token 不变。随机重分组同块长大约 +2.4°，说明收益来自解剖缝而非更大 block。

## 摘录 3：下游与接口

- 动力学读骨、交互读指、身份读整手；噪声下粗 HL-26 更稳。
- 逐指 token：无训练检测畸形指（AUC 0.953 / 0.823）；640 次 IK 编成查找表，Allegro 流式重定向约快 3 个数量级、误差约 0.7 mm。
- 发布 HandTok，按单元对齐比较 tokenizer。

**对 wiki 的映射：** [`wiki/entities/paper-digitcode.md`](../../wiki/entities/paper-digitcode.md)；交叉 [UHAS](../../wiki/methods/uhas-unified-hand-action-space.md)、[WiLoR](../../wiki/methods/wilor.md)、[灵巧手运动学](../../wiki/concepts/dexterous-kinematics.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（HandTok 待发布）
