# Semantics-to-Contact（arXiv:2611.07532）

> 来源归档（paper）

- **标题：** Semantics-to-Contact: A Stagewise Framework for Robust Contact-Rich Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.07532>
- **PDF：** <https://arxiv.org/pdf/2611.07532v1>（v1，2026-11-05 提交，cs.RO；NeurIPS 2026 投稿格式）
- **项目页：** <https://edem-ai.github.io/S2C/>
- **代码：** 未核实到公开仓库（见下「开源状态」）
- **机构（论文首页）：** 香港中文大学（深圳）；DexForce Co., Ltd.；深圳回环区研究院（Shenzhen Loop Area Institute）
- **作者：** Guanren Qiao, Ruixiang Ouyang, Sheng Xu, Yueci Deng, Yunxin Tai, Kui Jia, Guiliang Liu（通讯）
- **入库日期：** 2026-11-09
- **核对范围：** 下载 arXiv v1 PDF，核对首页标题与编号一致后，用 pdftotext 逐段读完正文第 1–6 节（含 Table 1–6、Algorithm 1、Fig. 1–2 的图注与文字）、参考文献、附录 A（六个任务的设置与 Table 7–12 训练细节）、附录 B（运行设置）、附录 C（视觉增强细节，式 12）、附录 D（VLM 提示词）和 NeurIPS 检查清单（逐项读到最后一项）。图像本身（Fig. 1–9 的像素内容）只读了图注与图中文字，没有逐图细看；项目页只静态读了页面文字（核对了 94.17% 和 78.75% 两个汇总数及其试验数），未观看视频，未下载或运行任何代码。
- **通读：** 已完成
- **一句话说明：** 两阶段真机 RL：先用 VLM（MolmoPoint-8B）在第三视角 RGB-D 上指出目标点并运动规划到接触前区域，再用「ACT 基策略 + 人在回路残差 SAC」在腕部相机、位姿与力/力矩观测上学接触精修；训练时用 YOLOE 掩码做以物体为中心的背景随机化。

## 开源状态

- 论文 NeurIPS 检查清单第 5 项答 `[No]`：「After acceptance, we will provide the code and data in the future」（附录检查清单）。
- 项目页（2026-11-09 静态读取）有摘要、方法说明、视频与结果表、BibTeX，未发现 GitHub / 数据集链接；按步骤 2.5 记为「宣称将开源，截至入库日未列仓库」。

## 核心摘录

1. **问题与动机（§1）：** 真机 RL 的视觉观测易受背景纹理和光照影响，训练场景里学到的策略难迁到新环境；仿真域随机化对接触丰富任务有大 sim-to-real 差距（文中举例紧配合间隙可小于 0.1 mm）。
2. **两阶段（§4、Fig. 1）：** 语义聚焦（VLM 出 2D 点，经深度与手眼标定变 3D 目标，运动规划到达）+ 接触精修（人在回路真机残差 RL）。
3. **残差 RL（§4.2、式 9–11、Algorithm 1）：** ACT 基策略用 BC 预训练；残差策略用 SAC 风格 off-policy 更新，在线缓冲与离线（含人工干预）缓冲各取一半 batch。
4. **以物体为中心的视觉增强（§4.1、式 6–8）：** YOLOE 用语言提示分出机械臂、夹爪与被操作物体的前景掩码，背景换成随机颜色 / 纹理 / 干扰物，前景再做颜色抖动和亮度变化。
5. **结果（§5、Table 1–6）：** 原训练设置 94.17%（113/120）；四组视觉+位置扰动下平均 78.75%；HIL-SERL 与 HIL-ConRFT 在前三组扰动下全 0。

## 对 wiki 的映射

- [paper-semantics-to-contact](../../wiki/entities/paper-semantics-to-contact.md)
