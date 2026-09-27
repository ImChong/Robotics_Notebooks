# ut-austin-rpl.github.io/Harmon（Harmon 项目页）

- **标题：** Harmon: Whole-Body Motion Generation of Humanoid Robots from Language Descriptions
- **类型：** site / project-page
- **URL：** <https://ut-austin-rpl.github.io/Harmon/>
- **arXiv：** <https://arxiv.org/abs/2410.12773>
- **机构：** 德克萨斯大学奥斯汀分校（The University of Texas at Austin）；英伟达研究院（NVIDIA Research）
- **作者：** Zhenyu Jiang*、Yuqi Xie*、Jinhan Li、Ye Yuan、Yifeng Zhu、Yuke Zhu（* equal contribution）
- **入库日期：** 2026-09-27

## 一句话摘要

语言描述 → **PhysDiff** 人体扩散先验 → IK 重定向到人形 → **VLM** 渲染迭代编辑（补头/指、修手臂语义）→ 上下身解耦执行；GR1-T1/T2 真机与仿真视频。

## 开源状态（步骤 2.5，2026-09-27）

| 资源 | 状态 |
|------|------|
| 项目页 / arXiv | **已发布** |
| 训练/推理代码 | **未列链接**（页上仅 Arxiv + Twitter Summary） |

**结论：** **未开源** — 复现以 arXiv 与项目页视频/方法图为准。

## 公开信息要点

- **管线（页首 Method）：** 语言 → 人体 motion → IK retarget → VLM 从初始描述抽 finger/head 子描述并生成 → 对渲染序列做迭代评估/修正 → 全身 motion。
- **真机：** Fourier GR1-T1（黑）与 GR1-T2（银）；手语分段（GPT-4 切句 + 逐段 motion）与自由描述跟做。
- **失败案例：** 高频拍手 VLM 难修；语义 T 形手势人体先验错误且偏离过大；腕部朝向缺 editing primitive。

## 关联资料

- 论文归档：[`harmon_arxiv_2410_12773.md`](../papers/harmon_arxiv_2410_12773.md)
- 沉淀实体：[`wiki/entities/paper-loco-manip-161-097-harmon.md`](../../wiki/entities/paper-loco-manip-161-097-harmon.md)
