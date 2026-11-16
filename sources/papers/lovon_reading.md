# LOVON: Legged Open-Vocabulary Object Navigator

> 来源归档（ingest · Robot Learning Paper Notebooks 深读笔记 + arXiv 全文通读）

- **标题：** LOVON: Legged Open-Vocabulary Object Navigator
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/LOVON__Legged_Open-Vocabulary_Object_Navigator/LOVON__Legged_Open-Vocabulary_Object_Navigator.html>
- **分类：** 08_Navigation
- **arXiv：** <https://arxiv.org/abs/2507.06747>（2025-07-09）
- **项目主页：** <https://daojiepeng.github.io/LOVON/>
- **官方代码：** <https://github.com/DaojiePENG/LOVON>
- **核对范围：** 全文通读论文文本 §I–§VIII、Fig. 1–6、Table I–IV（Gym-Unreal 四场景对比、模型/数据规模消融、Unitree Go2 实机消融、B2 与 H1-2 跨本体真机部署）。
- **通读：** 已完成
- **入库日期：** 2026-07-10（原文通读复核：2026-11-14）
- **一句话说明：** 针对足式机器人开放世界长时程目标导航中的词汇受限、多步规划与行走运动模糊/盲区丢失问题，LOVON 提出 LLM 分层规划（DeepSeek-R1）+ IOE 目标提取 + YOLO 开放词汇检测 + 拉普拉斯方差滤波 + 3.30M 参数 L2MM 运动预测及四状态恢复执行框架，可在 Unitree Go2 / B2 / H1-2 上跨本体部署。

## 核心摘录（策展与核对摘要）

- **核心模块参数**：Laplacian 方差阈值 $T_{\text{blur}}=150$ + 滑动平均滤波（MAF）带来 +25% 有效清晰帧；L2MM（`LOVON-Base`）为 3.30M 参数 4 层 Transformer 编码器，`[SEP]` 拼接输入并设 MSE($\beta=10$)+CE 联合损失。
- **合成数据与仿真评测（Table II & III）**：1M 合成轨迹（8:2 划分）在 i9-12900KF 上 <15 min 生成；Gym-Unreal 4 场景平均达到 499.75 EL / 1.00 SR（总训练仅 1.5 h，对照 TrackVLA 为 360 h）。
- **实机消融（Table IV，Go2 6 m 找背包）**：引入四状态执行逻辑与模糊滤波后，平均搜索时间 $T_s$ 从 178 s 降至 32.57 s，搜索次数 $N_s$ 从 5.95 降至 1.00。

## 对 wiki 的映射

- [paper-notebook-lovon-legged-open-vocabulary-object-navigator](../../wiki/entities/paper-notebook-lovon-legged-open-vocabulary-object-navigator.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/LOVON__Legged_Open-Vocabulary_Object_Navigator/LOVON__Legged_Open-Vocabulary_Object_Navigator.html>
- 论文：<https://arxiv.org/abs/2507.06747>
