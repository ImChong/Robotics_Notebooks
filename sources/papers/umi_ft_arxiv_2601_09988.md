# In-the-Wild Compliant Manipulation with UMI-FT

> 来源归档（ingest · REALab 14 篇盘点）

- **标题：** In-the-Wild Compliant Manipulation with UMI-FT
- **类型：** paper
- **状态：** ICRA 2026
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2601.09988>
  - 项目页：https://umi-ft.github.io/
- **代码：** https://github.com/real-stanford/UMI-FT
- **作者：** Hojung Choi, Yifan Hou, Chuer Pan, Seongheon Hong, Austin Patel, Xiaomeng Xu, Mark R. Cutkosky, Shuran Song
- **机构：** Stanford University
- **入库日期：** 2026-08-18
- **一句话说明：** UMI-FT（ICRA 2026）：指端 CoinFT 六维力+RGB/深度手持采集；自适应顺应策略预测位姿/抓取力/刚度；白板擦拭等三任务优于纯视觉；硬件软件已开源。

## 核心论文摘录（MVP）

### 问题与贡献

- **摘录要点：** 在 UMI 手持夹爪每指安装紧凑六维力传感器，野外采集多模态示范并训练自适应顺应策略，规模化学习力敏感操作。
- **对 wiki 的映射：**
  - [wiki/entities/paper-umi-ft.md](../../wiki/entities/paper-umi-ft.md)

### 方法与结果（归纳）

- **方法：** CoinFT 指端传感器 + iPhone RGB/深度；策略融合视觉、深度、F/T、本体感觉，输出位姿目标与抓取力/刚度给顺应控制器。
- **评测：** 白板擦拭、插灯泡、穿西葫芦三任务；相对无顺应/无力传感基线显著更稳。

## 当前提炼状态

- [x] 公众号盘点 + arXiv/项目页交叉核对
- [x] wiki 实体页：`wiki/entities/paper-umi-ft.md`
