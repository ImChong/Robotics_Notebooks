# Spirit-v1.5 官网技术博客归档：Clean Data Is the Enemy of Great Robot Foundation Models

> 来源归档（blog / 千寻智能官方技术博客）

- **标题：** Spirit-v1.5: Clean Data Is the Enemy of Great Robot Foundation Models
- **发布方：** 千寻智能（Spirit AI）
- **原文链接：** <https://www.spirit-ai.com/blog/spirit-v1-5>（英文镜像：<https://www.spirit-ai.com/en/blog/spirit-v1-5>）
- **发布时间：** 2026-01-11（博客 Figure 4 与正文口径「As of Jan 11, 2026」）
- **核心代码与权重：**
  - 代码仓库：<https://github.com/Spirit-AI-Team/spirit-v1.5>（MIT License；详归档见 [spirit-v1_5.md](../repos/spirit-v1_5.md)）
  - HF 模型权重：<https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5> 与 `Spirit-v1.5-for-RoboChallenge-move-objects-into-box`（Apache-2.0）
- **核对范围：** 读取 `www.spirit-ai.com/blog/spirit-v1-5` 渲染脚本完整正文，核对 Figure 1–4 图注、消融实验设定、RoboChallenge Table30 评测口径、开源资源链接与文末参考文献 `[1]`–`[7]`。
- **通读：** 已完成
- **沉淀到 wiki：** [`wiki/entities/cn-os-spirit-v1-5.md`](../../wiki/entities/cn-os-spirit-v1-5.md) · [`wiki/entities/spirit-ai.md`](../../wiki/entities/spirit-ai.md)

---

## 核心论点与「Clean Data 诅咒」

1. **传统精选数据集（Curated / Clean Data）的瓶颈：** 传统 VLA（如基于 Open X-Embodiment、AgiBot World、RoboCOIN 等高度脚本化演示的数据）往往要求任务固定、物体位置可控、只保留单条成功线性轨迹并丢弃偏离动作。博客认为这导致 **Low Diversity**（孤立任务岛，缺乏失败恢复、滑脱纠偏与复杂背景泛化）与 **Limited Scalability**（每增一任务都需工程师写脚本并重做质量控制，边际人工与管理成本难下降）。
2. **新范式：Open-Ended Diverse Collection（开放式目标驱动多样采集）：**
   - 只给一条约束：**「做一件有用的事（do something useful）」**，取消僵化脚本，允许采集员在一个连续会话里即兴串联多个子任务、尝试—失败—重试回环与自然切换。
   - **Figure 1（子任务路径图）：** 对比单向线性成功路径与交织失败重试循环及任务重叠转换的网络化轨迹。
   - 博客演示了 4 段 4× 速典型片段：调酒准备、笔记本素描、积木堆叠、玩具钓鱼，连续包含抓握、拧旋、插接与双手协同等原子技能。
   - **采集规模化收益（博客自报）：** 单人有效采集小时数提升 **200%**，研究员盯盘/介入需求下降 **60%**。

---

## 控制变量消融与 Scaling 发现

博客针对「高度无序的 Diverse Data 能否收敛并优于同体积脚本数据」设计了对照实验（同等预训练与微调数据量，微调到两类预训练均未见过的新任务）：

- **Transfer Efficiency（Figure 2，迁移效率）：** Group B（Diverse Collection 预训练）在下游未见任务的微调收敛速度与验证误差上持续优于 Group A（任务脚本演示预训练）；到达相同微调验证水平所需迭代步数 **减少 40%**。
- **Scaling Efficacy（Figure 3，扩展律）：** 将 Diverse Collection 数据体积从 Small（S）扩展到 Large（L）时，未见新任务的微调验证误差呈一致单调下降。
- **结论定位：** 博客主张「任务/经验多样性比单任务演示数量更重要」，且高多样性、松控制的物理交互经验可沉淀物理常识。*注：博客未公开 Group A/B 的精确小时数、S→L 的精确数据量比及超参数表。*

---

## RoboChallenge Table30 评测（截至 2026-01-11）

- **基准说明：** 引用 `RoboChallenge` 真实机器人桌面操作基准（Table30，覆盖 30 项日常桌面任务，如插花 `arrange_flowers`、三明治制作 `make_vegetarian_sandwich`、插网线 `plug_in_network_cable` 等，跨越 Franka、Arx5、UR5 单臂与 ALOHA 双臂多机位设置）。
- **测试协议：** 所有参评基础模型在相同 Table30 微调数据上微调后评测，比较基座模型的迁移与泛化上限。
- **博客公布结果：** 截至 **2026-01-11**，Spirit-v1.5 在 RoboChallenge Table30 位列 **#1**，显著超越 `π₀.₅`（博客 Figure 4）。

---

## 文末参考文献与边界

- 引用 `[1]` π0.5（PI）、`[2]` GEN-0（Generalist AI）、`[3]` ACT-1（Sunday Robotics）、`[4]` OXE、`[5]` AgiBot World Colosseo、`[6]` RoboCOIN、`[7]` RoboChallenge。
- **未公开信息边界：** 博客正文未披露预训练精确数据集成分比例、预训练代码或全量 Diverse Collection 原始语料下载入口；公开仓库主要覆盖微调/推理模型代码与 RoboChallenge 接入层，具体实现核查见 [spirit-v1_5.md](../repos/spirit-v1_5.md)。
