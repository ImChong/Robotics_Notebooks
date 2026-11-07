# Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data

> 来源归档（ingest · awesome-bfm-papers 第 09/41）

- **标题：** Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data
- **类型：** paper
- **BFM 分类：** 02 Goal-conditioned 学习（[awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers)）
- **出处：** 2025 · arXiv
- **论文链接：** <https://arxiv.org/abs/2511.17373>
- **代码/项目：** <https://github.com/OpenDriveLab/AMS>
- **索引来源：** [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) · [具身智能研究室 BFM 41 篇编译](../blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md)（<https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g>）
- **入库日期：** 2026-05-26（2026-11-07 按 arXiv v3 全文通读复核）
- **版本：** arXiv:2511.17373v3（2026-03-03），pdftotext 提取全文
- **核对范围：** 通读摘要、§I 引言、§II 相关工作、§III 方法（问题设定、算法 1 受控平衡动作生成、混合奖励、自适应采样式 (2)–(6)、自适应奖励塑形式 (7)）、§IV 实验（设置与指标、与 OmniH2O / HuB 对比、四组消融、OOD 评测、真机部署与遥操作）、§V 结论与局限；表 I / II 的数值列在 pdftotext 中错位，未引用具体数字；参考文献仅浏览；项目页视频与代码未读、未运行。
- **通读：** 已完成
- **一句话说明：** 异构数据下敏捷与稳定权衡；MoCap/仿真/视频等混合是 BFM 数据常态。

## 核心摘录（策展，非全文）

- **在 BFM 技术地图中的位置：** 02 Goal-conditioned 学习，编号 **09/41**。
- **公众号导读要点：** 异构数据下敏捷与稳定权衡；MoCap/仿真/视频等混合是 BFM 数据常态。
- **读者动作：** 方法细节以论文 PDF / 项目页为准；谱系对照见 [BFM 41 篇技术地图](../../wiki/overview/bfm-41-papers-technology-map.md) 与 [Behavior Foundation Model](../../wiki/concepts/behavior-foundation-model.md)。

## 对 wiki 的映射

  - [ams](../../wiki/methods/ams.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2511.17373>
- 策展列表：<https://github.com/friedrichyuan/awesome-bfm-papers>
- 微信公众号编译：[wechat_embodied_ai_lab_bfm_41_papers_survey.md](../blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md)

## 2026-11-07 通读复核补充（出自 v3 原文）

- **混合奖励的触发条件：** 通用跟踪奖励用于**所有**数据；平衡先验奖励（质心对齐、足接触一致性）**只**用于合成平衡动作，人类动捕数据不加。论文并没有按「低成功率」或「脚滑严重」去动态开关平衡奖励。
- **合成平衡动作（算法 1）：** 先采样支撑脚、摆动脚目标位姿、骨盆高度与下肢偏自然姿态的初始构型（上身关节随机化）；用 SE(3) 插值构造支撑脚 / 摆动脚 / 骨盆参考轨迹；两阶段批量轨迹优化——第一阶段含位姿跟踪、软关节限位、休息姿态正则与时间平滑，第二阶段加质心落在支撑矩形内的惩罚 $\lambda_{bal}\sum\max(0,d_t-\epsilon)$，用 Levenberg–Marquardt 求解，仅 $\max_t d_t\le\epsilon$ 的轨迹被接受。
- **自适应采样：** 按执行失败、平均关节位置误差（MPJPE）与最大关节误差评估；成功动作以误差的 P75 / P25 分位为差 / 好阈值，差者概率乘以提升因子（$\beta>1$）、好者乘以衰减因子（$\alpha<1$）、失败动作乘以失败提升因子；设最小采样概率 $p_{min}=\lambda/N$ 防止动作被完全忽略。自适应奖励塑形按动作与身体部位维护独立的误差容忍 $\sigma$，用指数滑动平均更新 $\sigma_{new}=(1-\alpha)\sigma+\alpha\,err$（承接 PBHC）。
- **数据与平台：** AMASS + LAFAN1 过滤后超过 8,000 条动作，另有 10,000 条合成平衡动作；Isaac Gym 训练；真机 Unitree G1（23 自由度，1.3 m，约 35 kg）；动作 23 维关节位置目标 + PD；教师（特权）→ 学生（可部署）蒸馏。
- **指标：** 成功率（平均偏离参考超过 0.5 m 记失败）、全局 / 根相对 MPJPE、接触失配率、支撑脚滑移速度。
- **作者自述局限：** 缺乏精确的末端执行器控制，难以用于操作与接触丰富任务；RGB 位姿估计遥操作的全局运动噪声大，使敏捷移动难以遥控。
