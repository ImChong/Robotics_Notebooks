# HumanX: Toward Agile and Generalizable Humanoid Interaction Skills from Human Videos

> 来源归档（ingest · 人形 RL 身体系统栈 42 篇 · 第 05/42）

- **标题：** HumanX: Toward Agile and Generalizable Humanoid Interaction Skills from Human Videos
- **类型：** paper
- **系统栈层：** 01 数据 · 重定向 · 遥操作
- **机构：** 香港科技大学；上海人工智能实验室
- **项目/论文链接：** <https://wyhuai.github.io/human-x/>
- **索引来源：** [具身智能研究室 · 42 篇 RL 运动控制长文](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)（<https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA>）
- **原始抓取：** [wechat_humanoid_rl_42_survey_2026-05-26.md](../raw/wechat_humanoid_rl_42_survey_2026-05-26.md)（Agent Reach + Camoufox）
- **入库日期：** 2026-05-26（2026-11-07 按 arXiv:2602.02473v2 全文复核）
- **原文：** <https://arxiv.org/abs/2602.02473>（v2，2026-10-09；CoRL 2026；项目页 <https://wyhuai.github.io/human-x/>），pdftotext 提取全文
- **核对范围：** 通读摘要、§1 引言、相关工作、XGen（锚点–物体配置、接触 / 非接触分段、数据增广）、XMimic（教师–学生架构、交互向量 IV 奖励、IV 终止、交互扰动、两种部署感知方案）、实验设置与仿真消融（表 1）、真机实验（表 2、图 8–11）、结论与局限，以及附录中统一奖励表与观测 / 奖励设置表（图 13）；表中数字列在 pdftotext 中错位处未引用；视频与代码未读、未运行。
- **通读：** 已完成
- **一句话说明：** HumanX 也从视频出发，但它关心的是 agile and generalizable humanoid interaction skills。它想把人类视频转成机器人可学习的交互技能，覆盖篮球、足球、羽毛球等任务。

## 核心摘录（策展，非全文）

- **在身体系统栈中的位置：** 01 数据 · 重定向 · 遥操作，编号 **05/42**。
- **公众号导读要点：** HumanX 也从视频出发，但它关心的是 agile and generalizable humanoid interaction skills。它想把人类视频转成机器人可学习的交互技能，覆盖篮球、足球、羽毛球等任务。
- **读者动作：** 方法细节以论文 PDF / 项目页为准；总框架见 [人形 RL 身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)。

## 对 wiki 的映射

- [paper-hrl-stack-05-humanx](../../wiki/entities/paper-hrl-stack-05-humanx.md)
- [humanoid-rl-motion-control-body-system-stack](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)
- [loco-manip-contact-category-01-contact-data](../../wiki/overview/loco-manip-contact-category-01-contact-data.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
- 姊妹篇 AMP 专题：[wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)

## 项目页与开源状态核查（2026-07-22）

- **论文/项目：** arXiv:2602.02473；项目页 <https://wyhuai.github.io/human-x/>。
- **代码：** 项目页 Code 链接指向 `github.com/wyhuai/humanx`，本次核查返回 404；`wyhuai/human-x` 为项目页资源仓，不视作可运行实现。
- **关键数字：** 五类域、10 个技能、zero-shot transfer 到 Unitree G1；human-robot passing 超 10 consecutive cycles；泛化成功率超过 prior methods 8×。
- **wiki 深化：** [paper-hrl-stack-05-humanx](../../wiki/entities/paper-hrl-stack-05-humanx.md)。

## 2026-11-07 通读复核补充（出自 v2 原文）

- **框架：** XGen（锚点为中心的数据合成：单目视频 → 人体运动重定向 → 接触 / 非接触分段 → 接触优化与物理仿真生成物体轨迹 → 几何缩放与轨迹变化增广，单条视频可扩成约 50 条交互片段）+ XMimic（交互优先的模仿学习）。
- **XMimic 三个设计：** ①交互向量（IV，身体点到物体中心的向量）模仿奖励；②IV 终止（交互失败时概率性提前终止）；③交互扰动（初始化时扰动机器人根位姿、关节角与物体位姿，训练中对机器人与处于接触阶段的物体施加随机外力）。
- **奖励（式 1–4）：** $r_t=r^{body}+r^{obj}+r^{rel/iv}+r^{c}+r^{reg}$；身体模仿项含位置 / 旋转 / 速度 / 关节项以及 **AMP（对抗运动先验）奖励**，用以增强平滑与自然度；$r^c$ 为接触模仿奖励，观测 / 奖励设置表列有 Contact Graph；正则含力矩、动作变化率、足滑移等。
- **教师–学生：** 对每种技能分别训练特权教师（本体 + 特权身体信息 + 物体状态，PPO）；学生在合并数据上训练，目标为 RL 奖励 + 对教师的行为克隆，并采用非对称 AC；学生可仅用本体感知（NEP 模式）或带 MoCap 物体观测（训练中模拟物体信号丢帧以零样本适应真实 MoCap 间断）。
- **部署：** Isaac Gym、单张 RTX 4090、16,384 并行环境、每策略 20,000 次迭代；策略与 MoCap 100 Hz、PD 1 kHz；Unitree G1。
- **结果：** 仿真中最终模型的泛化成功率（GSR）平均超过 80%，约为 HDMI 的 8 倍；真机 NEP 模式 10 次试验——投篮 8/10、运球 8/10、拾球 10/10、上篮 7/10、假动作转身后仰 9/10；MoCap 模式 50 次试验——传接球 41/50、反应格斗 37/50、搬货 43/50、足球踢球 42/50。
- **作者自述局限：** XGen 目前不支持手内重定向；不能处理可变形或可切割物体；未使用灵巧手；基于 MoCap 的感知系统难以感知多样物体且部署成本高。
