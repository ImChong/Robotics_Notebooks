# Humanoid Footstep Planning using Graphs of Convex Sets

- **类型：** paper
- **作者：** Hyunseok Kim, George Mesesan, Robert Schuller, Jinoh Lee, Hae-Won Park
- **机构：** 韩国科学技术院 KAIST（DRCD 实验室官方成员页列有 Hyunseok Kim 和 Hae-Won Park；其余作者所属机构待全文核实）
- **期刊：** IEEE Robotics and Automation Letters（RA-L），2026
- **DOI：** 10.1109/LRA.2026.3746865
- **论文官方入口：** https://doi.org/10.1109/LRA.2026.3746865
- **实验室：** https://www.dynamicrobot.kaist.ac.kr/
- **实验室成员页（官方）：** https://www.dynamicrobot.kaist.ac.kr/contact
- **摘要索引（非论文原文）：** https://eurekamag.com/research/112/001/112001467.php
- **项目页：** 用户给出的 hyuns-kim.gith… 链接已截断，未取得可核实的完整 URL
- **代码：** 用户给出的 github.com/Hyu… 链接已截断，尚未核实仓库、许可证和训练／部署入口；不等同于“未开源”
- **核查日期：** 2026-11-10
- **核对范围：** 论文标题、作者、期刊、DOI 及第三方索引转载的摘要；KAIST 实验室官网与成员页（列明 Hyunseok Kim 博士候选人、Hae-Won Park 教授）。IEEE 全文未获取，方法正文、公式、实验表、补充材料及源码均未通读。
- **通读：** 未完成

## 核心摘录（仅依据已核对的摘要）

1. 问题：人形机器人步位规划需要在时间约束内同时决定离散步序和连续落脚位置，并满足地形可行性与运动学可达性。
2. 建模：先用非凸二次约束二次规划（QCQP）表达地形与可达性约束，再构造带运动模式语义的图；节点对应运动学可行凸区域，边对应可行转换。
3. 求解：将图结构与 QCQP 纳入 Graphs of Convex Sets（GCS）框架，使用半定规划（SDP）获得凸松弛；摘要声明不需要预先给定离散路径与连续变量初值。
4. 验证：摘要报告数值研究和力矩控制人形机器人 TORO 的节奏式定时踏步实验，并描述实时计算预算内的规划能力。**无可信的定量表格可供摘录；不填具体耗时、成功率或重规划频率。**
5. 局限：目前未核对全文，不足以判断凸松弛紧性、具体目标函数、步态图构造、路径恢复过程、与混合整数求解器对比的实验数值或机器人控制接口。

## 开源与项目页核查

用户提供的项目页与 GitHub URL 均为截断文本，公开检索未发现可以可靠与本文 DOI 绑定的完整官方地址。**开源状态为“待核实”，不是“确认未开源”。** 完整项目页、具体代码模块及数据下载链接均留待补充。

## 全文追索记录（2026-11-10，未完成通读）

- **IEEE 出版入口：** [DOI](https://doi.org/10.1109/LRA.2026.3746865) 和 [IEEE Xplore](https://ieeexplore.ieee.org/) 已尝试，但当前检索环境无法读取本论文正文／PDF；不把链接可解析与读到全文混为一谈。
- **题名与摘要：** [EurekaMag 书目及摘要页](https://eurekamag.com/research/112/001/112001467.php) 可读取。此页是**二次索引，不是论文全文**，只支撑当前五条摘要级摘录。
- **机构一手来源：** [KAIST DRCD 官方成员页](https://www.dynamicrobot.kaist.ac.kr/contact) 可确认 Hyunseok Kim、Hae-Won Park 为实验室成员；不提供本文论文正文、代码或定量实验表。
- **预印本与作者版：** 对 arXiv、KAIST 域名及 DLR eLib 进行题名和作者组合检索，尚未发现能够与本 DOI 一一对应的公开全文；检出的其他 GCS、TORO 或 DLR 论文不能当成本论文。
- **项目／源码：** 用户原始消息中的 `hyuns-kim.gith…`、`github.com/Hyu…` 被截断；无法核实实际项目链接、README、求解器入口、许可证或部署路径。

**尚待完成：** 获取正式全文或作者授权预印本，逐节阅读方法／优化式／数值实验／TORO 实验／局限与附录；核对论文原图，并在确认官方可运行实现后补源码时序图。完成前保持 `通读：未完成` 与实体页 `status: draft`，节点健康度不得虚报 3/3。

## 对 wiki 的映射

- [GCS 人形步位规划论文实体](../../wiki/entities/paper-humanoid-footstep-planning-gcs.md)
- [Footstep Planning 概念页](../../wiki/concepts/footstep-planning.md)
