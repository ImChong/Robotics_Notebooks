# Multiple Contact Planning for Minimizing Damage of Humanoid Falls（IROS 2015）

> 来源归档（ingest）

- **标题：** Multiple Contact Planning for Minimizing Damage of Humanoid Falls
- **类型：** paper / humanoid fall mitigation / multi-contact planning
- **作者：** Sehoon Ha、C. Karen Liu
- **机构：** 佐治亚理工学院（Georgia Institute of Technology）
- **发表：** IEEE/RSJ International Conference on Intelligent Robots and Systems（IROS 2015），2761–2767
- **DOI：** <https://doi.org/10.1109/IROS.2015.7353756>
- **项目页：** <https://faculty.cc.gatech.edu/~sha9/projects/ha2015mcs/index.html>
- **论文入口：** 项目页提供 PDF 链接；该链接当前返回 404。IEEE 书目记录可核验 DOI 与会议页码。
- **入库日期：** 2026-10-14
- **一句话说明：** 将失稳后的人形跌倒转化为连续倒立摆子问题，以动态规划选择多接触顺序，逐步耗散初始动量并减小接触冲击；项目页报告了物理仿真和 BioloidGP 实机验证。

## 核心论文摘录（MVP）

### 1) 问题与目标（摘要）

- **问题：** 给定机器人的不稳定初始状态，怎样选择后续身体接触点与顺序，让初始动量逐步耗散并减少机器人承受的冲击。
- **核心贡献：** 用一个规划算法处理从轻微扰动后单步恢复，到高速跌倒时滚动卸力的不同响应，而不是在预设的独立控制策略间切换。
- **证据边界：** 项目页摘要称“minimal impacts”，没有展开目标函数细节。用户提供的截图把优化目标概括为降低最大单次冲击；该更具体表述尚不能从当前可访问的原文摘要独立核实。
- **对 wiki 的映射：** [多接触跌倒规划实体页](../../wiki/entities/paper-ha-liu-multiple-contact-fall-planning.md)、[Balance Recovery](../../wiki/tasks/balance-recovery.md)。

### 2) 方法：倒立摆链与动态规划（摘要）

- 将连续跌倒分解为一串倒立摆问题，并用动态规划高效求解接触规划。
- 规划结果是适合当前失稳状态的接触点序列；其作用是通过后续接触逐步耗散动量，而非假设机器人始终可以恢复直立。
- 摘要未提供各接触阶段的模型方程、状态离散化、代价权重与控制器实现细节；复现时应以完整论文为准。

### 3) 验证与实现状态

- **验证：** 项目页摘要报告了物理仿真验证，以及在 BioloidGP 人形机器人上的实验测试。
- **开放资源：** 官方项目页列有论文、视频和 BibTeX 入口，但当前 PDF、MP4 直链返回 404；页面未列出源代码、数据集或可运行包。故记为“项目材料部分可见、未发现官方可运行实现”，不据此断言作者没有其他未链接代码。
- **对 wiki 的映射：** [多接触跌倒规划实体页](../../wiki/entities/paper-ha-liu-multiple-contact-fall-planning.md)。

## 对 wiki 的映射（汇总）

- 论文归纳：[wiki/entities/paper-ha-liu-multiple-contact-fall-planning.md](../../wiki/entities/paper-ha-liu-multiple-contact-fall-planning.md)
- 交叉更新：[wiki/tasks/balance-recovery.md](../../wiki/tasks/balance-recovery.md)

## 参考来源（原始）

- [Georgia Tech 官方项目页](https://faculty.cc.gatech.edu/~sha9/projects/ha2015mcs/index.html) — 摘要、作者、IROS 2015 信息及项目资源入口。
- [IEEE DOI](https://doi.org/10.1109/IROS.2015.7353756) — 会议论文书目记录。
- [IROS 2015 会议文摘 PDF](https://www.iros2015.org/docs/IROS_Digest_WWW.pdf) — 会议日程与作者机构。
- 用户提供的短视频截图 — 中文解读提及将目标理解为降低最大单次冲击；缺少原帖链接，作为二手线索而非论文原文证据。
