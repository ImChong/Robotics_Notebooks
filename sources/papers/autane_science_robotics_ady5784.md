# Autane：面向港口持续作业的自主起重机（Science Robotics 2026）

> 来源归档（paper）；仅核验期刊摘要与研究团队官方介绍，尚未完成论文全文阅读。

- **英文标题：** Autane: An autonomous crane for sustained port operations
- **中文标题：** Autane：面向港口持续作业的自主起重机
- **作者：** Jun Yang, Zheng Tian, Junming Hong, Qin Rao, Jinya Su, Huan Chen, Shihua Li
- **机构：** 香港科技大学（广州）；东南大学；上海驭矩信息科技有限公司
- **期刊：** Science Robotics, 11(120): eady5784（2026-11-11）
- **DOI：** 10.1126/scirobotics.ady5784
- **论文：** <https://www.science.org/doi/10.1126/scirobotics.ady5784>
- **用户提供 PDF：** <https://www.science.org/doi/epdf/10.1126/scirobotics.ady5784>
- **摘要索引：** <https://pubmed.ncbi.nlm.nih.gov/43013691/>
- **研究团队官方说明：** <https://www.hkust-gz.edu.cn/2026/11/16/hkustgz-autonomous-crane-system-makes-science-robotics-cover/>
- **入库日期：** 2026-11-19
- **核对范围：** PubMed 上期刊题录、作者机构及摘要全文；港科大（广州）2026-11-16 官方研究介绍全文。Science.org 的 HTML 返回 403，用户提供的 PDF 无法通过本次读取工具获取；论文正文、图表、补充材料均未逐节核验。
- **通读：** 未完成

## 核心摘录（限定证据来源）

1. **研究对象（期刊摘要）：** 对港口堆场的轮胎式龙门起重机（RTG）进行自主集装箱抓取、运输与堆放；难点包括大尺度运动安全、厘米级操作精度及长距离感知。
2. **系统机制（期刊摘要）：** 多传感器融合、实时障碍检测、同步定位与建图（SLAM），结合路径调整，将感知、规划、控制及运维连接为闭环，以完成箱体对位与车辆交互。
3. **传感与控制（研究团队官方介绍）：** LiDAR、相机、GNSS、IMU 等感知；模型预测控制（MPC）与扰动抑制用于改善悬吊负载摆动。摘要未公开可复现的数学式、控制参数和模块接口。
4. **落地情况（期刊摘要）：** 作者报告跨多个港口部署、全天作业，以及自主处理超过 200 万 TEU；此数为论文摘要口径。
5. **后续披露（研究团队官方介绍，2026-11-16）：** 超过 60 台、覆盖 5 类起重机；累计超过 480 万 TEU；对超过 6 万次作业的统计给出 91.23% 任务成功率与 25.93 箱次/小时。该介绍并未交代两种 TEU 口径的时间截面/样本是否一致，不能直接将其作为论文实验表数字。
6. **工况（研究团队官方介绍）：** 提及风速 15 m/s、日降雨量 25 mm、能见度 50 m 的作业能力；是否为独立极端工况试验以及具体边界条件，仍待论文核验。

## 原文与复现核验状态

- **未完成：** 无法通读论文的方法、实验、图表、局限与补充材料。禁止将本页视为论文精读。
- **源码：** 本次在期刊摘要与研究团队官方介绍中未找到 Autane 对应的可运行仓库/发布入口；不能据此断言作者从未开源。
- **后续：** 获得出版社全文或授权作者版后，复核控制与感知模块图、模型预测控制目标/约束、故障恢复、安全接管以及成功率/吞吐量统计方法；按正文更新 wiki 和 Mermaid 图。

## 对 wiki 的映射

- [Autane 唯一论文实体](../../wiki/entities/paper-autane-autonomous-crane.md)

## 参考来源

- [论文 DOI（出版社）](https://doi.org/10.1126/scirobotics.ady5784)
- [PubMed 期刊摘要](https://pubmed.ncbi.nlm.nih.gov/43013691/)
- [港科大（广州）研究团队官方介绍](https://www.hkust-gz.edu.cn/2026/11/16/hkustgz-autonomous-crane-system-makes-science-robotics-cover/)
