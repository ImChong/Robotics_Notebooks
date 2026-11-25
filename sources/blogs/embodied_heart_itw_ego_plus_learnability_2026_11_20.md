# 采集小时数已不是竞争点，大规模数据可学习性的角逐才开始

- **资料类型：** 行业报道 / blog（二手资料，不是论文或官方数据卡）
- **作者：** 具身智能之心
- **发表：** 2026-11-20 21:00（据用户提供的文章信息）
- **原文：** <https://mp.weixin.qq.com/s/xFE6DeAzURIkaTUeMFORhw>
- **核对范围：** 用户提供的完整文字正文及所附原文链接；微信公众号网页本身未单独复核。
- **通读：** 已完成
- **对应实体：** [ITW-Ego Plus](../../wiki/entities/itw-ego-plus.md)
- **一手对照：** [Noitom Robotics 2026-11-05 官方公告](https://www.globenewswire.com/news-release/2026/11/05/3396456/0/en/noitom-robotics-introduces-itw-ego-plus-at-corl-2026-in-the-wild-egocentric-data-with-full-body-motion.html) · [ModalityNet 技术博客](https://huggingface.co/blog/noitomrobotics/art-of-modalities)

## 文章核心论点

文章认为，大规模第一视角视频往往看不见人体在镜头后的位姿、被遮挡的接触过程以及精细触觉信息，因此**采集时长不等于机器人可学习的有效监督量**。企业提出以 ITW-Ego Plus 的可选模态填补这些监督缺口。此为报道观点，不是经过独立对照实验验证的通用 scaling law。

## 本次可映射到已有实体的事实

| 报道要点 | 一手核对结论 |
|---|---|
| ITW-Ego Plus 在 CoRL 2026 展示 | Noitom Robotics 2026-11-05 官方发布信息确认 |
| 以 Ego 视觉为底座加 Body / Wrist / Exo / Touch / Sound | 官方公告逐项列明五种可选模态，**不代表所有样本均有** |
| Sol 头戴设备 + Jupiter 轻量可穿戴采集全身运动 | 官方公告确认 Body 展示与无需传统动捕服 |
| HiPHI-MOV 提供高精度人体与物体交互先验 | 官方技术文章及 HiPHI 论文支持这一**研究方向**；不能推导 Plus 逐帧都具备同等真值 |
| 4D 场景点云与原始 6DoF/IMU 可选输出 | 官方技术文章仅明确**部分** ITW 采用侧视 RGB-D 进行 4D 重建；未找到 Plus 全量字段规格 |
| 毫秒级全模态同步 / 精度 | 官方资料强调同步，但当前未找到 ITW-Ego Plus 独立公开的端到端时间误差与标定测试报告 |

## 场景配方（将报道的示例转为工程选型假设）

1. **移动操作**：Ego + Body，为动作重定向提供全身姿态与第一视角上下文；需验证根坐标、脚接触、人体→机器人 DoF 映射。
2. **遮挡下的灵巧操作**：Ego + Wrist，必要时加 Touch；腕摄像头补视角，不自动消除夹持接触真值的缺失。
3. **世界模型与场景推理**：Ego + Exo / Sound，若实际样本包含场景重建结果再使用 4D 几何；区分时序相关性与真实因果标签。

## 不能直接写成已证实结论的报道信息

- “200 万小时网络视频不如 2000 小时手套数据”未在本次核对的一手材料中找到**同任务、同训练预算、同指标**的可复核实验；不能直接概括为一般定律。
- 估值超 5 亿美元、年营收超 1000 万美元、2025 年分拆、10,000 小时自有动捕先验等公司信息，文章作了陈述，本次未取得对应独立证据，故不作为实体定量事实。
- “毫秒同步”“可直接支撑跨本体迁移”“让每小时规模增长都转化为能力提升”缺少 ITW-Ego Plus 数据字段、同步统计或下游消融支撑，应当表述为产品愿景或待验证假设。
- 与 HiPHI 和 HiPHI-Hand 的开放研究子集**不同**，未核实 ITW-Ego Plus 可以直接公开下载或采用 MORL v1.0 许可。

## 对知识库的映射

- 将官方可验证的产品定位、五种附加模态与工程边界更新到 [ITW-Ego Plus 实体](../../wiki/entities/itw-ego-plus.md)，不创建重复数据集页面。
- 在 [具身数据纵深路线 Stage 1](../../roadmap/depth-embodied-data.md) 增加“按监督缺口选模态”案例。
- 保留 [HiPHI](../../wiki/entities/paper-hiphi.md) 作为独立的数据集与论文节点。
