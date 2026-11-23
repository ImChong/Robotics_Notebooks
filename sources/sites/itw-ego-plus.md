# ITW-Ego Plus 官方资料归档

- **类型：** site / dataset / egocentric / motion capture
- **项目入口：** <https://modalitynet.com/>（ITW+）
- **官方公告：** <https://www.globenewswire.com/news-release/2026/11/05/3396456/0/en/noitom-robotics-introduces-itw-ego-plus-at-corl-2026-in-the-wild-egocentric-data-with-full-body-motion.html>（Noitom Robotics，2026-11-05）
- **官方技术文章：** <https://huggingface.co/blog/noitomrobotics/art-of-modalities>（Noitom Robotics，2026-07-21；介绍 ITW 大类，不等同于 11 月 Plus 产品逐条规格）
- **产品/数据目录：** <https://modalitynet.com/>（公开入口；登录后的数据样本和授权条款未读取）
- **核查日期：** 2026-11-21
- **核对范围：** 通读 2026-11-05 官方新闻公告全文与 2026-07-21 官方技术文章全文，并检查 ModalityNet 公开主页；登录后私有数据目录、完整样本、字段字典、同步误差评估和源码未核对。
- **通读：** 已完成
- **开源状态：** 未确认 ITW-Ego Plus 数据公开下载或训练/采集源码公开；**HiPHI / HiPHI-Hand 是不同的开放研究子集**。
- **关联报道：** [具身智能之心：采集小时数已不是竞争点](../blogs/embodied_heart_itw_ego_plus_learnability_2026_11_20.md)

## 官方发布确认的产品边界

2026-11-05 官方公告将产品线区分为 **ITW-EgoVision**（Sol 头戴第一视角视频）和 **ITW-Ego Plus**（围绕第一视角按需添加物理模态）。官方明确列出：

| 模态 | 发布时定义 | 使用前应核查 |
|---|---|---|
| Ego 基础 | Sol 第一视角视觉 | RGB/RGB-D、帧率、时间戳及相机标定的实际字段 |
| Body | 全身运动 | 关节表示、根轨迹、坐标系、传感器数量及校准 |
| Wrist | 腕部视角视频 | 是否左右腕齐备、遮挡率和腕相机标定 |
| Exo | 任务与操作者的固定第三人称视角 | 外参、时序、视野及部署是否一致 |
| Touch | 手部压力 | 压力通道、接触位置、可用性及动态范围 |
| Sound | 环境音频 | 与任务事件的时间对应及隐私条款 |

**不同样本不保证具备全部附加模态。** 官方 CoRL 2026 的重点展示是 **Body**，由自研 Sol 与 Jupiter 设备协作；Jupiter 使用少量可穿戴装置，在无需传统动捕服的条件下获取全身运动。

## ITW 技术路线与先验

- 官方 2026-07-21 技术文章区分：**HiPHI-MOV** 关注全身运动与物体交互参考，**HiPHI-OM** 关注精细操作的高精度多模态真值，**ITW** 关注自然场景的多样性。
- 文章描述使用 HiPHI 与 ITW 视觉、稀疏身体感知训练的多模态模型，恢复腕部与手部动作，形成动作标签；这是对 **ITW 数据增强方式** 的公开说明，不证明所有 ITW-Ego Plus 样本采用同一增强链。
- 官方只说 **部分**带侧视 RGB-D 的 ITW 数据用于 4D 全身/场景重建，不能据此视作 Plus 所有数据的标准交付项。
- 全身人体重建、机器人重定向和接触力监督不是同一件事；不能将人体 BVH/SMPL-X 的存在视为机器人动作与力标签已经验证。

## 目前不能验证的关键规格

| 项目 | 当前结论 |
|---|---|
| ITW-Ego Plus 总小时数、实际样本数量、具体覆盖任务 | 未取得可核实的公开产品数据卡 |
| 统一格式、坐标系、相机内外参与时间误差 | 未见本产品可供复现的完整字段规格/标定统计；应与销售或授权数据目录确认 |
| 所有样本具备 4D 重建、Body/Touch/Exo/Sound | **不能推定**，官方描述是可选模态 |
| 可下载方式、授权与商业使用许可 | 尚未核实；不得套用 HiPHI / HiPHI-Hand 的 MORL 许可 |
| 训练/推理/采集代码与运行入口 | 官方产品页与公告未提供可核查的开源运行实现；**源码运行时序图不适用** |
| 跨本体迁移、场景成功率与数据 scaling 收益 | 本次未见 ITW-Ego Plus 独立的公开对照实验与数字，不能声称已验证 |

## 对 wiki 的映射

- [ITW-Ego Plus 实体](../../wiki/entities/itw-ego-plus.md) — 按模态选型、采集流程、使用边界。
- [HiPHI](../../wiki/entities/paper-hiphi.md) — 独立的高精度人体–物体交互数据集，不与 ITW-Ego Plus 混并。
- [具身数据 Stage 1](../../roadmap/depth-embodied-data.md) — 用“监督缺口 → 采集模态”说明采集工程。
