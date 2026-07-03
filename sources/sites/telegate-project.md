# TeleGate 项目页（anywitresearch.github.io/TeleGate）

> 来源归档

- **标题：** TeleGate: Whole-Body Humanoid Teleoperation via Gated Expert Selection with Motion Prior
- **类型：** site（项目页 + 演示视频 + 数据集申请）
- **URL：** <https://anywitresearch.github.io/TeleGate/>
- **论文：** <https://arxiv.org/abs/2602.09628>
- **机构：** 中国科学技术大学（USTC）、芜湖哈特机器人技术研究院（Wuhu Hit Robot Technology Research Institute）；项目页域名 **AnyWit Research**（research@anywit.cn）
- **作者：** Jie Li*、Bing Tang、Feng Wu（arXiv 另列 Rongyun Cao）
- **会议：** Robotics: Science and Systems（**RSS 2026**）
- **硬件：** Unitree G1（29 DoF）；**惯性动捕** 全身遥操作
- **入库日期：** 2026-07-03
- **一句话说明：** 门控专家选择 + VAE 运动先验的全身人形遥操作：冻结多域专家、轻量门控实时路由，避免蒸馏性能损失；仅 **2.5 h** 自采惯性动捕数据即可在 G1 上跟踪跑跳、跌倒恢复等高动态动作。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Abstract | 统一全身遥操作；门控网络按本体态与参考轨迹动态激活专家；VAE 从历史参考提取隐式未来运动意图 |
| Whole-Body Teleoperation | 跌倒起身、踢球、抓放玩偶、走跑、格斗、躯干运动等真机演示视频 |
| Dataset | **惯性动捕数据集** 研究用途开放；需签署中英文 permission form 并邮件 research@anywit.cn |
| BibTeX | `@misc{li2026telegatewholebodyhumanoidteleoperation,...}` arXiv:2602.09628 |

## 对 wiki 的映射

- 主实体：[TeleGate（论文实体）](../../wiki/entities/paper-telegate.md)
- 论文摘录：[telegate_arxiv_2602_09628.md](../papers/telegate_arxiv_2602_09628.md)
- 任务交叉：[Teleoperation](../../wiki/tasks/teleoperation.md)
