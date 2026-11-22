# Motion-Between BFM-2（智元官方发布归档）

> 来源归档（官网官方文章 + Humanoid Motion Intelligence 主表）

- **标题：** “身外化身” 再进阶！智元发布全球首个端到端Motion-Between BFM-2运动基座模型
- **类型：** project-site / official-article
- **技术路线分组：** 动作跟踪与全身控制
- **官方链接：** <https://www.agibot.com.cn/article/315/detail/161.html>（规范路径：<https://www.agibot.com.cn/news/company/motion-between-bfm-2-161>）
- **公开视频：** <https://www.bilibili.com/video/BV1ZzGe6oEmk/>
- **发布时间：** 2026-05-26
- **入库日期：** 2026-07-30（2026-11-21 补充官方全文通读）
- **核对范围：** 通读智元官网官方发布全文（<https://www.agibot.com.cn/article/315/detail/161.html>，含 DOF Feather Motion Generator 与二阶段 Motion-Between 架构描述、与 VLA 关系及 BFM-3 规划）；经官网与 GitHub 检索确认无公开技术论文、代码仓库或模型权重。
- **通读：** 已完成
- **一句话说明：** 当前全身动力学、接触状态与高层目标构型共同条件化二阶段Motion-Between模型，输出连续的状态间过渡。它为VLA提供身体层执行接口，公开证据尚不足以判断控制频率和跨本体范围。
- **开源状态：** **确认未开源**（仅官方新闻稿与公开视频演示，无公开论文/GitHub 仓库/权重）
- **策展入口：** [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)
- **沉淀到 wiki：** 是 → [`wiki/entities/agibot-bfm-2.md`](../../wiki/entities/agibot-bfm-2.md)

## 官方原文要点

1. **核心机制：** 首次在大规模全身运动基座中引入端到端 **DOF Feather Motion Generator** 生成式训练机制，对机器人全身动力学状态空间做连续概率建模，结合 **二阶段 Motion-Between 架构**，根据实时全身动力学状态、接触约束状态与高层目标指令，在高维运动流形中自主推演从任意当前姿态到任意随机指令构型的完整过渡轨迹。
2. **定位区别：** 不同于依赖人工预设过渡轨迹、或仅支持有限预设动作的“摔倒自恢复”控制器，强调在仰躺、趴地、大幅倾斜、悬空偏移、外力踹击与指令跳变等状态下的通用运动推理与动态重新规划。
3. **系统衔接与演进：** 作为运动“小脑”底座缓解“高层 VLA 输出不稳定、底层运控要求绝对连续稳定”的接口矛盾；官方预告后续 **BFM-3** 将从纯本体状态输入扩展为融合视觉、触觉、语音、空间语义与环境拓扑的多模态运动基座。

## 对 wiki 的映射

- [BFM-2（智元运控基座）](../../wiki/entities/agibot-bfm-2.md)
- [Humanoid Motion Intelligence](../../wiki/entities/humanoid-motion-intelligence.md)
