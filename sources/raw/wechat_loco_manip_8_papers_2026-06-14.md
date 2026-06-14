# 原始抓取：人形机器人 Loco-Manip 这周都在卷啥？这 8 篇论文挺有意思

> Agent Reach 原始正文归档（非 wiki 归纳）

- **来源 URL：** https://mp.weixin.qq.com/s/Ez87ljBYmCyIpLKjMjEyaQ
- **抓取日期：** 2026-06-14
- **抓取工具：** Agent Reach v1.4.0 + `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）
- **编译导读：** [wechat_embodied_ai_lab_loco_manip_8_papers_survey.md](../blogs/wechat_embodied_ai_lab_loco_manip_8_papers_survey.md)

---

（以下为抓取正文，保留 frontmatter 与结构；推广/群二维码段落已省略。）

**标题：** 人形机器人 Loco-Manip 这周都在卷啥？这 8 篇论文挺有意思  
**作者：** 具身智能研究室  
**发表：** 2026-06-14

核心论点：人形 Loco-Manip 的数据瓶颈不仅是轨迹数量，更是 **数据从哪来、如何对齐身体、能否变成可执行经验**。8 篇论文按四组读：

- Ego-Pi、EgoPriMo：第一视角数据进入策略
- GenHOI、OASIS：生成视频与仿真数据
- VAIC、M3imic：动作参考统一到控制器
- WT-UMI、X-OP：触觉与跨本体遥操作

五类试探方向：第一视角语义/动作、生成视频、仿真批量、触觉力反馈、跨本体遥操作。

各篇要点（策展级）：

1. **Ego-Pi**（Stanford/Meta, CVPR 2026 ext, arXiv:2606.08107）：人类 ego + 机器人数据共微调 Pi0.5；人类视频供 **高层语义/任务链**，机器人数据负责 **动作落地**。
2. **EgoPriMo**（arXiv:2606.08495）：ego 观察 + 文本 → **SMPL 全身动作** → 人形控制器；补「第一视角到全身」缺口。
3. **GenHOI**（arXiv:2606.12995）：现实感知 → 生成 HOI 视频 → 提取接触/物体轨迹 → 优化目标；生成视频作 **交互线索** 而非直接执行。
4. **OASIS**（arXiv:2606.08548）：真实图重建资产 → 仿真遥操作采集 → 域随机化层级策略 → **零样本 G1**；仿真数据覆盖可超真实 teleop。
5. **VAIC**（arXiv:2606.09286）：解耦命令（速度/交互阶段/物体状态）+ 视觉；搬箱/推车/拉车/滑板。
6. **WT-UMI**（arXiv:2606.13232）：全身触觉接口 + 力监督接触规划；补视觉看不见的接触状态。
7. **X-OP**（arXiv:2606.07934）：MPC 重定向跨本体全身遥操作；遥操作作 **多机器人数据入口**。
8. **M3imic**（arXiv:2606.04829）：多模态动作参考编码进 **共享潜在命令空间** + 统一 WBC。

收束：数据入口从单点采集走向混合生产链；须问来源、身体对齐、接触信息、训练链路统一性、跨平台复用与可部署控制器。
