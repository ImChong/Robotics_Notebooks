# HumanoidTTT 论文来源归档

## 元数据

- **题目：** HumanoidTTT: Test-Time Capability Reuse for Efficient Humanoid Control
- **作者：** Jingtai Yang、Yining Wu、Yanjun Li、Zeyu Zhang、Hao Tang
- **机构：** 北京大学计算机学院（School of Computer Science, Peking University）
- **预印本：** https://arxiv.org/abs/2610.00198
- **HTML：** https://arxiv.org/html/2610.00198v1
- **PDF：** https://arxiv.org/pdf/2610.00198
- **版本：** v1，2026-09-18，cs.RO
- **项目主页：** https://aigeeksgroup.github.io/HumanoidTTT/
- **官方代码：** https://github.com/AIGeeksGroup/HumanoidTTT
- **模型：** https://huggingface.co/AIGeeksGroup/HumanoidTTT
- **一句话说明：** 为人形机器人重复出现的动作请求复用已验证的完整动作，并在有限存储内按后续复用效用在线调整保留策略。

## 论文要解决的问题

连续部署中，机器人可能重复收到相同动作请求，但中间执行的其他动作会改变当前入口状态。仅按请求 ID 命中并重放旧轨迹可能不安全；每次都重新生成又浪费计算。与此同时，已验证动作会不断累积，有限容量的存储需要决定保留哪些动作。

## 方法摘要

- **Selective Full-Motion Reuse（选择性完整动作复用）：** 将已成功执行的完整运动、动作身份和入口适用性证书作为能力存入 Full-Motion Store。请求命中后，先检查动作/运行时签名、机器人稳定性、接触、关节限位、历史帧等物理条件，再检查当前入口描述是否落在该动作的认证适用域。通过时直接执行存储动作，跳过冻结动作生成器；未通过则回退到新生成及执行验证。
- **入口状态表示：** 45 维候选相对描述由关节和根部偏差、速度摘要、足部接触、物理裕度及短时执行历史构成。成功执行时记录适用域中心；每个动作的证书在收录后保持冻结。
- **Test-Time Capability Consolidation（测试时能力整合）：** 存储满时，对新通过验证的能力做 SKIP 或 REPLACE(j) 决策。共享动作评分器输入为 71 维，网络结构为 71→128→64→1；Double-DQN 依据两个满容量写入决策之间观测到的成功复用比例在线更新。
- **计算路径：** 已接受的复用与小型整合策略运行在 CPU；命中时不唤醒 GPU 动作生成器。未命中才调用 GPU 生成路径。

## 论文报告的结果

- **复用可靠性：** 200 个测试入口中接受 180 个，180 个全部通过安全标签；观察到 0 次不安全接受、887 个 OOD 入口中的误接受为 0，540 次已接受重放全部成功。
- **命中准备延迟：** 接受复用的中位数为 29.514 ms，Fresh OMG 为 5057.996 ms，论文报告 171.4 倍加速；P95 为 63.115 ms。
- **端到端延迟：** 固定的 200 请求高复用轨迹上，平均延迟 268.66 ms，相比 Fresh OMG 的 4415.82 ms 为 16.4 倍加速。
- **有限容量保留：** 容量 K=10、15 条匹配的 200 请求流上，HumanoidTTT 平均避免 131.4 次生成调用；同一预训练策略关闭在线更新时为 118.2 次，差值 13.2 次。
- **真机演示：** Unitree G1 上展示 A–B–A 连续序列，包括前行、转向、挥手、下蹲再站起；由 HoloMotion 执行全身跟踪。

## 解释边界

上述零不安全接受来自论文设定的测试入口、固定物理判据与回放协议，不等于对任意机器人、状态分布或长期部署的形式化安全保证。证书目前在能力收录后冻结，持续动力学变化、环境变化及适用域漂移尚未纳入在线适配。源码库提供方法组件；动作生成器、HoloMotion 运行时、动作档案、入口几何/尺度和执行验证循环由应用提供。

## 对 wiki 的映射

- 详情页：[HumanoidTTT 实体页](../../wiki/entities/paper-humanoidttt.md)
- 对照阅读：[TWIST2](../../wiki/entities/paper-twist2.md)、[YAHMP](../../wiki/entities/paper-yahmp.md)
- 相关方法：[SONIC 全身运动跟踪](../../wiki/methods/sonic-motion-tracking.md)

## 一手来源

- [arXiv 摘要](https://arxiv.org/abs/2610.00198)
- [arXiv HTML 全文](https://arxiv.org/html/2610.00198v1)
- [官方项目页](https://aigeeksgroup.github.io/HumanoidTTT/)
- [官方代码](https://github.com/AIGeeksGroup/HumanoidTTT)
