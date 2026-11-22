# Zhengyi Luo：From Motion Tracking to World Action Models for Humanoids（2026-11-18）

- **类型：** talk / GRASP SFI Seminar / humanoid whole-body control
- **讲者：** Zhengyi (Zen) Luo，NVIDIA GEAR Lab
- **英文题目：** From Motion Tracking to World Action Models for Humanoids
- **中文题目：** 从人形运动跟踪到世界动作模型
- **主办：** University of Pennsylvania，GRASP Lab
- **活动时间：** 2026-11-18 15:00–16:00（美国东部时间），Levine 307
- **一手活动日历：** <https://almanac.upenn.edu/at-penn-calendar>
- **讲者本人预告：** <https://x.com/zhengyiluo/status/2122812673645740041>（2026-11-17；预告链接可能因平台访问限制不可用）
- **讲者研究主页：** <https://www.zhengyiluo.com/>
- **二手中文整理：** human five，2026-11-22，<https://mp.weixin.qq.com/s/tnafAt6AmFgInGRU1IsFXQ>
- **入库日期：** 2026-11-22
- **核对范围：** 核验宾夕法尼亚大学正式活动日历中的日期、题目、地点，并比对讲者本人公开预告；阅读用户提供的中文转述全文；技术机制与实验数值按 SONIC、GR00T 2、GRAIL 的官方论文归档及项目页交叉核对。**尚未获得可核验的完整官方录像、幻灯片或讲者逐字稿**，故不能将中文转述视为已核实逐字记录。
- **通读：** 未完成

## 演讲主线（来自中文转述；技术要点已尽可能与官方资料交叉核对）

1. **基础任务：** 将 motion tracking 类比 LLM 的 next-token prediction。它是形成可迁移全身控制先验的训练目标，不等于自主决策。
2. **SONIC：** 海量人体动作捕捉、重定向、PPO 训练，以及通用 motion latent/token 接口；上游可从 VR、人体视频、运动生成器、VLA 提供条件，下游负责接触、平衡与跟踪。
3. **操作数据：** 全身遥操作能采机器人示范，代价是操作员、环境和本体占用；GRAIL 用有尺度的 3D 资产/相机/视频生成/4D 重建/物理追踪合成训练数据。
4. **WAM：** GR00T 2 联合预测未来视频和动作块；其先验由 egocentric human、UMI、真实机器人演示共同训练。该联合建模不是简单的「先完整渲染视频，再顺序执行动作」。
5. **板载双层控制：** GR00T 2 Turbo 在 Jetson Thor 上产生 1.6 s / 80 个 latent 动作单元，SONIC 在 Orin 上 50 Hz 解码；关节 PD 在 1000 Hz 执行。用 action chunk 缓冲降低慢速模型与快速物理闭环的冲突。
6. **边界：** 状态到首次可用动作的延迟、示范质量、物理一致性、执行器及本体形态限制，决定演示能否转成可靠任务。

## 逐项核对与存疑记录

| 中文转述中的说法 | 独立核对 |
|---|---|
| SONIC 数据/模型/算力扩展有效 | SONIC 官方论文与项目页有规模实验；见下列原文 |
| GR00T 2 Turbo 生成 80 token / 1.6 s | GR00T 2 白皮书附录 H.3 支持；SONIC 运行 50 Hz |
| 「SONIC 每 5 ms 消费一个 Token」 | **与 80/1.6 s = 50 Hz（20 ms）不一致**；应以官方资料的 50 Hz 为准 |
| 「170 ms round-trip」与「2–3 ms SONIC state-to-action」 | 转述中的口头数字；当前官方项目页核实了 Thor 约 124 ms model prediction，但这些定义不同，未独立核实前两项 |
| 「3 小时遥操作可训练出洒漏清理」 | 中文转述报告，未从项目页独立验证采集时长；官方白皮书给出洒漏清理 14/30 成功 |
| 「爆米花任务 100%」 | 官方白皮书任务协议下为 48/48；不是任意新场景泛化率 |
| GR00T 2 Turbo「纯粹更快，没有质量差异」 | 官方白皮书表 1 的 YAM / Dexter 任务比较有相对性能下降；速度与任务表现需同时报告 |
| 「NeuroDynamics」等口头术语 | 转述含可能的语音转写/译名误差，未按字面写成经证实的算法名称 |

## 官方一手资料

- [SONIC 原始论文 / 论文通读归档](../papers/sonic_reading_arxiv_2511_07820.md)
- [SONIC 官方项目页](https://nvlabs.github.io/GEAR-SONIC/)
- [GR00T 2 白皮书通读归档](../papers/gr00t2_whitepaper_2026.md)
- [GR00T 2 官方项目页](https://research.nvidia.com/labs/gear/gr00t-2/)
- [GRAIL 原始论文 / 论文通读归档](../papers/grail_arxiv_2606_05160.md)
- [GRAIL 官方项目与源码](https://research.nvidia.com/labs/dair/grail/)
- [SONIC 训练与部署源码](https://github.com/NVlabs/GR00T-WholeBodyControl)

## 对 wiki 的映射

- [从 Motion Tracking 到 WAM 的人形双层闭环](../../wiki/queries/sonic-gr00t2-wam-humanoid-dual-brain.md)
- [SONIC](../../wiki/entities/nvidia-sonic.md)
- [GR00T 2](../../wiki/entities/paper-gr00t-2.md)
- [GRAIL](../../wiki/entities/paper-grail.md)
