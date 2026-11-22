# RoboTryOn：将人类视频转换为灵巧机器人示范（arXiv:2611.05163）

> 来源归档（paper）；本次是对已有论文归档的原文核验升级，复用 `wiki/entities/paper-robotryon.md`。

- **标题：** Synthesizing Dexterous Robot Demonstrations from Human Videos via Virtual Try-On（RoboTryOn）
- **作者：** Seungyong Lee、Suhyeok Jang、Minyoung Kim、Jinwoo Shin
- **机构：** 韩国科学技术院（KAIST）、瑞沃世界（RLWRLD）、高丽大学（Korea University）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.05163>
- **原文版本：** arXiv:2611.05163v3（2026-11-05，HTML / PDF）
- **HTML：** <https://arxiv.org/html/2611.05163>
- **PDF：** <https://arxiv.org/pdf/2611.05163>
- **项目页：** <https://ryan-seungyong-lee.github.io/RoboTryOn/>
- **代码：** <https://github.com/ryan-seungyong-lee/RoboTryOn>
- **权重：** <https://huggingface.co/RyanL22/RoboTryOn>
- **入库日期：** 2026-11-05
- **复核日期：** 2026-11-22
- **核对范围：** 论文 HTML v3 全文：Abstract、§1–5、References、Appendix A–F（方法、训练配置、真机/仿真评测、消融、代价与失败案例）；官方项目页、GitHub README、`scripts/run_pipeline.py` 与 `configs/pipeline.yaml` 对照。
- **通读：** 已完成
- **原始线索：** [具身智能小站 2026-11-05 43 篇](../blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)
- **沉淀到 wiki：** [paper-robotryon](../../wiki/entities/paper-robotryon.md)

## 原文问题与贡献（§1–3）

人类第一视角操作视频缺目标机器人视觉外观和关节动作。RoboTryOn 不直接把人手关节重定向作为监督：先用人手骨架和目标本体参考图像，以自回归关键帧翻译 + 视频生成得到目标机器人视频，再用基于深度特征的逆动力学模型（IDM）预测关节动作，并以人体重建的腕部位置做 IK 校正。合成示范含视觉观测、伪关节状态/动作与语言指令，可以单独或混合遥操作数据训练 VLA。三模型在已有机器人遥操作视频上学习，不要求人机成对示范。

## 关键实测数据（§4；Appendix C、D、F）

- **对象：** OpenArm + 6-DoF Inspire RH56F1、RB-Y1 + 20-DoF WujiHand2；4 个有机器人示范的域内取放任务，7 个仅有人视频的域外任务；每任务 28 次真机评估。双手任务按两只手各 0.5 分计。
- **π₀.₅ 域外成功评分：** 遥操作 only：OpenArm 3.1%、RB-Y1 1.0%；加入 RoboTryOn：OpenArm 73.0%、RB-Y1 46.9%，两机器人平均约 2.1%→60.0%（Table 1 / Table 11）。
- **域内：** OpenArm 47.3%→75.9%；RB-Y1 69.6%→79.5%，对照为遥操作 only 与遥操作 + RoboTryOn。
- **标签可执行性：** 66 个场景匹配的仿真开环回放，直接重定向 34.8%，IDM + 腕部校正 69.7%（Figure 4 / Table 8）。
- **视觉域：** Fréchet SigLIP distance：原始人视频 125.5，RoboTryOn 32.7（Figure 3 / Table 7），不能和真机任务成功率互换。
- **关键消融：** OpenArm 域外 73.0%；移除手部骨架引导 21.4%，移除后续关键帧 39.3%，移除手腕校正 48.0%（Table 3）。
- **成本口径：** 在固定性能附近，人类视频合成示范可按约三分之一的人员采集工时对齐 50 条遥操作示范训练；计时来自两人遥操作与单人视频采集，计算资源开销另计（Figure 6、Tables 9–10）。
- **计算代价：** Appendix B Table 6：15 秒演示（单本体、单 H100）逐阶段 134+173+150+150+22+66 秒，合计 695 秒，约 11.6 分钟；不属于实时推理。

## 方法与工程细节（§3；Appendix B）

- 关键帧：双腕速度局部低谷，按位移/置信度控制间隔；长片段上限约 145 帧（20 Hz 下约 7.25 s）。
- 条件：17 手部关键点、遮罩、前一机器人关键帧；第一关键帧由目标本体参考图指定。
- 图像模型：FLUX.2 [klein] 9B；视频模型：LTX-2.5；IDM：Depth Anything V2-L + flow matching action head。
- 逆动力学：两个间隔 20 帧的视觉观测预测 20 步动作；滑窗融合并平滑，状态按从遥操作数据测得的控制状态滞后得到；手臂 IK 只校正手腕位置，保留 IDM 的手指动作。
- VLA 结果分别测试 π₀.₅ 和 RLDX-1-Img，但机器人实机的主要评估是 OpenArm、RB-Y1。

## 局限与误读（Appendix A、F）

1. **无目标任务机器人示范 ≠ 无机器人训练数据。** 新本体仍要先有一些自身遥操作视频训练/适配生成模型与 IDM。
2. **没有力标签**；细接触、严格配合和不可达的手部动作会失败，RB-Y1 的 Air Fryer 项为 0%。
3. 摄像头位置依赖机器人头部视点和标定；论文底座与头颈在训练中固定，移动视角和广义野外视频的泛化未经充分验证。
4. 视觉生成会出现物体消失、穿模等失败；伪动作亦可能关节/抓握位置不准。
5. GR-1 仅做视频形态翻译示例，未报告物理 GR-1 机器人策略实测。

## 开源与复现核对

2026-11-22 打开官方项目页，确认 Code、Checkpoints、Hugging Face Demo 入口；官方 GitHub 包含预处理、三模型训练/推理脚本和端到端 `scripts/run_pipeline.py`。权重公开但机器人模型资产不随仓库提供，训练与推理依赖多个环境；细见[源码归档](../repos/robotryon.md)和[项目页归档](../sites/robotryon.md)。

## 对 wiki 的映射

- [唯一论文详情节点：paper-robotryon](../../wiki/entities/paper-robotryon.md)
- [43 篇技术地图](../../wiki/overview/embodied-station-2026-11-05-43-papers-technology-map.md)
