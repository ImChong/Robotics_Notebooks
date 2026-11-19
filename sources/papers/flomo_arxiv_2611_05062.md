# FloMo（arXiv:2611.05062）

> 来源归档（paper）；本次按官方 arXiv HTML v2 核对，非二手摘录

- **标题：** FloMo: 3D Scene Flow as a World Action Model Intermediate for Learning from Human Video
- **中文标题：** FloMo：以三维场景流作为世界–动作模型的中间表示，从人类视频学习机器人控制
- **作者：** Jeremy A. Collins、Namra Patel、Ayush Agarwal、Shitij Govil、Animesh Garg
- **机构：** 佐治亚理工学院（Georgia Institute of Technology）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.05062>
- **HTML：** <https://arxiv.org/html/2611.05062>
- **PDF：** <https://arxiv.org/pdf/2611.05062>
- **版本：** arXiv:2611.05062v2，2026-11-04（初稿 2026-11-03）
- **项目页：** <https://flomo-wam.github.io/>
- **代码：** <https://github.com/pairlab/flomo>
- **权重：** <https://huggingface.co/shitijgovil/FloMo>
- **入库日期：** 2026-11-05；原文复核：2026-11-19
- **核对范围：** arXiv HTML v2 全文：摘要、引言、相关工作、§III 方法 A–G、§IV 实验 A–D、§V 局限、附录 A 架构/训练/推理/损失与附录 B 数据/场景流提取/采样权重；同时核对官方项目实验表及仓库 README 与训练、预处理、推理入口。
- **通读：** 已完成

## 原文核心摘录（释义）

1. 未来视频预测包含纹理、光照、背景等对控制可能冗余的细节；论文检验用 3D Scene Flow 替代下游预测目标，而**不放弃视频生成模型预训练**。
2. SpatialTrackerV2 在 16 帧内追踪首帧 64×64 网格的 4,096 个点，统一到首帧相机坐标系，构造累计位移。对 XYZ 归一化、渲染三通道场景流图，使用冻结 Wan VAE 获得 video-compatible motion latents。
3. Wan 2.2 5B DiT + LoRA rank 64 在首帧图像/指令条件下，联合做运动 token 与动作 token 的 Flow Matching；默认 motion/action loss=1/0.1，视频 loss=0；机器人单独微调阶段 action loss 增至 1。
4. 人类视频（约 81 h EgoVerse + 0.7 h 内部无动作标签视频）提供 motion-only 监督；机器人示范包含 motion 与 action。
5. 真机已见 3 任务的成功率：MA 91%、VMA 51%、VA 40%，各任务各 15 次。三个 OOD 条件的 MA 18/30、VMA 2/30、VA 2/30；人视频移除消融为 1/15（每条件 5 次）。
6. RoboTwin 2.0 Easy 五个单任务（每任务 50 示范、100 评测）：FloMo 63.6%、π₀.₅ 44.8%、FAST-WAM 41.0%、BC 19.2%。
7. 逆动力学 probe 在 1% 动作标注数据下，真实运动条件比真实视频条件的动作误差约低 22%；动作查询对 motion 的平均注意力约为 video 的 3 倍。二者是解释性分析，不是额外的端到端成功率对照。
8. 局限：每 16 帧 tracking 约耗时 2.4s（L40S）、预测 motion 与 action 可能不一致、正面视角的 motion prediction 遭遮挡会退化、文本目标识别有失败案例。

## 原文口径提醒

- OOD 主比较每条件 10 次，去人类视频消融每条件 5 次；两者成功率差是论文结果，不应忽略样本规模差异。
- 训练用的 20D 旋转 6D 动作表示，与官方 `infer_example.py` 输出经后处理的 14D yaw/pitch/roll 双臂末端控制格式不同。
- §III-G 与附录 A-C 对推理 GPU/耗时给出不同配置数字；避免将它们混写为统一设备速度。
- 本页来自论文、项目页和开源 README；早先公众号转述仅用于发现资料。

## 对 wiki 的映射

- [FloMo 独立论文节点](../../wiki/entities/paper-flomo.md)
- [WAM 纵深路线](../../roadmap/depth-wam.md)
- [既有 43 篇策展对照](../../wiki/overview/embodied-station-2026-11-05-43-papers-technology-map.md)
