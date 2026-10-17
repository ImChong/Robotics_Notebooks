# DevGRU: Depth-guided Visual Navigation using a Collision-aware Recurrent Model（arXiv:2608.18470）

> 来源归档（ingest · 论文摘录）

- **标题：** DevGRU: Depth-guided Visual Navigation using a Collision-aware Recurrent Model
- **类型：** paper / visual navigation / point-goal navigation / collision avoidance / topological map
- **arXiv abs：** <https://arxiv.org/abs/2608.18470>
- **HTML：** <https://arxiv.org/html/2608.18470>
- **PDF：** <https://arxiv.org/pdf/2608.18470>
- **项目页：** <https://graphics.ewha.ac.kr/projects/details/devgru/> — 归档见 [devgru 项目页](../sites/devgru.md)
- **官方代码：** [EwhaGlab/devgru](https://github.com/EwhaGlab/devgru) — 归档见 [devgru 仓库](../repos/devgru.md)
- **期刊：** IEEE Robotics and Automation Letters (RA-L), 2026；DOI: 10.1109/LRA.2026.3730213
- **作者 / 机构：** Kyung Min Han、Eunsom Kim、Young J. Kim；梨花女子大学（Ewha Womans University）
- **入库日期：** 2026-10-17
- **一句话说明：** 面向室内长程点目标导航，DevGRU 将深度图、机器人位姿与拓扑子目标输入轻量 GRU 策略，预测避障路点并主动修正累计漂移的子目标位姿；独立碰撞预测器补充近场风险判断。

## 摘要级要点

- **问题：** 视觉导航策略在狭窄、结构化室内环境容易碰撞；局部避障成功仍可能因里程计误差累积而最终偏离路线。
- **方法：** 深度图 + 点目标条件的 Action Predictor (AP) 预测短期碰撞安全路点与修正后的子目标位姿；Collision Predictor (CP) 评估近场碰撞风险，高风险时把当前子目标位姿偏差传播给后续拓扑子目标。
- **模型：** EfficientNet-B0 编码深度/目标特征，位姿编码器表达机器人与子目标配置，GRU 捕获时间依赖。训练数据约 10K 无碰撞与 10K 碰撞样本。
- **结果：** 六个未见校园地点的九张拓扑图上，平均 Node Coverage (NC)=0.89、Success weighted by Path Length (SPL)=0.80；T-B6 中只有 DevGRU 完成全程。
- **效率：** 19.44M 参数、平均推理 50.01 ms；相对 NavDP 约小 7×、快 17×。延迟是在 Ryzen 9 6900HX + RTX 3070 笔记本上测得，不能直接视为任意嵌入式设备的实时指标。
- **开源：** 公开代码仓提供训练与机器人部署流程，并链接预训练权重和数据集。许可证仅授权教育、研究及非营利用途。

## 对 Wiki 的映射

- **wiki/entities/paper-devgru.md**：建立统一论文 / 项目 / 代码实体，归纳 AP、CP、子目标漂移修正、评测与运行入口。
- **对照：** [NoMaD](../../wiki/entities/paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md) 与 [NavDP](../../wiki/entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md) 提供策略建模与安全处理对照。
