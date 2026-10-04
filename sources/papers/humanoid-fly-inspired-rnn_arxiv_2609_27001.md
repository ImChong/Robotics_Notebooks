# Humanoid Locomotion with a Fly-Inspired Recurrent Controller（arXiv:2609.27001）

> 来源归档（paper）

- **标题：** Humanoid Locomotion with a Fly-Inspired Recurrent Controller
- **作者：** Isabel Guan、Yuntian Zhao、Dingyuan Zhang、Shipeng Lyu
- **机构：** 香港科技大学（HKUST）、Zenbot、南洋理工大学（NTU）、香港理工大学（PolyU）
- **arXiv：** <https://arxiv.org/abs/2609.27001>
- **HTML：** <https://arxiv.org/html/2609.27001>
- **PDF：** <https://arxiv.org/pdf/2609.27001>
- **提交日期：** 2026-09-22（v1 标注 2026-09-22；HTML 日期 2026-09-23）
- **论文状态：** arXiv 预印本；页面标注为模拟研究、尚未同行评审
- **项目页 / 代码 / 权重：** 未找到作者公开链接；论文的数据可用性说明称作者持有的评测归档尚无公共归档标识，推理包受权限限制
- **一句话说明：** 在 MuJoCo Unitree G1 仿真中剖析一个 3,609 连续状态、果蝇神经元标签启发的循环控制器，使用地形/速度/yaw 条件测试与状态干预研究实际生效的控制路径。

## 开源与复现核查

- arXiv HTML 的 S6 指出：检查到的材料包括 T_graph ONNX checkpoint、部署接口、机器人模型、地形描述和 t1_w5 神经元标注。
- 同一节说明：原始图提取代码、所选生物连接关系、teacher 来源、训练目标与配置、训练/验证划分均未提供；`w5` 的筛选含义也未说明。
- “Data availability”节称作者持有的 protocol、147 个 episode 摘要、采样轨迹、干预记录、分析脚本、模型哈希及可视化记录；但公共归档 ID 尚未建立，完整复跑需要另行授权的 inference bundle。
- 因此按**未公开可复现包 / 部分可检查材料**记录；不要把论文中的模型描述当成已发布的 GitHub 代码或可下载权重。arXiv 页面未给出项目站、GitHub、Hugging Face 或数据集下载链接。

## 核心摘录

1. **研究对象与问题：** 工作分析一个现成的 T_graph checkpoint 如何经输入投影、循环核心、运动神经元标签读出及关节伺服连接到模拟 G1；问题重点是部署行为实际依赖哪些通路，而非证明果蝇连接组结构本身优于其他网络。
   **对 wiki 的映射：** [paper-humanoid-fly-inspired-rnn](../../wiki/entities/paper-humanoid-fly-inspired-rnn.md)「核心机制」「局限与风险」。

2. **控制器与接口：** 核心包含 3,609 个连续状态；135 个 motor-neuron-labelled 项经读出映射为 15 个腿部和腰部目标，另外 14 个手臂关节固定保持。82 维本体/指令输入包含重力投影、角速度、平面/偏航速度指令、关节位置与速度、上一时刻 raw action。另有 256 维状态的深度处理支路输出 128 维 bottleneck。
   **对 wiki 的映射：** [paper-humanoid-fly-inspired-rnn](../../wiki/entities/paper-humanoid-fly-inspired-rnn.md)「控制器与身体的闭环」。

3. **评测与干预：** 固定 checkpoint 在 7 种地形、3 个速度（0.3 / 0.5 / 0.8 m/s）、3 个初始 yaw 偏移上评估，共 63 个条件；按“存活 12 秒且前进距离达阈值”计，T_graph 完成 61/63，带 187 个特权地形高度样本的 R1 参考为 62/63。名义 yaw 条件下每次策略调用前把整个循环状态清零，成功从 19/21 降到 0/21。252 个记录状态上的深度替换及 upstream-state 清零不改变 raw action；本实验运行中记录到的 descending bottleneck 输出为零。
   **对 wiki 的映射：** [paper-humanoid-fly-inspired-rnn](../../wiki/entities/paper-humanoid-fly-inspired-rnn.md)「实验与评测」「结论」。

4. **复现边界：** 策略为 50 Hz，MuJoCo 物理步进为 1 kHz。成功率标准不包含侧向偏移；论文指出一个 0.8 m/s 离散台阶 rollout 虽达成前进距离阈值，终点仍横向偏离 3.17 m。实验只覆盖固定仿真 checkpoint 和指定场景，不是 Unitree G1 真机验证。
   **对 wiki 的映射：** [paper-humanoid-fly-inspired-rnn](../../wiki/entities/paper-humanoid-fly-inspired-rnn.md)「工程读法」「局限与风险」。

## 相关一手资料

- [MaleCNS v1.0 下载/文档](https://male-cns.janelia.org/download/) — 论文所引的果蝇中枢神经系统连接组资料；不是 T_graph 控制器本身。
- [arXiv PDF](https://arxiv.org/pdf/2609.27001)
- [arXiv HTML 正文与补充材料](https://arxiv.org/html/2609.27001)
