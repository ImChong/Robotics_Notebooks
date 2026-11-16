# STEP：用于感知式人形运动的显式预测时空训练（arXiv:2611.08097）

> 来源归档（paper）

- **标题：** STEP: Spatiotemporal Training with Explicit Prediction for Perceptive Humanoid Locomotion
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.08097>（v1，2026-11-05，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.08097>
- **作者：** Youhao Lyu、Chenghao Sun（共同一作）、Zhenxuan Ding、Shuting Zhang、Ziyang Dong、Wenhao Cui（通讯）
- **机构（论文首页脚注）：** 中国科学技术大学（USTC，第一作者）；松延动力 Noetix Robotics（北京，Sun / Dong / Cui）；北京化工大学（BUCT，Ding）；国防科技大学（NUDT，Zhang）
- **代码：** 论文正文未给出代码或项目页链接（已逐段检索 github / project / code / video 关键词，仅命中无关词）；开源状态：未见公开
- **入库日期：** 2026-11-09
- **一句话说明：** 单阶段训练的深度 + 本体感知人形运动框架：四个可学习查询把两种模态聚合为 64 维策略状态，训练时再加双向掩码跨模态预测与动作条件下一状态预测，部署时去掉预测头；23 自由度 Noetix E1 实机 98/100 次成功。

## 核对范围

- **版本：** arXiv v1 PDF（9 页）；用 pdftotext 抽取全文逐段读，Table III、Table V、Fig. 3、Fig. 4 对照渲染页面图核对。
- **已读：** 摘要、引言、相关工作、方法（III-A 至 III-C，含式 (1)–(12)、Table I 网络设置、Table II 奖励项）、实验（IV-A 至 IV-D，Table III–VI、Fig. 3–4）、结论、参考文献列表。
- **未读 / 缺项：** 论文无附录与补充材料（全文 9 页止于参考文献）；未读被引论文（AME-2、CReF、LeJEPA、Kepler 等）原文；未运行任何代码（正文也无代码）；Fig. 3(a) 的延迟数值只能从散点图读出近似值，论文未给表格。
- **通读：** 已完成

## 核心摘录

1. **问题设定（摘要、I 引言）：** 作者认为特征聚合本身不规定应学哪些跨模态关系，同时刻观测之间的预测又不显式约束动作驱动的状态演化；STEP 在共享控制表征中同时监督两者（作者自述）。
2. **聚合（III-B，Table I，式 5）：** 5 帧 × 77 维本体历史 + 16×24 深度图（深度 CNN 输出 8×12 = 96 个 token，宽度 32 线性升到 64）形成共享 key-value；4 个与输入无关的可学习查询 Q0 经 2 个残差交叉注意力块（各 4 头）更新，取平均得 64 维策略状态 s_t。
3. **跨模态预测 CM（III-C，式 6）：** mask-depth 只读 5 个本体 token 得 s^{-d}，mask-proprio 只读 96 个深度 token 得 s^{-p}；两路分别经预测头回归 EMA 教师的深度投影（4×32）与本体投影（32），两方向等权。
4. **动作条件下一状态预测 Next（III-C，式 7–9）：** g_φ([s_t, sg(a_t)]) 预测下一步 EMA 教师在未投影 64 维空间里的融合状态（87→64→64，GELU）；动作先裁剪到 [-18, 18]。
5. **总损失与稳定项（式 10–12）：** L_main = L_PPO + α_k·L_Next + L_CM + 0.02·R_modal + 0.005·R_joint，α_k = 0.1·min(k/200, 1)；SIGReg（引自 LeJEPA）防表征塌缩；教师 EMA 系数 0.99/0.01。预测头与教师部署时全部去掉。
6. **actor 与速度估计（III-A，式 1–2）：** 稠密 Soft-MoE（4 专家）输出 23 维动作；输入 x_t = [当前本体 o_t, sg(速度估计 v̂_t), s_t] ∈ R^144；Velocity Estimator 单独优化、仅向 actor 提供 detach 的速度；critic 用特权信息；奖励为任务奖励与 AMP 风格项组合（式 3，Table II）。
7. **仿真设置（IV 开头，Table IV）：** Isaac Lab + RSL-RL 的 PPO，每个配置用 2048 个并行环境评测；课程含粗糙地面、平台、上 / 下楼梯、沟隙、踏脚石六类地形。
8. **仿真对比（IV-A，Table III）：** STEP 整体均值 95.17%，w/o Prediction 91.02%，w/o Multi Query 69.98%，Proprio-Query 86.17%；差距主要在 L8 下楼梯与踏脚石。
9. **深度退化（IV-C，Table V，L6）：** 持续深度丢帧时所有方法在沟隙与踏脚石上成功率都是 0；一步（20 ms）深度延迟下 STEP 在五种地形全部领先。
10. **实机（IV-D，Table VI）：** Noetix E1（23 自由度，机载 IMU / 关节编码器 / 机身深度相机，俯仰 60°），5 项任务各 20 次、无人工干预，共 98/100；两次失败都在 70 cm 沟隙（18/20）。

## 开源状态

- 正文无代码 / 项目页 / 数据集链接；未执行项目页核查（无项目页可查）。**未见公开**（2026-11-09 依论文正文判断）。

## 与周报说法的核对

| 周报（eiBI）说法 | 论文核对 |
|------------------|----------|
| 机构：中科大、松延动力、北京化工大学、国防科大 | 与首页脚注一致 |
| 双向掩码预测学习深度与本体对应关系 + 动作条件预测器预测下一融合状态 | 一致（III-C） |
| 深度丢帧和延迟下仍保持稳定 | 过度概括：仅「一步延迟」下全面领先；「持续丢帧」下沟隙与踏脚石全部为 0（IV-C，Table V），且此项只在仿真里测 |
| 实机在楼梯、40 cm 平台、70 cm 沟隙、踏脚石完成 98/100 | 数字对；另含第 5 项「15 cm 台阶倒走（后方无地形视野）20/20」，周报漏列；70 cm 沟隙为 18/20（Table VI） |

## 对 wiki 的映射

- [paper-step-spatiotemporal-humanoid-locomotion](../../wiki/entities/paper-step-spatiotemporal-humanoid-locomotion.md)
