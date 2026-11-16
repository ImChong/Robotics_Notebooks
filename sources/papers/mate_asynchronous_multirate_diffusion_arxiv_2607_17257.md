# MATE：异步多速率扩散策略组合（arXiv:2607.17257v2）

> 来源归档（primary paper · 通读）

- **标题：** MATE: Asynchronous Multi-Rate Diffusion Policy Composition
- **旧题名：** Asynchronous Multimodal Diffusion Policy Composition via Latency-Aware Guidance Fusion（v1；LAG-Fusion）
- **作者：** Zihao He, Hongjie Fang, Shirun Tang, Cewu Lu, Hao-Shu Fang
- **机构：** 上海交通大学（Shanghai Jiao Tong University）、Noematrix、上海创智学院（Shanghai Innovation Institute）；作者页另有 independent researcher 标记
- **类型：** paper / diffusion-policy / asynchronous-control / multimodal-composition / contact-rich-manipulation
- **arXiv：** <https://arxiv.org/abs/2607.17257>
- **通读版本：** arXiv:2607.17257v2，2026-11-06，<https://arxiv.org/html/2607.17257v2>
- **项目页：** <https://lag-fusion.github.io/>
- **代码开放：** 2026-11-10 核查项目页：`code coming soon`，尚无官方可运行仓库
- **入库日期：** 2026-11-10
- **核对范围：** 通读 v2 Abstract、I Introduction、II Related Works、III Method（III-A/B/C，式 1–12）、IV Experiments（IV-A/B/C/D，Table I/II、Fig 1–6）、V Conclusion 及 Limitations、References；附录 VI（VI-A/B/C，Table III、式 13、运行时）与 VII（VII-A/B/C、Fig 7–11、failure analysis）。并打开官方项目主页核对在线实验、代码状态和重命名信息。
- **通读：** 已完成
- **一句话说明：** 将独立训练的视觉、力觉或不同视角扩散策略按各自原生推理频率异步运行，经时间重叠选择与扩散变量参考帧重映射组合去噪指导，无须联合重训。

## 核心机制（原文 `III）

1. **适用条件**：各分支共享兼容动作空间和扩散参数化；不同查询时刻、动作时域和参考坐标系可以不同；预测时域要有重叠。
2. **动作定义**：每步 10D 相对动作 = 三维平移 + 列式 6D 旋转（前两列）+ 夹爪标量；基于两个查询时刻末端位姿得 `T_{2←1}=T_2^{-1}T_1`。
3. **几何重映射**（式 4–8）：干净动作 `a'=Aa+b`，扩散噪声样本 `a'_k=Aa_k+√(ᾱ_k)b`，噪声预测 `ε'_k=Aε_k`。`A` 为正交块旋转，保留高斯噪声协方差；不能把平移偏置直接加到预测噪声。
4. **异步组合**（式 9–12）：截取重叠时段；旧预测与当前采样重映射至公共参考帧；用非负归一权重组合噪声；映射回完整时域，替换重叠段，再 DDIM 去噪。慢策略的最近预测可以缓存复用。
5. **时域权重**（附录 VI-C，式 13）：视觉权重沿重叠段从 0.8 线性降到 0.2，力分支取互补权重；该示例权重随动作时域位置变化，不等同于一个泛化学习到的权重网络。

## 实验与参数（原文 `IV + 附录 VI–VII）

- 平台：Flexiv Rizon4、FT03S 六轴力/力矩传感器；每任务 50 条遥操作示范；图像/深度采集 20 Hz，低维数据采集 1000 Hz，注意这与**策略推理频率**不同。
- Flip、Polish Curve：3D 视觉策略（RISE-based）5 Hz + 力觉策略 25 Hz，合成后平均推理时间约 40 ms。后者使用 1 s 100 Hz 的六轴力历史（100×6），预测 1 s / 50 步、50 Hz 动作轨迹。DDPM 训练 100 步、DDIM 推理 10 步。
- Table I：Flip 项目中 MATE 的 **Flip 得分 60.0%**（同步组合 25.0%，RDP 35.0%）；Polish Curve **Score 55.0%**（同步组合 25.0%，RDP 45.0%）。表中 Flip 的“部分翻转”计半次成功；Polish 是按完成程度分档得分，均不宜泛称二元成功率。
- 遮挡抓取放置（Occluded PnP）：全局 3D 视觉 2 Hz + 腕部 2D 视觉 5 Hz，MATE Reach/Pick/Place 为 100% / 90% / 85%（Table II）；相对同步 90% / 80% / 75%。
- Fig 5：固定视觉频率，力策略频率升高时接触任务得分提高；附录接触阶段独立力策略频率研究为不同实验设置，不能与完整任务指标混同。
- 评测范围：每任务 20 次测试，初始物体摆放固定集合；20 次样本和单台机械臂设置限制统计外推。

## 对比、局限与失效（`II / `V / VII-C）

- 区别于 Joint multimodal policy：不把高频力信号锁在慢视觉的推理节拍；区别于 RDP/慢快架构：分支可以独立训练、部署时组合；区别于同步 PoCo/GPC：需要对异步查询的相对动作参考帧做几何对齐。
- 限制：仅验证扩散策略，Flow Matching 扩展尚属未来工作；实验用平行夹爪，未验证灵巧手；各分支须有相容的动作与噪声表示。
- 失败观察：无接触阶段力信号噪声可导致漂移；慢反馈可错过翻转时机；视觉策略可能在尚未接触时切换阶段；持续打磨时低频策略难以维持接触力；PnP 遮挡仍可能抓取失败。
- 开源：截至 2026-11-10，官方项目页显示 `code coming soon`；本记录**未**声称已核验实际代码或完整复现。

## 对 wiki 的映射

- [MATE 论文实体](../../wiki/entities/paper-mate-asynchronous-multirate-diffusion.md)
- [Diffusion Policy 方法](../../wiki/methods/diffusion-policy.md)

## 一手资料

- [论文 v2 HTML](https://arxiv.org/html/2607.17257v2)
- [MATE 官方项目页](https://lag-fusion.github.io/)
