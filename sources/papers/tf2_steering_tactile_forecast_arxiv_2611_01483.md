# TF²-Steering：基于触觉预测与反馈的速度场动作精修（arXiv:2611.01483）

> 来源归档（paper）；直接阅读作者论文，不以二手盘点替代原文。

- **标题：** TF²-Steering: Action Refinement with Tactile Forecasts and Feedback via Velocity-Field Steering
- **作者：** Yiming Chen、Xiaofei Wei、Chun Gu、Wei Xiao、Peidong Liu、Li Zhang
- **机构：** 浙江大学（Zhejiang University）、上海创智学院（Shanghai Innovation Institute）、西湖大学（Westlake University）、复旦大学（Fudan University）
- **类型：** paper / tactile / flow-matching / dexterous manipulation
- **版本：** arXiv:2611.01483v1，2026-11-01
- **arXiv：** <https://arxiv.org/abs/2611.01483>
- **PDF：** <https://arxiv.org/pdf/2611.01483>
- **作者发布的全文 HTML：** <https://arxiv.org/html/2611.01483>
- **项目页：** <https://yiming-cc.github.io/tf2-steering/>
- **初次入库：** 2026-11-04（来自[具身智能小站 57 篇索引](../blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)）
- **原文核对日期：** 2026-11-10
- **核对范围：** 通读 arXiv 官方 HTML 版 v1：Abstract、§1 Introduction、§2 Related Work、§3 Method（3.1–3.3，公式 1–6）、§4 Experiments（4.1–4.3，表 1–3）、§5 Conclusion、AI Use / Reproducibility Statements、附录 A.1–A.4（表 4–7、消融配置、真机动作展示说明）。PDF 原始文件较大，工具未能直接加载，全文内容以官方 HTML 版本为核对基准。
- **通读：** 已完成

## 核心机制（对照 §3）

1. **视觉 WA 主干：** LTX-Video 视频世界模型（带多视角交叉注意力）→ Q-Former 压缩视觉上下文 → Flow Matching action expert 生成 action chunk；视觉流不接收触觉特征。
2. **触觉预测：** 约 500M 参数的 tactile world model 以当前触觉及视频特征为条件预测未来触觉，推理时取其末层特征作为 cached prior，不必解码预测触觉图像。
3. **触觉修正：** tactile expert 内有 predicted-tactile / real-time-tactile 两个子专家，以各自 sigmoid 权重融合；单向读取 action expert 特征并 stop-gradient，在每次 Flow Matching 迭代输出速度残差：v_corr = v_base + Δv。修正目标在生成阶段的**速度场**，不是生成完成后对最终动作加残差。
4. **异步双速率：** 世界模型低频生成并固定 chunk 期间的视觉与触觉先验；高频执行每 k 步接收新触觉，重新生成剩余未执行的动作。训练模拟反馈延迟 d，使用窗口 W 与相对位置编码 ρ_i = i − d。
5. **独立损失：** action expert 优化 L_act = ||v_base − v*||²；tactile expert 优化 L_steer = ||v* − (sg[v_base] + Δv)||²。stop-gradient / 单向注意力避免触觉损失反向破坏视觉基策略。

## 训练配置（§3.3；附录 A.1）

- **视觉-动作主干预训练（先行）：** AgiBot World、RoboMIND、Open X-Embodiment、RoboCOIN；100k steps，global batch 512；LTX-Video + Q-Former + action expert；128 维统一 action/state（含无效位 mask）。
- **Stage 1 触觉预测预训练：** 冻结视觉主干，只训触觉世界模型；OpenNeoData / T-REX / FTP-1-Sharpa 按 30% / 40% / 30% 混合；50k steps，global batch 1024。
- **Stage 2 触觉 steering 预训练：** 联合训练视频世界模型、Q-Former、action expert、tactile world model、tactile expert；触觉数据 90%，原视觉动作数据 10%；50k steps，global batch 512；触觉 expert 的 Δv 输出头零初始化。触觉偏移采样 0/4/8/12 steps、steering window 10 steps，训练动作采样 30 Hz、horizon 50。
- **Stage 3 任务后训练：** 保持 L_pred^v + L_pred^τ + L_act + λ_steer L_steer 联合目标，在仿真或真机演示数据上适配。

## 实验核验（§4；附录 A.2–A.3）

| 设置 | TF²-Steering | 最强对照 | 口径 |
|---|---:|---:|---|
| UniVTAC 六项仿真 | 68.3% | FTP-1：57.7% | 六任务平均成功率，作者评测方法每任务 50 trials；摘要报 +10.7 pp（未四舍五入均值），表中一位小数直接相减为 +10.6 pp |
| 六项双手真机 | 71.5% | T-Rex：62.4% | 六任务平均**进度成功率 PSR**，各任务 20 trials；+9.1 pp，**不是严格二值成功率** |
| 三项真机消融 | 72.3% | 去双触觉：61.7%；动作空间残差：61.3% | 三任务 PSR，不可与六任务 71.5% 混用 |
| 其他消融 | 去实时触觉：63.0%；去预测触觉：69.7%；去异步：66.7%；去 attention mask+stop-grad：62.7% | 完整模型 72.3% | 三任务 PSR |

- 仿真任务：Lift Bottle、Pull Out Key、Lift Can、Put Bottle、Insert Hole、Insert Tube；单视角四项 / 多视角两项各自一组 checkpoint。
- 真机任务：Fruit on Plate、Close Zipper、Wipe Whiteboard、Twist Cap、Split Cups、Dropper Transfer。双 UR5e（每臂 6 DoF）+ 两只 Sharpa 灵巧手（每手 22 DoF）；D455 顶部相机 + 两台 D405 腕部相机，Meta Quest / Manus 遥操，每任务 100 条 30 Hz 演示，共 600 条。
- 进度得分是各任务里程碑式 0–1 分，按 N=20 次的平均 ×100 得到 PSR；部分完成也能得分，指标不等同于全流程硬成功率。
- 论文报告单张 RTX 4090 上生成一段动作约 240 ms；文中未给可复现的端到端双速率部署代码或实时安全保证。

## 原文局限与审读提示

- 结论限定在 UniVTAC 与实验室双手桌面任务；没有证明未知物体、移动双足平台或人机共域的安全鲁棒性。
- 真机每任务 20 次，主指标是 partial-credit PSR；没有在文内报告显著性检验，不宜直接宣称相同幅度的二值任务完成率提升。
- 双世界模型与触觉专家、三阶段预训练有较高系统 / 数据成本，异步推理需要保证时间戳、延迟窗口及未执行动作的一致性。
- 项目源码和可下载权重状态须以作者项目页实时公布为准；论文的训练参数不等价于已可复现代码。

## 对 wiki 的映射

- [TF²-Steering 唯一实体节点](../../wiki/entities/paper-tf2-steering-tactile-forecast.md)
- [触觉感知](../../wiki/concepts/tactile-sensing.md)
- [接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)
