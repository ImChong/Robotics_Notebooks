# WorldEcho / WorldSync（动作条件世界模型的动作跟随评测与对齐）

> 来源归档（ingest）

- **标题：** Do Robotic World Models Really Follow Actions? Diagnosing and Aligning Action-Conditioned Generation for Policy Learning
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.24885>
- **作者：** Sixiang Chen、Jiaming Liu、Jixian Wu、Yichen Guo、Tinghao Wang、Siyuan Qian、Hao Chen、Jiajun Cao、Jian Tang、Shanghang Zhang
- **机构：** 北京大学多媒体信息处理国家重点实验室 / 计算机学院；北京人形机器人创新中心；纽约大学；电子科技大学；南洋理工大学；香港中文大学
- **项目页 / 代码：** 论文与公开检索均未列项目页或 GitHub
- **入库日期：** 2026-08-27
- **一句话说明：** **WorldEcho** 用视觉完整性门控 + \(\mathrm{SE}(3)\) 末端 NDTW 评测专家与四类 off-expert 动作跟随；**WorldSync** 用覆盖扩展 + Action-Forcing Expert + Intervention-Effect 监督把 Cosmos 系 AC-WM 训成更可信的策略改进模拟器。

## 核心摘录（MVP）

### 1) 专家榜掩盖 off-expert 失败：视觉崩 vs 动作无视

- **摘录要点：** 动作条件世界模型被当成策略评估/后训练模拟器，前提是「任意合法动作都会被忠实生成」。现有基准多停在专家演示。WorldEcho 把查询扩到 Demonstrated / Cross-State Replay / Local Perturbation / Policy Rollout / Feasible-Space Sampling 五类，并在 RoboTwin 回放得到动作特异真值。联合评 **视觉门控**（MUSIQ 画质、插帧平滑、EEF 可见、臂完整性）与 **pose-aware NDTW**；门失败则赋固定惩罚 \(\kappa\)。六套专家训模型 off-expert 门控误差升 **0.029–0.099 m**，视觉失败率升 **6.3–28.1 pt**：要么手臂扭曲/夹爪消失，要么画面好看但不跟命令。
- **对 wiki 的映射：**
  - [WorldEcho / WorldSync](../../wiki/entities/paper-worldecho-worldsync.md) — 评测协议与失败模式。
  - [生成式世界模型](../../wiki/methods/generative-world-models.md) — AC-WM 作策略模拟器的前提。
  - [评测选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) — ② 层「动作忠实 ≠ 视觉逼真」。

### 2) WorldSync 三轴：覆盖、AFE 接地、IE 介入效应

- **摘录要点：** 仿真专家+off-expert 轨迹与少量真机演示在共享基座系相对 \(\mathrm{SE}(3)\) 末端增量空间混合。视频骨干 flow matching。**AFE** 从视频中间特征解码未来末端轨迹，推理时去掉。**IE** 用同观测、同噪声、不同动作的配对，对齐 \(\Delta_\theta=v_\theta^A-v_\theta^B\) 与真值 \(\Delta^*=x_0^B-x_0^A\)。消融：IE 主拉轨迹（raw NDTW 最低 0.0170）；AFE 单独不降动作误差，但和 IE 一起把门控误差压到 **0.0695**。
- **对 wiki 的映射：**
  - [WorldEcho / WorldSync](../../wiki/entities/paper-worldecho-worldsync.md) — 训练配方。
  - [Ctrl-World](../../wiki/entities/paper-ctrl-world.md) — 主对照骨干与策略改进基线。

### 3) RoboTwin 50 任务与两轮策略改进

- **摘录要点：** 50 任务宏平均：WorldSync 门控误差 **0.0661**、视觉通过 **84.51%**（略优于 Expanded CtrlWorld 0.0670 / Motus 84.34%）；Cosmos-Predict2.5 Expanded 的 raw NDTW 更低（0.0127 vs 0.0223），WorldSync 赢在门控平衡而非单项碾压。VLAW 式两轮改进、预算对齐：RoboTwin bin-dumping **51–52%→65%**（CtrlWorld 约 +5 pt）；真机叠杯 **48%→68%** vs CtrlWorld **56%**。
- **对 wiki 的映射：**
  - [WorldEcho / WorldSync](../../wiki/entities/paper-worldecho-worldsync.md) — 数字读法。
  - [动作后果分类 04](../../wiki/overview/wm-action-consequence-category-04-eval-posttrain.md) — 评测→后训练闭环。

### 4) 开源状态（截至 2026-08-27）

- **摘录要点：** 无项目页、无 GitHub、无权重 URL；正文未承诺发布代码。按步骤 2.5：**确认未开源**。
- **对 wiki 的映射：**
  - [WorldEcho / WorldSync](../../wiki/entities/paper-worldecho-worldsync.md) — 局限节点明不可复现。

## 当前提炼状态

- [x] arXiv HTML 方法/表/消融已对齐摘录
- [x] 确认无官方代码或项目页
- [x] wiki 映射：`wiki/entities/paper-worldecho-worldsync.md` 新建
