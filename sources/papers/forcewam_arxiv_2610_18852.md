# ForceWAM: Decoupling Future and Response Learning from Demo and Play Data for Contact-Rich Manipulation（arXiv:2610.18852）

> 来源归档（ingest）

- **标题：** ForceWAM: Decoupling Future and Response Learning from Demo and Play Data for Contact-Rich Manipulation
- **arXiv：** <https://arxiv.org/abs/2610.18852>（HTML：<https://arxiv.org/html/2610.18852>）
- **项目页：** <https://force-wam.github.io/>（归档：[forcewam-project.md](../sites/forcewam-project.md)）
- **代码：** <https://github.com/SII-ZijunX/ForceWAM>；截至 2026-10-30 仅有占位 README，实现待发布（[仓库归档](../repos/forcewam.md)）
- **作者：** Zijun Xu、Wendi Chen、Xinkai Wang（共同一作）等 19 人，通讯作者 Lixin Yang；含 Chuan Wen、Jieru Zhao、Wenchao Ding、Cewu Lu
- **机构（项目页）：** 复旦大学；上海交通大学；东南大学；上海创智学院（Shanghai Innovation Institute）；南方科技大学
- **入库日期：** 2026-10-17
- **文章来源：** 具身智能小站 2026-10-17 公众号文章（见 sources/blogs 由主编入库）
- **一句话说明：** 把「任务相容的未来」与「动作如何改变接触」拆成两个模型：成功示教训练世界模型预测视频 + 六轴力矩未来，任务数据与无任务 play 数据训练逆动力学模型，部署时用新力反馈把未来转成动作。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** 页面仅有 "Code coming soon"，无仓库链接。
- **2026-10-30 更新：** 官方仓库已公开地址，但仅有 README，训练及推理实现仍待发布。本次未能访问项目页和论文全文，原有方法及实验摘录尚待以 [v1 全文](https://arxiv.org/abs/2610.18852v1)复核。

## 摘录 1：方法（摘要 + arXiv HTML）

- **ForceWM：** 由 Cosmos Predict 2B 权重初始化的视频扩散模型，联合预测未来腕部相机视频潜变量与六轴力/力矩；五槽潜变量（空白 / 当前力矩 / 当前图像 / 未来力矩 / 未来视觉）；仅在未来槽上计算 EDM 去噪损失；8 GPU 训练 20,000 iter。
- **IDM：** ResNet-34 图像编码、GRU 因果力矩编码、7 层 Transformer 解码、18 个动作 token、14 维相对位姿动作；输入「当前观测 + 记录的视觉-力矩未来」，用同段数据的动作作监督。
- **慢-快执行器：** ForceWM 每 6 个动作刷新一次参考，其间 IDM 以缓存噪声与不断增长的实测力矩前缀重新去噪，只执行当前动作；ForceWM 约 100 ms，IDM 约 60 ms（5 步 DDIM）。
- **关键观点：** play 轨迹不必完成任务，也能教会「动作如何改变接触」。

**对 wiki 的映射：** [`paper-forcewam.md`](../../wiki/entities/paper-forcewam.md)；与 [World Action Models](../../wiki/concepts/world-action-models.md)、[逆动力学模型](../../wiki/concepts/inverse-dynamics-model.md) 互链。

## 摘录 2：实验（项目页）

- 机器人：Flexiv Rizon 4S 7 自由度臂；末端六轴 F/T（型号未给出）；Hanoi 用 Robotiq 2F-85，Slide/Flip 用带柔顺指尖的摇杆工具（arXiv HTML）。
- 每任务族 200 条示教 + 200 条 play；Flip 与 Flip-D 共享数据；每项评测 20 次。
- 成功率（Slide / Flip / Flip-D / Hanoi）：ForceWAM 全 play IDM 85 / 95 / 85 / 90；仅任务数据 IDM 40 / 75 / 65 / 75；最强基线各列约 20 / 60 / 40 / 65（FA-WAM、Fast-WAM 等）。
- 对 ImplicitRDP、Cosmos Policy 直接加 play 数据反而下降（作者归因于未来学习与响应学习需要分离目标）。

**对 wiki 的映射：** 评测与结论小节；play 数据的价值读法。
