# ObjDream: Guiding Vision-Language-Action Models with Object-Level Understanding and Future State Prediction（arXiv:2610.20223）

> 来源归档（ingest）

- **标题：** ObjDream: Guiding Vision-Language-Action Models with Object-Level Understanding and Future State Prediction
- **arXiv：** <https://arxiv.org/abs/2610.20223>（Submitted 2026-10-14，v1；PDF：<https://arxiv.org/pdf/2610.20223>）
- **项目页：** <https://huailiangma.github.io/objdream/>（页内 Paper 指向 arXiv；Code 按钮指向 <https://github.com/HuailiangMa/ObjDream>，标注 **coming soon**）
- **代码（页面链接，未核实）：** <https://github.com/HuailiangMa/ObjDream> — 本次会话无 GitHub 访问授权，仓库是否存在、是否公开均 **未核实**
- **作者：** Huailiang Ma, Yuxuan Li, Ziyi Zhao, Xinkai Wang, Mingyu Li, Xinyu Guo, Lifeng Zhu, Aiguo Song, Cewu Lu, Jianhua Sun（通讯）
- **机构（PDF 首页）：** 东南大学（Southeast University）；上海创智学院（Shanghai Innovation Institute）；上海交通大学（SJTU）；中国科学院自动化研究所（Institute of Automation, CAS）
- **入库日期：** 2026-10-15
- **类型：** paper / vla / object-centric / future-prediction / libero / robotwin
- **一句话说明：** 把 VLA 的视觉输入从图像级表征改为时空物体槽（身份通道 + 状态通道），用身份引导路由调制 VLM 注意力，并在训练期预测动作条件下的未来物体状态；推理时不需要未来预测。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-15）：** Code 按钮文字为 "coming soon"，链接目标为 github.com/HuailiangMa/ObjDream。
- **仓库核查：** 本会话无 GitHub 读取授权（API 返回 access not enabled），无法确认仓库存在、可见性或内容。
- **论文核查：** PDF 中未见代码发布声明。
- **结论：** **待发布**（项目页标注 coming soon；仓库状态未核实，不能判为已开源）。

## 摘录 1：问题与主张（Abstract / §I）

- **痛点：** 多数 VLA 仍使用 VLM 继承的图像级表征，缺乏细粒度物体理解；物体布局、杂乱与背景变化下鲁棒性下降。
- **主张：** 把持久的物体身份与动态物体状态分开表示；身份用于路由 VLM 注意力；训练时预测动作条件下的未来物体状态作为辅助监督。
- **贡献声明（§I）：** (i) 物体中心 VLA 框架，时空物体编码器分离身份与状态，身份引导路由；(ii) 动作条件物体动力学模块，仅训练期预测紧凑的未来物体状态，并建模物体间关系；(iii) LIBERO / LIBERO-Plus / RoboTwin 2.0 与真机评测。

## 摘录 2：实验要点（§4，数字取自 Table 1–3 与 Fig. 4）

| 基准 | 结果 |
|------|------|
| LIBERO（平均） | **98.6%**（Spatial 99.2 / Object 99.6 / Goal 98.2 / Long 97.4） |
| LIBERO-Plus（零样本，总分） | **84.4%**，排名第二；π0.5 为 85.7，OA-WAM 为 83.9，VLA-JEPA 为 79.5；语言 90.6、光照 98.4、背景 98.5、噪声 88.6；相机视角 66.2、机器人初始化 75.8、布局 81.2 |
| RoboTwin 2.0（8 任务子集，50 clean demos/任务） | Easy **60.25%**，Hard **22.75%**；相对 StarVLA 分别 +12.87 与 +15.62 个百分点 |
| 消融（Table 2，LIBERO-Plus 总分） | 骨干基线 81.2；完整模型 84.4；去掉 Fact. Injection 82.5；去掉 Id. Routing 82.6；去掉 ObjDyn 83.2；去掉 ObjRel 84.1 |
| 真机 ID（Table 3，4 任务平均） | ObjDream **74%**；π0.5 66%；ControlVLA 60% |
| 真机 OOD | ObjDream **55%**；π0.5 53%；ControlVLA 31% |

- 骨干：4B 规模的预训练 VLM（具体型号见原文），采用 StarVLA 训练配方与 layerwise flow-matching action expert。
- 物体编码器：离线自动标注（VLM 生成实例短语）预训练后冻结，VLA 训练期间不更新。
- 训练：8 张 H100；评测推理：RTX 4090。

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-objdream.md`](../../wiki/entities/paper-objdream.md)：强调「物体身份/状态分离 + 训练期未来物体状态监督」与 VLA 鲁棒性的关系；开源状态写为待发布（仓库未核实）。
- 与 [VLA 方法页](../../wiki/methods/vla.md)、[StarVLA](../../wiki/methods/star-vla.md)、[π0.5 页](../../wiki/entities/paper-pi05-open-world-vla.md) 互链。
