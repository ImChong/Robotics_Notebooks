# DobotTrack: Target-Grounded Vision-Language-Action Learning for Embodied Person Tracking（arXiv:2611.09041）

> 来源归档（原文通读 · 支撑 `wiki/entities/paper-dobottrack.md`）

- **标题：** DobotTrack: Target-Grounded Vision-Language-Action Learning for Embodied Person Tracking
- **类型：** paper / embodied-visual-tracking / vla / person-following / quadruped
- **arXiv：** <https://arxiv.org/abs/2611.09041>（v1，2026-11-06，cs.RO；PDF：<https://arxiv.org/pdf/2611.09041v1>，共 12 页）
- **作者：** Shengjun Deng、Yinghan Zhang*、Chengrui Wei、Liu Liang、Songyuan Liu、Pengkun Zheng†、Yajun Zheng、Ying Zhang、Yuan Tian、Xi Li、Teng Wang*（*通讯，†项目负责人）
- **机构（论文首页）：** 佛山大学；清华大学；Dobot Robotics（越疆）；香港大学
- **代码：** <https://github.com/Dobot-Team/dobot_track>（README 静态读取，见下「开源状态」）
- **项目页：** <https://dobot-team.github.io/dobottrack-web/>（本次未能核实页面内容，仅在 README 徽章中见到链接）
- **入库日期：** 2026-11-10
- **一句话说明：** 在 InternVL3 骨干的 0.8B VLA 上，用「人物掩码 + 边界框」两路辅助监督强化目标表征，再用以目标区域为条件的 Q-identity 查询压缩把历史帧由 256 token 压到 16 token；EVT-Bench 上 STT/DT/AT 成功率 90.2/72.5/70.9，真机部分为定性展示。
- **核对范围：** 通读 arXiv v1 全文 §I–§VI（摘要、引言、相关工作、方法 §III、数据集 §IV、实验 §V、结论）、表 I–V、图 1–7 的图注与可读的图内文字、参考文献列表；论文无附录 / 补充材料（12 页 PDF 未见）。图中的曲线 / 照片只看了图注与 pdftotext 抽出的文字，未逐像素核对。未读被引的 TrackVLA、TrackVLA++、ReferTrack、OmTrackVLA 等原文，表 I 中他方数字仅按本文转述；README 只读了主 README，未读 `deploy/dobottrack/README.md`、配置 YAML 与源码，也未运行任何代码；项目页未能访问。
- **通读：** 已完成

## 开源状态（步骤 2.5）

- **仓库 README（静态读取，2026-11-10）：** README 新闻栏写「2026-10-30 训练与 Habitat 评测代码已可用」；含安装（Habitat-Sim 0.3.1 + Bullet、Python 3.10、SAM2 固定 commit）、仿真数据采集脚本（`trackvla_bench/` 下的 `run.py`、`make_track_data.py`）、评测脚本 `scripts/run_trackvla_habitat_eval.sh`、训练配置 `configs/dobottrack_internvl3_1b_q_identity_box_mask.yaml` 与 `scripts/train.py`、Jetson Orin 部署指南入口。
- **未发布项（README 自述）：** 真实世界数据集徽章为「Coming soon」、Hugging Face 权重为「Coming soon」；README 明说 DobotTrack 表内结果「尚未绑定到本仓库的公开 checkpoint 与冻结的评测配置」。
- **依赖说明：** README 安装段让用户另从 gitee 拉取 `inffni-robotics/embodvla` 的某分支并 `pip install -e trackvla_bench/habitat-lab`；仿真资产（场景、人形、机器人、episode）需外部准备，不随仓库提供。
- **结论：** **部分开源**（训练 / 仿真评测代码有，权重与真实数据集未放出）。未运行、未审计代码。

## 核心摘录（均出自原文）

- **任务设定（§III-A）：** 语言指定的具身行人跟踪；输入为外观描述指令 ℓ 与 H=8 帧前向 RGB（5 Hz，Δ=0.2 s，448×448）；输出 P=8 个未来自身坐标系航点 (x,y,θ)，10 Hz，预测视野 0.8 s。
- **骨干与结构（§III-A、§V-A）：** 构建在 InternVL3 上，视觉编码器冻结，多模态投影器 / 语言模型 / Q-identity / 三个头联合训练；每帧 N=256 个视觉 token；掩码 token 与框 token 排在 8 个航点 token 之前，因果注意力让航点能看到二者。一次前向同时输出目标掩码、框、可见性与航点。论文给出模型规模为 0.8B，未在正文说明具体是 InternVL3 的哪个尺寸（README 配置文件名含 `internvl3_1b`，属推测对应关系）。
- **辅助任务（§III-B、§V-C）：** 系统比较五类：人物分割、框定位、视觉空间分类（VSC，左/中/右 + 近/中/远 或大小）、未来 DINO 特征预测、未来帧预测。终版只保留分割 + 框 + 轨迹。掩码头 = 可训练投影 + 预训练 SAM2 掩码解码器，输入当前帧稠密特征。
- **Q-identity（§III-C）：** 每个历史帧独立压缩为 Nq=16 个 token；做法是 16 个可学习查询与「目标区域 token」拼接做目标条件自注意力，再对全图特征做交叉注意力，共 3 个块，dc=512，8 头；训练时目标区域来自被扰动的真值框（概率 0.8 扰动），推理时用此前决策缓存的框与可见性，可见性 <0.5 或无缓存则用「无目标」token。8 帧窗口视觉 token 从 8×256=2048 降到 7×16+256=368。
- **损失（§III-D）：** L = L_traj + 0.001·L_seg + 0.01·L_box；轨迹为加权 MSE，(x,y,θ) 权重 (1,3,2)；分割 = BCE + Dice + IoU 分数回归 + 目标存在性；框 = 5·SmoothL1 + 2·GIoU + 1·可见性 BCE。训练约 28,000 步，8 张 NVIDIA B300，全局 batch 256，AdamW、BF16，语言模型与 SAM2 解码器峰值学习率 2e-5，其余 4e-5（§V-A）。
- **仿真数据（§IV-A）：** 取自 EVT-Bench（Habitat 3.0，HM3D / Matterport3D），约 180 万序列样本，STT:DT:AT = 1:2:1；掩码来自模拟器全景观测；框为掩码最紧外接矩形；VSC 标签用水平中心阈值 1/3、2/3 与归一化面积阈值 0.05、0.20，距离标签阈值 1.5 m、3.0 m。
- **真实数据（§IV-B）：** 四足机器人在室内办公室、室外开阔区与公共步行区采集，约 10.2 小时、126 个跟踪 episode、596,669 帧原始 RGB（960×540）、对齐后 565,994 个训练样本、596,323 个稠密目标掩码、33 个目标外观档案（图 3 注：24 女 9 男）；室内 5.68 h、室外 4.53 h，约 1 小时含相似干扰人；跟随动作由人类遥操作与一个专家模型产生；标注为 Qwen3.5 先定位出框，再交给 SAM3 出掩码，之后人工复核修正。真实数据只有水平位置与表观尺度两类 VSC 标签（真实距离不可靠）。
- **评测协议（§V-A）：** EVT-Bench 的 STT / DT / AT，指标 SR（回合结束时面向目标且距离 1–3 m）、TR、CR（与目标人碰撞而终止的回合比例）。
- **主结果（§V-B，表 I）：** 0.8B DobotTrack：STT 90.2/92.3/3.6、DT 72.5/83.3/8.3、AT 70.9/85.5/6.6（SR/TR/CR）；平均 SR 77.9（摘要）。相对 1B OmTrackVLA（81.4/41.5/60.0）SR 分别 +8.8、+31.0、+10.9，三项 CR 均更低；相对 7B 单视角 TrackVLA++（86.0/66.5/51.2）SR +4.2、+6.0、+19.7，但 STT / DT 的 CR 更高（3.6 对 2.1，8.3 对 4.7）；相对 ReferTrack（89.4/73.3/74.1）仅 STT SR 更高，DT / AT SR 更低，AT CR 更低（6.6 对 7.7）；LightNav-0（RVQ 动作 token + RL 后训练）STT 91.7、DT 82.6，高于本文 1.5 与 10.1 个点，AT 未报告。多相机方法仅作参考，作者称对比「提供背景而非受控评估」。
- **辅助任务消融（表 II，DT，0.8B）：** 仅轨迹 SR/TR/CR 55.9/83.6/16.5；+框 59.6/84.6/14.7；+VSC 59.6/83.8/13.1；+掩码 61.3/81.5/12.0；+框&掩码 63.7/84.5/13.2；+未来帧预测 60.9/80.9/13.9；+DINO 未来预测 58.9/80.6/15.9。作者结论：显式目标级监督优于自监督未来特征预测。表题未说明这些变体用何种历史压缩（掩码行数字与表 IV 的 16-token 池化掩码行相同，推测为 16-token 池化）。
- **目标条件消融（表 III，仿真数据训练）：** 掩码监督、Q-identity 无目标条件：STT/DT/AT SR 79.2/55.3/57.7；加目标条件：84.8/67.6/65.6（DT +12.3、AT +7.9）；再加框监督：87.6/69.0/67.8，但 DT / AT 的 CR（13.1 / 11.5）高于仅掩码的有条件变体（10.3 / 7.6）。
- **压缩与预算（表 IV，掩码监督）：** 256-token 池化 平均 SR 69.3、Orin Nano P95 延迟 203.9 ms；16-token 池化 67.6、91.0 ms；16-token Q-identity 72.7、93.4 ms。同为 16 token，Q-identity 相对池化 DT +6.3、AT +8.6，STT 相近（84.8 对 84.6）；相对 256-token 池化延迟约 2.18 倍快，比 16-token 池化多 2.4 ms。
- **真实数据的作用（表 V）：** 仅仿真 87.6/69.0/67.8；仿真 + 真实 90.2/72.5/70.9（均为 EVT-Bench 仿真评测）。
- **真机（§V-D、图 6–7）：** 四足机器人（前视相机、LiDAR、板载 Orin Nano）上室外步行街场景的叠加可视化与两段与轨迹基线的定性对比；无真机成功率、碰撞率或回合数等量化结果。
- **作者自述局限（§VI）：** 依赖缓存的定位预测质量与有限观测窗口；未来工作要研究错误目标关联与长时遮挡下的鲁棒性，并在更广的真实场景与目标身份上量化闭环表现。

## 小站说法逐条核对（具身智能小站 2026-11-10 条目）

- 「结合掩码与框监督」：属实，见 §III-B、表 II。
- 「按目标身份压缩视觉历史」：论文的做法是以**上一时刻缓存的目标框**所圈出的区域为条件压缩，不是对身份做显式 ReID；「身份」是作者的命名（Q-identity）。
- 「遮挡和干扰下更容易被持续锁定」：量化证据是 EVT-Bench 的 DT / AT 切分 SR 提高（表 II–IV）与图 5、图 7 的定性对比；真机无量化。
- 「主要量化结果来自 EVT-Bench，真机部分为定性展示」：属实（§V-D）。
- 小站未写：参数量仅 0.8B；真实数据集（126 episode）是论文的第三项贡献；训练纯监督、无强化学习（表 I 的 RL 列为 ✗）。

## 对 wiki 的映射

- [paper-dobottrack](../../wiki/entities/paper-dobottrack.md)
