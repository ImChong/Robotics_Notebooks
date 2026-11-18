# MoViC: Real-Time Humanoid Whole-Body Control from Monocular Video Streams（arXiv:2611.19998）

> 原始论文归档（ingest）；本页是来源摘录，结构化知识见 wiki 实体页。

- **标题：** MoViC: Real-Time Humanoid Whole-Body Control from Monocular Video Streams
- **中文标题：** MoViC：基于单目视频流的实时人形机器人全身控制
- **类型：** paper / humanoid whole-body tracking / monocular video / vision-based teleoperation
- **论文：** <https://arxiv.org/abs/2611.19998>
- **HTML 全文：** <https://arxiv.org/html/2611.19998>
- **论文版本：** arXiv:2611.19998v1（2026-11-15，预印本；16 页、10 图）
- **作者：** Yimin Han、Jiahui Zhang、Wanyue Li、Zeren Luo、Yeke Chen、Liu Zhao、Peizhuo Li、Xiaoyu Ji、Shihao Dong、Xuechao Chen、Zhangguo Yu、Annan Tang、Peng Lu
- **机构：** 香港大学（The University of Hong Kong）；新加坡国立大学（National University of Singapore）；北京理工大学（Beijing Institute of Technology）；EngineAI（深圳）
- **项目主页：** <https://movicproject.github.io/>
- **代码状态：** 2026-11-18 查阅官方主页，显示「Code(coming soon)」，没有可访问的 MoViC 官方训练/部署代码链接；不可视为已开源
- **入库日期：** 2026-11-18
- **核对范围：** 通读 arXiv HTML v1 §I 引言、§II 相关工作、§III-A–D 方法（动作重定向、非对称 PPO、频域编码、SEG 外推、Transformer actor）、§IV-A–E（实现、仿真、消融、两组对比、PM01 真机演示）、§V 结论，并核对表 I–V、图 1–10 与官方主页的算法图/演示说明。
- **通读：** 已完成

## 原文要点

1. **问题与整体链路：** 单 RGB 相机实时估计人体动作，重定向到 EngineAI PM01 后执行人形全身控制；不依赖操作者佩戴 VR、动捕衣或传感器。视觉人体重建约 10 Hz，机器人参考与跟踪控制 50 Hz，采集、估计和控制异步运行。
2. **参考动作数据：** AMASS、LAFAN1、Motion-X++、OMOMO 和自采运动；先统一人体表示、重采样，再通过带全局/局部关键点、朝向及限位/平滑正则的可微运动学优化重定向到 PM01。
3. **两个不能混为一谈的频域模型：** 局部重建式 DFM 编码器对每个窗口更新频率、幅值、偏置和相位，冻结后给 actor 提供 320-D 的相位增强 θ 目标；部署期另用多步预测训练的周期潜动力学（论文称 FLD 外推）在视觉更新间补出控制参考。前者提供时序上下文，后者承担 10→50 Hz 的预测。
4. **跟踪策略：** 10 帧本体感觉历史（10×81=810-D）+ 下一控制步参考（52-D）+ θ（320-D）= 1182-D；4 层、6 注意力头的 Transformer actor 产生 24 个关节位置目标，随后经 PD 控制。训练用 PPO 和仅训练时可见特权状态/未来参考的 critic。
5. **SEG 在线重建：** TensorRT YOLO11-pose 检测人框；仅恰有一个置信度 >0.75 的人物时继续重建；SAM 3D Body 将 512×512 crop 经 MHR 映射至 SMPL，再用 GMR 机器人重定向。手部解码未进入流式输出。
6. **参考外推的测量锚定：** 以最新接受的参考 x(t) 加上「预测未来解码值 − 当前解码值」的差作为预测参考；在满足一致性门限时接受视觉更新，否则沿最近可信状态外推。门限无法保证过滤所有错误视觉估计。

## 实验数据（须按条件解读）

| 实验 | 对照 | MoViC / SEG | 条件与含义 |
|---|---|---|---|
| 跟踪观测消融：1 个未来参考 | 成功率 90%，MPJPE 0.47 rad，MPJVE 4.80 rad/s | — | 无 θ；论文表 IV |
| 跟踪观测消融：10 个未来参考 | 成功率 95%，MPJPE 0.28 rad，MPJVE 4.55 rad/s | — | 无 θ；1330-D 输入 |
| θ + 单个未来参考 | — | **成功率 99%，MPJPE 0.25 rad，MPJVE 4.30 rad/s** | 1182-D；同属论文仿真消融，不能称真机成功率 |
| 关节预测误差：FLD 外推 vs 线性 | 位置 RMSE 0.7561 rad、速度 RMSE 8.6816 rad/s | **0.0655 rad、1.2372 rad/s** | 论文表 III |
| EMDB 视频重建 | GVHMR+GMR：MPJPE 123.45 mm | **SEG：117.54 mm** | 论文表 V，EMDB-1 有效帧 |
| 高动态舞蹈跟踪 | SONIC、OmniH2O、TWIST 基线 | 论文图 8 三项误差均最低 | MuJoCo；基线适配同款 PM01、相同参考动作数据集，未在正文给三项数表 |
| PM01 真机 | — | 本草纲目舞蹈、双臂搬箱、单臂提包（提包连续演示 139 s） | 定性真机展示；没有独立任务总体成功率 |

**量化边界：** 表 IV 的 99% 是模拟参考轨迹跟踪成功率，不是「真实世界抓取成功率」；视觉重建对严重遮挡和单目深度歧义仍敏感；策略不直接观测物体或接触力，精细物体对齐依靠操作员视觉反馈与本体闭环。

## 工程复现的约束

- **真机：** EngineAI PM01，24-DoF；Logitech BRIO 1920×1080@30 fps，固定且水平安装；视觉估计约 10 Hz，控制 50 Hz。
- **计算资源（论文 §IV-A）：** 部署 PC 单张 RTX 5090；训练使用 4 张 A100。
- **算法训练：** PPO（γ=0.99，GAE λ=0.95，clip ε=0.1）；频域编码窗长 H=31、30 Hz，64 latent channels，θ=5×64=320 维。
- **可复现性：** 论文公式与工程配置已公开，但官方主页代码入口仍写 coming soon，不能据论文推导编造可运行源码入口或时序图。
- **局限：** 单人检测条件、遮挡、单目物体深度/接触感知缺失；所示 PM01 系统不等于已支持其他机型的直接即插即用。

## 对 wiki 的映射

- [MoViC 独立论文实体](../../wiki/entities/paper-movic-monocular-video-whole-body-control.md)
- [MoViC 官方项目页归档](../sites/movicproject-github-io.md)
- [人形遥操作知识页](../../wiki/tasks/teleoperation.md)
- [人形遥操作纵深路线](../../roadmap/depth-teleoperation.md)

## 参考来源

- 论文 HTML v1：<https://arxiv.org/html/2611.19998>
- arXiv 摘要与版本：<https://arxiv.org/abs/2611.19998>
- 官方项目与演示：<https://movicproject.github.io/>
