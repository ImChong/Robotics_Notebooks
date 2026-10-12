# AutoBio+ 论文来源归档

## 元数据

- **题目：** AutoBio+: Benchmarking Dexterous Manipulation with Dynamic Experimental Phenomena in Realistic Laboratories
- **作者：** Xuanye Wu, Yue Shi, Yongcan Yang, Yiqun Wang, Zhaoqi Wan, Dayu Xia, Jiezhang Cao, Xiangfei Li, Yao Mu, Guangtao Zhai
- **单位（arXiv HTML 页首）：** Shanghai AI Laboratory；Huazhong University of Science and Technology；Shanghai Jiao Tong University；University of Southern California（页面未标明各作者对应关系）
- **预印本：** https://arxiv.org/abs/2610.13643 （cs.RO，v1，2026-10-09）
- **HTML：** https://arxiv.org/html/2610.13643v1
- **项目页：** https://autobio-plus.github.io/AutoBioPlus
- **代码：** 截至 2026-10-12 未发布（项目页写 “Code coming soon”，无仓库链接；见 [sources/sites/autobio_plus_project_page.md](../sites/autobio_plus_project_page.md)）
- **入库日期：** 2026-10-12

## 论文要解决的问题

已有灵巧操作仿真基准多以静态刚体桌面为主，缺少真实生物实验室里“状态随时间变化”的现象（火焰、蒸汽、气泡），也缺少与真实实验室外观对齐的背景。AutoBio+ 提供 14 个任务、4 种机器人形态的仿真基准，并带并行示范生成与闭环评测框架。

## 方法与训练信息

1. **任务与本体：** 14 个任务分四类能力：功能性工具使用、精细操作、语言条件视觉定位、仪器操作。涉及 Shadow Hand、Wuji Hand、XHand 和双臂 Aloha（共 4 种形态）。任务例：Attach Pipette Tip（Shadow）、Manipulate Tweezers（Wuji）、Ignite Bunsen Burner（XHand）、Close Thermal Cycler Lid（XHand）、Tighten Tube Cap（Aloha）。各自由度数页面未给出。
2. **动态现象：** 用与物理引擎解耦的粒子系统实现。火焰为上升的立方体粒子加随机漂移、透明度指数衰减和颜色渐变（颜色随本生灯两个调节环位置变化）；蒸汽为膨胀球形粒子云；气泡为带浮力和旋转偏移的上升粒子。位置更新 x(t+Δt)=x(t)+[d(x,t)+η(t)]Δt，不透明度 α(t)=α0·exp(−(t−t0)/τ)。
3. **混合渲染：** 6 个真实实验室场景用 Lixel K2 空间扫描仪采集并重建为高斯泼溅背景；前景由 mjlab 渲染，按 I_final = M⊙I_origin + (1−M)⊙R({G_i}, T_cam) 合成（M 为前景掩码）。
4. **并行框架：** 基于 mjlab/MuJoCo Warp；Parallel Finite-State Machine（PFSM）让每个环境保持自己的阶段与路点索引，批内异步推进；回合先无渲染滚动，仅对成功轨迹批量重放渲染。纹理预先在 MJCF 中索引（Warp 不支持运行时换纹理）。
5. **渲染开销（339 帧、2 相机、640×480）：** MuJoCo Warp 13.95 s；AutoBio+ 含高斯背景 15.57 s（约 +11.6%）；Blender（前作 AutoBio）1092.02 s（约 70.14 倍）。
6. **评测策略训练：** 每个模型在 1 张 NVIDIA H200 上微调 30,000 步（π0.5 batch 32）；Operate Mixer Panel 用进度奖励，其余任务二值奖励。示范总条数原文页面未报告。

## 论文报告的结果

- **仿真（12/14 个任务，4 种策略：Diffusion Policy、π0.5、GR00T-N1.7、Fast-WAM；3 个随机种子 × 100 回合，均值±SEM）：**
  - Close Centrifuge Lid（Wuji）：DP 30.0，π0.5 81.7，GR00T-N1.7 100.0，Fast-WAM 43.3。
  - Close Thermal Cycler Lid（XHand）：DP 87.0，π0.5 5.3，GR00T-N1.7 49.7，Fast-WAM 99.7。
  - Ignite Bunsen Burner（XHand）：DP 60.7，π0.5 77.7，GR00T-N1.7 40.0，Fast-WAM 23.3。
  - Attach Pipette Tip（Shadow）：DP 5.0，π0.5 12.0，GR00T-N1.7 2.7，Fast-WAM 20.7。
  - Tighten Tube Cap（Aloha）：四种策略均 ≤0.7。Load Centrifuge、Operate Mixer Panel、Aspirate with Pipette 等多数接触丰富或语言定位任务均低于约 6%。
  - 没有任何策略在所有任务上占优，任务间差异很大。
- **真机 sim-to-real：** UR5e + XHand，腕部 RealSense D405、第三视角 D435；任务 Transfer Tube（XHand）；对比“10 真机 + 90 仿真示范”下仅背景不同的两种仿真数据（高斯背景 vs 标准 MuJoCo 背景），每条件 20 次试验（10 个试管位置各 2 次）；评分 抓取 3 / 提起 5 / 开始插入 8 / 完全插入 10 分。归一化均分：高斯背景 66%，MuJoCo 背景 17%。纯真机 100 条示范作为另一参照。
- 论文讨论：语言条件视觉定位是主要瓶颈；灵巧手预训练对容差大的简单任务有帮助，对接触丰富任务帮助有限。

## 复现与解释边界

- 代码与数据截至 2026-10-12 未发布；项目页为匿名提交页，Resources 栏为空。
- 液体行为和整体场景交互不一定高保真（作者在局限中提到，建议 4D 高斯或动态神经场）。
- 真机只验证了一个任务（Transfer Tube）和一种背景差异，不代表其余 13 个任务的迁移差距。
- 12/14 任务入评，另外 2 个任务的策略结果表中未列。
- 阅读边界：附录 PDF 未读取。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-autobio-plus-lab-dexterous-benchmark.md
- 汇总页：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md

## 一手来源

- arXiv：<https://arxiv.org/abs/2610.13643>
- HTML：<https://arxiv.org/html/2610.13643v1>
- 项目页：<https://autobio-plus.github.io/AutoBioPlus>
