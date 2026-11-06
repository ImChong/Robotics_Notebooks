# Commodity-Price Closed-Loop Evaluation（arXiv:2611.07441）

> 来源归档（paper）

- **标题：** Closed-Loop Physics-Based Manipulation Evaluation at a Commodity Price: What 4.4 L4 GPU hours at $3.50 Buys You
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.07441>
- **PDF：** <https://arxiv.org/pdf/2611.07441>
- **代码：** <https://github.com/rohungry/evaluation_harness_wrapper>
- **入库日期：** 2026-11-06
- **线索来源：** [只在虚拟房间练习，人形机器人走进陌生房间能做什么？](../blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)（具身智能小站，2026-11-06）
- **一句话说明：** 在仿真搬盒任务上，8 张 L4 并行跑完 200 次评测，耗时约 38 分钟，按论文单价花费 3.49 美元。

- **作者 / 版本：** Rohun Kshirsagar、Diana Rypkema（equal contribution）；arXiv v1，2026-11-05，IROS 2026 Scale-Infra Workshop。
- **核对范围：** 已下载并通读 arXiv PDF 全文（摘要、方法、报告协议、策略描述、表 I/II、负面结果、结论与局限、代码与数据声明、参考文献）；未运行代码、未逐文件阅读仓库。
- **通读：** 已完成

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-06）：rohungry/evaluation_harness_wrapper 经 git ls-remote 可达；无项目页。论文「代码与数据」节称仓库含 Modal 镜像定义、训练脚本与配置、已训练检查点和每个报告运行的逐 episode 结果；基准软件与数据归 TRI。

## 核心摘录（已对照原文）

1. **架构：** 基于 Barreiros et al.（Science Robotics 2026，lbm_eval）；Drake 双 Franka FR3 桌面仿真，任务 pick_and_place_box（右臂抓随机摆放的盒子放入桌中圆形区域，限时 15 s），该任务不在原论文评测套件内。Modal 无服务器扇出：8 个容器、每个 1 张 L4，按互不相交的 scenario_index 切片分片（0-25、25-50…175-200）；随机种子由索引派生，分片并集与不分片运行完全一致；分片内串行；策略由独立 gRPC 服务进程提供；Xvfb + EGL 无头渲染。
2. **两个未文档化的构建步骤：** GPU 运行时只挂载驱动、不含用户态 GL 库，需写入 Drake 渲染器要求的 EGL vendor ICD 文件；Drake 资产下载脚本假设的解释器路径与基础镜像不符，需加符号链接。栈为 Debian-slim、Python 3.12、四个公开 lbm_eval wheel、PyTorch。
3. **报告协议：** R1 完成时间小于 1 s 的成功视为基准伪影，剔除 5 个场景（索引 6、94、116、140、146，约 0.3 s 即成功，盒子初始就在目标上方），分母为 195；R2 每个成功都人工看录像；R3 按 N/195 报告并给 95% Wilson 区间。不剔除时无动作策略也会报 2.5%。
4. **策略：** 共享编码器 ImageNet 预训练 ResNet-18（单相机视角）+ 18 维末端位姿 MLP，输出 20 维动作（双臂末端位姿含 6D 旋转 + 两个夹爪宽度）；训练数据为 TRI 公开的 99 条可用 episode、4,203 个关键帧；差别仅在解码器与 (H, K)（H 为每次预测动作数，K 为重规划前执行数）：v1 为 BC（MSE）H=1，v2a 为单步 DDPM，v2b 为分块 DDPM (16,8)，v2b-bc 为分块 BC H=8。
5. **成本（表 I）：** 单次 200 rollout 复现 37.7 min 墙钟（8 分片），4.36 L4-GPU-小时，按 $0.80/L4-小时为 $3.49；平均 rollout 约 78 s；训练 26 min，训练 + 一次复现约 $3.85；表 II 全部工作约 $34；整个研究含环境搭建约 $34。
6. **负面结果（表 II，195 场景）：** v1 原版 5.1%（[2.8, 9.2]）、seed1 11.8%（[8.0, 17.1]）、seed2 2.6%（[1.1, 5.9]）；v2a DDPM 0%（[0, 1.9]）；v2b 分块 DDPM (16,8) 0%，且欠拟合；v2b-bc (8,4) 及两个种子、(8,1) 逐步重规划均为 0%。三个独立训练的分块检查点共 780 rollout 无真实成功，三个单步检查点均大于 0。验证 MSE 不能预测成功率（0.0148 与 0.0152 对应 11.8% 与 2.6%）。人工检查：基线因漂移累积失败，分块检查点生成的轨迹不跟随场景，即使每步重规划也如此。
7. **局限（作者自述）：** 仅一个任务、一个数据规模、一个模型规模；分块结论不应外推；数据充足时动作分块有效（Diffusion Policy、ACT），约 100 条示范且成功率仅几个百分点时可能有害；未来需扩展到更多单/多任务基准，并用 Castro 等的柔顺接触仿真生成合成数据。
8. 论文披露使用 LLM 编程助手起草代码与稿件。

## 对 wiki 的映射

- [paper-commodity-price-closed-loop-evaluation](../../wiki/entities/paper-commodity-price-closed-loop-evaluation.md)
- [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-06-20-papers-technology-map.md)
