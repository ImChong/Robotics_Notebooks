# DeepSoMo（arXiv:2611.11242）

> 来源归档（paper）

- **标题：** DeepSoMo: Deformable Object Motion Estimation under Proprioceptive Partial Observability via Predictive Latent Dynamics
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.11242>
- **PDF：** <https://arxiv.org/pdf/2611.11242v1>
- **项目页：** <https://ai4ce.github.io/DeepSoMo>
- **作者 / 机构：** Hanwen Zhao, Weicheng Dai, Luoyao Chen, Chen Feng（通讯）；均属纽约大学（NYU，Brooklyn）；NSF Grant 2024882 资助（首页脚注）
- **入库日期：** 2026-11-11
- **一句话说明：** 用内置双相机的内部形变图像做输入，ResNet-18 + LSTM 估计软体球的体坐标系位移增量、旋转增量、表面形状与接触点；训练时加一个「潜状态对齐下一帧 / 下两帧编码特征」的自监督潜预测约束；全部结果来自 PolyFEM 有限元仿真。
- **核对范围：** 读了 arXiv v1 PDF 全部 8 页正文（摘要、I 引言、II 相关工作、III 方法含式 (1)–(5)、IV 实验含表 I–V、V 结论与局限），图 2 与图 4 对照图像 / 图注阅读，图 1/3/5 读图注；参考文献只读到第 23 条左右。没有读到：补充材料（正文称动画与更多可视化在 supplementary material，PDF 不含）、项目页视频（项目页视频栏为空）、代码与数据（均未公开）。
- **通读：** 已完成

## 开源状态

- 论文未给代码链接。项目页（2026-11-11 访问）有指向 GitHub `ai4ce/DeepSoMo` 的 Code 按钮，提示 "Code (public soon)"，数据集区写 "We are preparing the DeepSoMo dataset for public release"：**均未发布**。

## 核心摘录

1. **问题（III-A）：** 状态 s_t = {平移增量 x_t, 旋转增量 r_t, 表面形状 p_t, 接触 c_t}，均在附着于物体的体坐标系下；全局轨迹靠累积组合增量得到。观测只有内部图像 I_t，f_t = φ_enc(I_t)，l_t = φ_dyn(f_0..f_t)，s_t = φ_dec(l_t)（式 1–3）。
2. **方法（III-C/D）：** ResNet-18 共享编码器（每图 512 维，两相机拼接）→ LSTM 得 l_t；l_t 对齐 f_{t+1}，级联的 h_t 对齐 f_{t+2}；L_H = Σ MSE(f_{t+1}, l_t) + MSE(f_{t+2}, h_t)（式 4）；总损失 = MSE(x) + MSE(r) + Chamfer(p) + BCE(c) + L_H（式 5）；旋转用 6D 表示，形状解码器改自 FoldingNet。
3. **数据（III-B）：** PolyFEM 仿真，直径 0.5 m 的软球在 20×20 m 带高斯噪声的地形网格上滚动；100 个地形 × 10 种配置 = 1000 条仿真，每条 100 步、0.1 s/步；200 条留作测试（来自留出地形），其余 640 训练 / 160 验证。两个 256×256、170° 视场的灰度相机背对背装在体内，Blender 渲染。
4. **主结果（表 II，3 个种子）：** NTE / ATE(cm) / ARE(deg) / CPC(%) / CD(1e-1 mm)：ResLSTM 8.3 / 56.8 / 19.3 / 99.0 / 3.1；ObsPred 7.7 / 49.9 / 19.0 / 99.1 / 2.9；DeepSoMo 6.6 / 38.1 / 16.8 / 99.2 / 2.5。文内称 ATE 相对 ResLSTM 降 32.9%、相对 ObsPred 降 23.6%（按表数字复算吻合）。ObsPred 约 156M 可训练参数，DeepSoMo 约 18M。
5. **消融（表 III / IV）：** ResNet 优于 VGG、ViT；LSTM 优于 RNN、DCC、Transformer（Transformer ATE 3592.4 cm）；预测视野 2、3 步仅有边际变化；换预测目标（l_{t+1}、f_{t+2}）优于无预测但不及默认设计。
6. **物体变化（表 V）：** 形状变化（x 向缩放 1.6）与形状 + 材料变化（缩放 0.8，杨氏模量 1e4 → 3e4）均从规范模型 checkpoint 微调，各 320 训练 / 80 验证 / 100 测试；ATE 37.1 → 31.5、57.1 → 35.6 cm（ResLSTM → DeepSoMo）；形状变化下 NTE 误差棒有重叠（7.5±0.3 vs 6.8±0.8）。
7. **训练设置（IV-C）：** 单张 A100，200 epoch，batch 5，Adam lr 1e-3 + ReduceLROnPlateau，约 5 小时 / 模型，3 个随机种子。
8. **论文自述局限（V）：** 看不到无形变的运动（自由落体、无摩擦滑动）；只做仿真单物体 + 静态地形，无真机验证、无多物体 / 软–软碰撞；真机需解决软体制造、嵌入传感、标定、sim-to-real；长时程仍会累积误差；未来考虑 IMU、触觉 / 压力 / 应变传感与执行器反馈。

## 对 wiki 的映射

- [paper-deepsomo](../../wiki/entities/paper-deepsomo.md)
