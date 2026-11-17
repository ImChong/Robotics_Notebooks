# PixVid / Back to Pixels（arXiv:2611.17336）原文归档

- **标题：** Back to Pixels: Lifting Latent Video Diffusion into Pixel Space（回到像素：将潜空间视频扩散迁移至像素空间）
- **类型：** paper
- **作者：** Yanke Song, Shuo Chen, Chris Rockwell, Guande He, Nan Liu, Desai Xie, Tianpei Gu, Yong-Hyun Park, Aryan V S, Dennis Yuren Menn, Ming-Hsuan Yang, Xun Huang
- **机构：** Roblox、加州大学默塞德分校（UC Merced）、宾夕法尼亚大学（University of Pennsylvania）、德克萨斯大学奥斯汀分校（UT Austin）
- **发表：** arXiv:2611.17336v1，2026-11-12；预印本
- **论文：** https://arxiv.org/abs/2611.17336
- **HTML：** https://arxiv.org/html/2611.17336v1
- **PDF：** https://arxiv.org/pdf/2611.17336
- **项目页：** https://chenshuo20.github.io/PixVid/
- **官方代码：** 截至 2026-11-17 尚无可访问的正式发布仓库链接；以项目页后续更新为准
- **入库日期：** 2026-11-17
- **核对范围：** arXiv v1 HTML 正文（引言、相关工作、`3 方法、`4 实验、`5 结论）及附录 A–C（Patch 初始化、Detailer、数据/消融/实时推理成本、定性示例），并对照官方项目页的方法、低延迟展示、结果及资源区；未运行模型或独立复现实验。
- **通读：** 已完成

## 核心摘录

1. **VAE 的结构限制：** 主流时空 VAE 将多帧压成一个 latent，造成动作/深度条件无法逐帧对齐。分辨率加倍会使固定 patch latent token 数约 4 倍增长（注意力项约 16 倍）。
2. **迁移初始化（`3.2 / 附录 A.1）：** 从 Wan2.1 的预训练 DiT 迁移；用 500 个训练片段构造像素 patch 与 latent 配对，离线 Orthogonal Procrustes 拟合线性映射 A，再复合原 patch embedding 初始化像素输入层；以比例 c 做 timestep/SNR 校准。14B 模型 c=2.85，1.3B 模型 c=3.78。
3. **逐帧输出：** 每帧 patch 覆盖 `1×p×p` 像素，保持 DiT 主干，删除原 VAE 解码链路，换成空间上采样、时间因果的 3D 卷积 detailer；其每层读取当前及前两帧特征并保留缓存。
4. **优化与采样：** 预测干净视频 x（x-prediction）而不是直接预测 velocity；训练用 flow matching velocity-space loss，加时间帧差 Charbonnier 与 LPIPS；增大噪声 shift，像素空间用 Euler 采样，缓解颜色漂移。
5. **高分辨率（`3.3、`4.3）：** patch 从 16 调到 32，480p 832×480 与 960p 1664×960 都保持每帧 30×52 tokens；相应增加一级 detailer up block。480p VBench 总分 84.78，960p 85.09；Wan2.1-14B 原始 latent 基线为 84.05。VBench 评分会缩放到至多 512 像素，不能单用该分数证明高分辨率细节更好。
6. **对齐编辑（`3.4、`4.4）：** 对 1,000 段测试视频按光流分桶，快运动片段中 PixVid 的运动保持率 98%，同配方微调 latent 基线为 83%；快运动流误差 10.7 vs 17.9 像素。FVD 并非所有指标领先。
7. **交互生成（`3.5、`4.5）：** 将 SolarWM-5B 双向基座迁移后以 Self Forcing 蒸馏成 4 步因果学生，每次生成一个帧块；相对 12 帧块 SolarWM，控制到屏幕中位延迟 34 ms vs 495 ms（同一 B200 定制内核）；代价是吞吐 29.4 vs 72.0 FPS，Rot/Tr 相机跟随误差更大。此处动作指相机控制，不等同低层机器人控制。
8. **工程代价与局限：** 14B 视频骨干微调约 64K 合成视频 / 1.3M 训练窗口，约 4K B200 GPU 小时。逐帧展开会增加常规时长推理成本、长序列相机轨迹误差累积。项目页暂未提供可公开获取的代码/权重入口，本次不声称已复现。

## 对 wiki 的映射

- [PixVid：逐帧像素视频扩散论文实体](../../wiki/entities/paper-pixvid-back-to-pixels.md)
