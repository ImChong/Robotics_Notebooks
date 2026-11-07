# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis — 原文通读

- **原作者：** Ben Mildenhall et al.，UC Berkeley / Google Research / UC San Diego
- **原论文：** https://arxiv.org/html/2003.08934 （ECCV 2020，arXiv v2）
- **官方源码：** https://github.com/bmild/nerf
- **通读：** 已完成
- **核对范围：** 已全文阅读 §1–7、Eq 1–7、Fig 1–4、Table 1–2 与附录 A–D 的网络、采样、坐标映射、结果、消融、失败边界和参考资料；核对作者官方 README、run_nerf.py、run_nerf_helpers.py 的主要执行入口，未重现训练。

## 方法与模型结构

- 5D input = 3D position + 2D viewing direction；位置位置编码经 8 层 × 256 MLP 预测 density sigma 和 256维 feature；另加视角编码后通过128 hidden units 生成 RGB view-dependent radiance。
- 体积渲染对采样位置的 sigma 和 RGB 执行 alpha-compositing，整段 camera ray differentiable；已知 camera pose/内参将真实图像颜色和渲染 RGB 对齐计算 MSE。
- positional encoding 恢复高频细节，hierarchical importance sampling 先 64 coarse samples，fine 再 128，合成场景必要时 64+128；原文附录测试渲染合计最高每 ray ~256 queries。
- 实景 forward-facing 的 normalized device coordinates 重新投影到近/远平面体积；需要准确 SfM pose 和 intrinsics。

## 论文实验

论文 Table1：diffuse synthetic 360 PSNR **40.15dB**；realistic synthetic 360 PSNR **31.01dB**、SSIM **0.947**、LPIPS **0.081**；real forward-facing PSNR **26.50dB**、SSIM **0.811**、LPIPS **0.250**，LLFF real LPIPS **0.212** 好于本方法。**这三种数据条件不同**。

Table2 realistic synthetic 中完整模型 31.01dB，去 positional encoding 降到 **28.77dB**，去 view dependence **27.66dB**，去 hierarchical importance sampling **30.06dB**。原文 §6/附录指出单场景优化 12h+、V100 约30秒/帧（在作者模型与图像规格下），不是实时神经 SLAM。

## 局限

NeRF 输入是**已经知道位姿的静态场景照片**，不直接估计在线相机运动，也不天然支持动态场景。亮度/反射变化和错误 pose 会伤害重建；单场景优化很慢，后续 Instant-NGP/3DGS 提速不能归功于这份原始 TF NeRF 实现。

- [独立深读节点](../../wiki/entities/paper-nerf-mildenhall.md)
