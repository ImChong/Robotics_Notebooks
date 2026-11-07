# Monodepth2 — 一手资料归档

> 来源层（官方代码/仓库）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** Monodepth2
- **类型：** repo
- **链接：** https://github.com/nianticlabs/monodepth2
- **入库日期：** 2026-11-08
- **一句话说明：** 自监督单目深度估计实现。
- **核对范围：** 官方 README 全文与真实训练、模型与评测入口已核对：train.py → Trainer(opts).train() → trainer.Trainer.run_epoch/process_batch → networks.DepthDecoder/pose → generate_images_pred → compute_losses → Adam step；evaluate_depth.py 和 test_simple.py 是评估、推理单独入口。；读数来自原论文而非复现。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [Monodepth2 源码](https://github.com/nianticlabs/monodepth2) **（本归档主类型）**

## README 与代码核对

train.py → Trainer(opts).train() → trainer.Trainer.run_epoch/process_batch → networks.DepthDecoder/pose → generate_images_pred → compute_losses → Adam step；evaluate_depth.py 和 test_simple.py 是评估、推理单独入口。

**方法**：DepthNet 用 ResNet18 encoder-decoder 预测目标帧逆深度；PoseNet 由两帧 RGB 预测 SE(3) 相对位姿。相机 K 与预测深度构建目标到来源帧的 warp，使用 L1+SSIM 光度误差。min reprojection 对每像素选最适合的来源帧，缓解遮挡；auto-masking 拒绝相机不运动或违反运动假设的像素；full-res multi-scale 让不同尺度视差在原图分辨率比较重投影误差。边缘平滑避免无纹理区域无约束发散。

**限制**：方法依赖图像亮度一致性和近似刚体场景假设；动态/反射或无纹理区域可能失效。单目视频训练的尺度无法凭网络自行获得米制真值。作者仓库训练环境为旧版 PyTorch，未在本次复现。

## 方法与边界

自监督单目深度估计实现。已核对官方 README 与主要执行模块，并依据原文核对研究方法，未在本次运行指标复现。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 已通读对应原论文并建立独立归档。
- [x] 已核对源码入口和模型路径，时序图见 wiki。
- [x] 已建立 [Monodepth2 深读节点](../../wiki/entities/paper-monodepth2.md)，健康度待 CI 复核。
