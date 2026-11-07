# Digging Into Self-Supervised Monocular Depth Estimation（ICCV 2019）原始论文通读记录

- **论文：** https://arxiv.org/html/1806.01260
- **论文编号：** arXiv 1806.01260
- **源码：** https://github.com/nianticlabs/monodepth2
- **通读：** 已完成
- **核对范围：** 读完正文、§3–5、KITTI 消融、§A–I 附录和局限，核对官方源码入口，未运行训练。

## 方法

DepthNet 用 ResNet18 encoder-decoder 预测目标帧逆深度；PoseNet 由两帧 RGB 预测 SE(3) 相对位姿。相机 K 与预测深度构建目标到来源帧的 warp，使用 L1+SSIM 光度误差。min reprojection 对每像素选最适合的来源帧，缓解遮挡；auto-masking 拒绝相机不运动或违反运动假设的像素；full-res multi-scale 让不同尺度视差在原图分辨率比较重投影误差。边缘平滑避免无纹理区域无约束发散。

## 实验和评估口径

arXiv v4 Table2 KITTI Eigen，640×192、仅单目视频 M：AbsRel 0.115、δ1 0.877；baseline（无三项改进）AbsRel 0.140、δ1 0.852。训练 20 epochs / ResNet18。单目评测按图 median scaling，非直接公制深度。反射、颜色饱和和边界模糊有论文 Fig8 明确失败示例。

## 源码与局限

train.py → Trainer(opts).train() → trainer.Trainer.run_epoch/process_batch → networks.DepthDecoder/pose → generate_images_pred → compute_losses → Adam step；evaluate_depth.py 和 test_simple.py 是评估、推理单独入口。

方法依赖图像亮度一致性和近似刚体场景假设；动态/反射或无纹理区域可能失效。单目视频训练的尺度无法凭网络自行获得米制真值。作者仓库训练环境为旧版 PyTorch，未在本次复现。

- [深读节点](../../wiki/entities/paper-monodepth2.md)
