# Connecting Touch and Vision via Cross-Modal Prediction（VisGel 原文与仓库核对笔记）

- **来源：** <https://arxiv.org/abs/1906.06322>（`arXiv:1906.06322v1`，CVPR 2019）
- **项目页：** <http://visgel.csail.mit.edu/>
- **代码仓库：** <https://github.com/YunzhuLi/VisGel>（仓库根目录下未含独立 LICENSE 文件，使用与二次分发需遵循作者声明）
- **作者与机构：** Yunzhu Li, Jun-Yan Zhu, Russ Tedrake, Antonio Torralba（MIT CSAIL）
- **核对日期：** 2026-11-14
- **核对范围：** 论文全文 `arXiv:1906.06322v1`（第 1–1013 行）及 `YunzhuLi/VisGel` 仓库 `README.md`、`train.py`、`eval.py`、`data.py`、`models.py`、`scripts/*.sh`
- **通读：** 已完成

---

## 开源与数据核查

- **开源状态：** **代码与数据集均已公开**。训练/推理/评估脚本通过 `YunzhuLi/VisGel` 发布（环境基线为 Ubuntu 16.04、Python 3.7、PyTorch 0.3.1），预训练权重（约 500 MB）、Demo（约 570 MB）、数据索引列表（约 430 MB）、`data_seen`（328 GB）与 `data_unseen`（83.2 GB）均通过 `scripts/download_*.sh` 下载；仓库根目录下无显式 `LICENSE` 文件。

---

## 原文核心事实摘录

1. **硬件与自动化采集：** **KUKA LBR iiwa** 机械臂末端搭载 **GelSight** 光学触觉传感器（约 `1.5 cm × 1.5 cm` 弹性体凝胶，含 R/G/B/White LED 与黑色标记点阵）+ 固定于机械臂后方三脚架上的网络摄像头。利用 **ElasticFusion** 稠密 RGB-D 3D 重建识别场景几何，自动采样垂直桌面方向的接触点候选并跳过空白桌面与已移动物体区域，实现时间同步的连续触碰采集。
2. **数据规模与划分：** 覆盖 **195 个日常物体**（其中 45 个来自 YCB，150 个选自食物/工具/厨房用品/织物/文具等日常物品），每次桌面随机摆放 4–10 个物体。共收集 **12,000 次触碰 × 250 帧/次 = 3,000,000 对视觉-触觉同步图像**（视觉分辨率约 272 缩放至 `256×256` 裁剪，GelSight 为 `128×128` 低分辨率输出目标；10 Hz 级或同步序列）。训练集 **10,000 次触碰（250 万对，165 个物体）**，测试集 **2,000 次触碰（50 万对）**，均分于 30 个已见物体与 30 个未见物体。
3. **跨模态条件生成方法：**
   - **双编码器 + Skip-U-Net 生成器 `_netG_resnet`：** 参考编码器 `resnet_ref` 接收背景参考对 `r = (x_ref, y_ref)`（6 通道 RGB 拼接，用于显式传递相机视角、物体尺度与静态背景空间对齐信息）；源模态编码器 `resnet_src` 接收含时序窗口的源序列 `x̄_t = {x_{t-4}, x_{t-2}, x_t, x_{t+2}, x_{t+4}}`（`w_timewindow=1` 时将 5 帧转灰度拼成 5 通道以提供时序上下文与接触瞬间线索）。两路 512 维特征拼接为 1024 维，再经 6 层转置卷积解码器逐层与 `resnet_ref` 的多尺度跳连接拼接输出 `Tanh` 目标图像。
   - **解决尺度与位置落差（Scale & Location Gap）：** 将未接触/桌面参考图像对 `(x_ref, y_ref)` 作为显式空间锚点输入生成器与判别器。
   - **解决模式坍塌（Mode Collapse）：** 全程约 60% 帧处于非接触悬空状态导致模型倾向输出平坦零触觉帧；在 `data.py`（`calc_weight`）计算 `cv2.Laplacian(gray - gray_ref).var() ** 2`（高频梯度变化方差），并在 `train.py` 中启用 `WeightedRandomSampler` 进行数据重采样（`--w_resampling 1`）。
   - **训练目标：** LSGAN 对抗损失 + `λ_L1 = 10.0` 的 L1 重建损失，Adam 优化器（`lr=0.0002, beta1=0.5, batch_size=64, niter=100`）。
4. **核心实验指标：**
   - **Vision → Touch：** AMT 真假判别（越高越逼真，50% 为完全以假乱真）：Seen `46.63%` vs pix2pix `28.09%`；Unseen `38.22%` vs pix2pix `21.74%`。接触瞬间时刻误差与标记点 L2 形变位移误差显著低于消融变体（README 复现指标：Seen `0.6439`，Unseen `0.7573`）。
   - **Touch → Vision：** "Feels Similar" 人类主观一致性评测（400 张测试图）：Unseen `85.29%` / Seen `90.37%`（对比 pix2pix `25.21%` / `44.52%`，以及 1,000 帧人工标注微调的监督 Stacked Hourglass `83.44%` / `89.20%`）；AMT 真假判别通过率 Seen `30.50%`、Unseen `24.22%`。

---

## 仓库代码核对

- `scripts/train_vision2touch.sh` / `scripts/train_touch2vision.sh` → 调用 `train.py --direction <vision2touch|touch2vision> --w_timewindow 1 --w_resampling 1`。
- `data.py` 定义 `VisionTouchDataset` 与 `variance_of_laplacian` / `calc_weight`，返回 `(ref_src_lowres, ref_des_lowres, src_lowres, des_lowres, ref_src, ref_des, src)`。
- `models.py` 定义 `_netG_resnet`（双 ResNet-18 编码器 + `decoder.convt_0..5`）、`_netD`（5 层下采样判别器，输入通道随 `w_timewindow` 为 `nc*3+5=14`）与 `GANLoss`。
- `eval.py` 根据 `--branch demo|eval` 读取对应列表并渲染预测/真值对比序列到 `dump_<direction>/`。

---

## 映射到 Wiki

- [dataset-visgel.md](../../wiki/entities/dataset-visgel.md)
