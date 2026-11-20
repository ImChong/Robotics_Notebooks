# HY-WorldPlay / HY-World 1.5 — 官方代码与运行入口

- **标题：** HY-WorldPlay / HY-World 1.5
- **类型：** repo / 官方 README + 运行入口
- **机构：** 腾讯混元（Tencent Hunyuan）
- **代码：** <https://github.com/Tencent-Hunyuan/HY-WorldPlay>
- **项目页：** <https://3d.hunyuan.tencent.com/sceneTo3D?tab=worldplay>
- **论文/技术报告：** <https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf>
- **读入日期：** 2026-11-20
- **核对范围：** 官方代码仓 README.md 全文、run.sh（蒸馏分支）与 hyvideo/generate.py（generate_video/main）；核对模型发布说明、安装/推理入口及许可证说明（未实际下载权重或跑 GPU）。
- **通读：** 已完成

## 可复核的实现事实

- **功能：** 以重建的历史上下文支持相机可控交互视频续写；提供 HunyuanVideo 8B 与轻量 WAN 5B 双推理路径。
- **主要组件：** HunyuanVideo-1.5、WAN、动作/相机条件与 context memory
- **公开边界：** 模型权重、推理和训练文档部分开放；个别模型变体仍待发布
- **官方实现入口：** run.sh（蒸馏分支）与 hyvideo/generate.py（generate_video/main）

## 数据与控制流

1. 输入首帧、文本和相机轨迹
2. 将动作/姿态嵌入模型条件
3. 双向教师或自回归/蒸馏世界模型
4. 重建历史上下文并预测后续视频
5. 输出可交互视频帧

## 复现步骤

在 README 按 `download_models.py` 下载模型，将权重路径填入 `run.sh`，再运行 `bash run.sh`；默认蒸馏分支用 `torchrun ... hyvideo/generate.py --model_type ar --few_step true --num_inference_steps 4`。`POSE='w-31'` 代表前进方向的 latent 轨迹。

## 可信边界与限制

README 的 RL+蒸馏联合权重仍列 To be released；使用者须区分仓库代码、已发布 checkpoint 与待发组件。`hyvideo/generate.py` 采用 Tencent Hunyuan Community License，不能擅自按 Apache 条款处理。项目技术报告出自 2025 年，不是 2026 年首次提出。

## 关联节点

- [HY-WorldPlay / HY-World 1.5 详情](../../wiki/entities/hy-worldplay-1-5.md)
- [11 项世界模型技术对照](../../wiki/comparisons/open-source-world-models-11-2026.md)
