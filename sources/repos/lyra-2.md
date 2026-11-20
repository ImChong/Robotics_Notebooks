# Lyra 2.0 — 官方代码与运行入口

- **标题：** Lyra 2.0
- **类型：** repo / 官方 README + 运行入口
- **机构：** 英伟达空间智能实验室（NVIDIA Spatial Intelligence Lab）
- **代码：** <https://github.com/nv-tlabs/lyra>
- **项目页：** <https://research.nvidia.com/labs/sil/projects/lyra2/>
- **论文/技术报告：** <https://arxiv.org/abs/2604.13036>
- **读入日期：** 2026-11-20
- **核对范围：** 官方代码仓 Lyra-2/README.md 全文、Lyra-2/lyra_2/_src/inference/lyra2_zoomgs_inference.py、vipe_da3_gs_recon.py；核对模型发布说明、安装/推理入口及许可证说明（未实际下载权重或跑 GPU）。
- **通读：** 已完成

## 可复核的实现事实

- **功能：** 单图与相机轨迹先生成空间连贯视频，再用 VIPE + Depth Anything 3 重建可导出的 3D Gaussian 场景。
- **主要组件：** 相机条件视频模型、逐帧 3D cache、VIPE、Depth Anything 3、3DGS
- **公开边界：** 训练/推理代码和权重已发布，模型许可限科研开发
- **官方实现入口：** Lyra-2/lyra_2/_src/inference/lyra2_zoomgs_inference.py、vipe_da3_gs_recon.py

## 数据与控制流

1. 起始图和相机轨迹
2. DA3 深度/相机估计与每帧几何缓存
3. 重投影检索历史可见帧
4. 自回归生成可探索视频并抑制漂移
5. VIPE + DA3 估计位姿深度
6. 3DGS 重建导出 PLY

## 复现步骤

官方推理入口 `PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference --input_image_path assets/samples --sample_id 4 --checkpoint_dir checkpoints/model --prompt_dir assets/samples --output_path outputs/zoomgs`；再运行 `python -m lyra_2._src.inference.vipe_da3_gs_recon --input_video_path outputs/zoomgs/videos/4.mp4`。超过 500 帧建议 `vipe_da3_chunked_gs_recon` 分块重建。

## 可信边界与限制

论文的主场景是静态环境，不显式模拟动态物体；训练数据光照变化可传入生成结果，损害 3DGS 重建。代码 Apache-2.0，但模型权重使用 NVIDIA Internal Scientific Research and Development Model License，不应误称可任意商用。

## 关联节点

- [Lyra 2.0 详情](../../wiki/entities/paper-lyra-2.md)
- [11 项世界模型技术对照](../../wiki/comparisons/open-source-world-models-11-2026.md)
