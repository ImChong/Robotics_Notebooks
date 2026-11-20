# AlayaWorld — 官方代码与运行入口

- **标题：** AlayaWorld
- **类型：** repo / 官方 README + 运行入口
- **机构：** Alaya Lab（AlayaWorld Team）
- **代码：** <https://github.com/AlayaLab/AlayaWorld>
- **项目页：** <https://alaya-lab.github.io/AlayaWorld/>
- **论文/技术报告：** <https://arxiv.org/abs/2607.18367>
- **读入日期：** 2026-11-20
- **核对范围：** 官方代码仓 README.md 全文、inference/run.py，alaya/inference/pipeline.py，README Quick Start 和 docs/vigeo/README.md；核对模型发布说明、安装/推理入口及许可证说明（未实际下载权重或跑 GPU）。
- **通读：** 已完成

## 可复核的实现事实

- **功能：** 因果视频模型结合 sink frame、压缩时序历史、几何对齐空间缓存与最近帧，支持相机轨迹及生成中切换文本提示。
- **主要组件：** LTX-2.3 DiT、FlashAlayaPipeline、sink、时序缓存与几何缓存
- **公开边界：** 推理、训练、checkpoint 已发布，外部依赖/门控模型和许可证需要单独核对
- **官方实现入口：** inference/run.py，alaya/inference/pipeline.py，README Quick Start 和 docs/vigeo/README.md

## 数据与控制流

1. 首帧、相机轨迹和可切换提示
2. 构建 sink + 压缩历史 + 空间记忆 + 最近帧
3. 因果 4-step/teacher 视频块生成
4. 重投影缓存并更新滚动上下文
5. 视频解码与交互流展示

## 复现步骤

`pip install -r requirements.txt`、安装 Depth-Anything-3 依赖、下载官方相应模型后，运行 `bash inference/run.sh` 或 `python -m inference.run --input playground/case1/case1`。训练入口 `CONFIG_PATH=configs/<config>.yaml bash scripts/finetune/train.sh`；浏览器交互从 `reactor/` 单独部署。

## 可信边界与限制

部署涉及大权重、DA3/ViGeo 外部依赖与受门控的 Gemma 文本编码器，不能把“开放代码”解读为所有依赖即取即用。论文多版本 checkpoint 功能范围不同，不能混用；生成一致性不能证明符合真实碰撞动力学。

## 关联节点

- [AlayaWorld 详情](../../wiki/entities/paper-alayaworld.md)
- [11 项世界模型技术对照](../../wiki/comparisons/open-source-world-models-11-2026.md)
