# VICAR 官方 GitHub 仓库归档

> 来源归档（官网链接指向的公共仓库；读取日期：2026-10-13）

- **仓库：** <https://github.com/ember-center-berkeley/VICAR>
- **默认分支：** `website`
- **公开状态：** Public
- **功能定位：** 项目页、论文技能视频与 Viser 交互动作预览；不是论文完整训练仓库
- **对应论文：** [arXiv:2610.13079](https://arxiv.org/abs/2610.13079)
- **项目页：** <https://ember-center-berkeley.github.io/VICAR/>
- **整体许可证：** 根目录未发现统一项目 LICENSE；Viser 客户端保留 MIT，G1 展示模型保留 Unitree BSD 3-Clause 通知，研究轨迹/论文图由权利人持有。
- **入库日期：** 2026-10-13
- **关联 wiki：** [VICAR 实体](../../wiki/entities/paper-vicar-contact-aware-video-imitation.md)

## 仓库公开内容

官方 README 自称 static research project page。仓库当前包含 HTML/CSS/JS 项目网站、技能 MP4 与海报、生成的 viewer manifest、Viser 静态客户端、可移植 motion NPZ、G1 的可视化 URDF/mesh、motion import/export、桌面本地播放和 browser checks。

本地播放示例（Python 3.10+）：

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r visualization/requirements.txt
python -m visualization list
python -m visualization view --task ladder-climbing
```

通过 `python -m visualization export` 可导出静态 Viser 播放资产；该浏览器动作预览是已保存运动学轨迹的回放，不是在线物理仿真，也不运行 VICAR optimizer 或 PPO。

## 研究代码开放状态

官方 `docs/VISUALIZATION_SOURCES.md` 明确区分：

- **此公开仓库的 `visualization/`：** 可移植的数据准备、导入校验、共享渲染器、本地 player、静态站点 exporter。
- **外部 TT_PLayer checkout：** 原始训练、refinement 与 augmentation 源码；该文档锁定的来源 revision 为 `b377ba951d2bd12bef45e23a5e623342038121e0`，其中列出 `augment_*_g1.py` / `refine_*_g1.py` 任务脚本。
- 因而，仅克隆 VICAR 项目页仓库不足以复现论文完整的接触重定向优化或 Isaac Lab/PPO 训练。网页的可视化依赖预生成运动资产；source map 中的外部路径不代表这些训练脚本已包含于当前公开仓库。

## 许可与资产边界

- arXiv HTML v1 标注论文 CC BY 4.0；这是论文文本许可。
- 仓库说明 Viser 1.1.1 client 自带 MIT license。
- 展示用 Unitree G1 资产保留 Unitree BSD 3-Clause notice。
- 仓库未发现统一根级 LICENSE，且 README 指出论文图、视频/研究轨迹保留各自权利人权利；复用时应逐项核对资产来源/许可。
