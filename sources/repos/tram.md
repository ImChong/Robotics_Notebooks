# TRAM 官方实现

- **项目页：** <https://yufu-wang.github.io/tram4d/>
- **代码：** <https://github.com/yufu-wang/tram>
- **论文：** <https://arxiv.org/abs/2403.17346>
- **关联 wiki：** [TRAM](../../wiki/entities/paper-tram-global-human-motion.md)
- **核查：** 2026-10-02

README 指定 `--recursive` 克隆，编译修改过的 `thirdparty/DROID-SLAM`，下载模型与样例视频。运行入口顺序为 `scripts/estimate_camera.py`（掩码 SLAM 与人体检测跟踪）→ `scripts/estimate_humans.py`（VIMO 人体）→ `scripts/visualize_tram.py`（世界系输出）。SMPL 与第三方模型另有使用许可。
