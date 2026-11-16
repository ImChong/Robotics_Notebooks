# CamUVA Rollouts 视频展示与同步脚本（官方仓库）

- **GitHub：** https://github.com/camx-anonymous/camuva-rollouts
- **站点：** https://camx-anonymous.github.io/camuva-rollouts/
- **核查日期：** 2026-11-09
- **核对范围：** README、index.html 评测分组/视频渲染逻辑、tools/build_videos.py 的读取/编码入口；未逐帧播放全部视频或重建原始 rollout。
- **通读：** 未完成

## 资料结论

- index.html 读取 videos.json（任务、平台、策略、回放片段、成功/失败、相机视图与策略输入标签），按 in-distribution embodiment 和 cross-embodiment 两类设置展示。
- README 表示原始 episode 的多相机 timestamp 被统一到 30 fps 时间轴，第三视角相机仅用于观看、不输入策略；RB-Y1 的第三方手机视角单独对齐。
- tools/build_videos.py 从原始 episode 文件、相机 zarr 时间戳、任务 metadata 生成 videos/、posters/、videos.json；相关脸部检测脚本用于排除带有旁观者面部的片段。
- 此仓库为评测**视频记录/发布工具**，不是公开的 CamUVA 网络训练、模型权重或机器人实时部署栈。不能把 success 标签当外部独立复测。

## 可辨认源码入口

```bash
python tools/build_videos.py --root rollouts --site /path/to/camuva-rollouts
```

需先有原始 rollout episode、相机视频与时间戳及 ffmpeg / zarr / numpy / av 等依赖；源码包含同步、视频拼接、poster 生成和后处理。

## 知识库映射

[CamX / CamUVA 主实体](../../wiki/entities/project-camx-camuva.md)；[官方主页](../sites/camx-anonymous-github-io.md)；[CamX 配套仓库](camx-dataset-site.md)。
