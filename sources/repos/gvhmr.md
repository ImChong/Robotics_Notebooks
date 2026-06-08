# GVHMR

> 来源归档

- **标题：** GVHMR（Gravity-View Human Motion Recovery）
- **类型：** repo
- **链接：** https://github.com/zju3dv/GVHMR
- **入库日期：** 2026-06-08
- **一句话说明：** 单目视频人体全局运动恢复（SMPL 系），常作为 GMR / 人形重定向上游的「视频→人体轨迹」模块。
- **沉淀到 wiki：** 是 → [`wiki/entities/gvhmr.md`](../../wiki/entities/gvhmr.md)

## 与重定向链的关系

```
单目视频 → GVHMR（人体姿态/根轨迹）→ GMR / PHC / HTD-Refine 等 → 机器人参考
```

- [HTD-Refine](../../wiki/entities/paper-htd-refine-monocular-hmr.md) 论文实验将 GVHMR 作为可插拔初始化。
- [GMR](https://github.com/YanjieZe/GMR) README 声明支持 GVHMR 输入。
