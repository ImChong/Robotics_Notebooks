# 恒源云（GPUShare）

> 来源归档

- **标题：** 恒源云用户文档
- **类型：** site（国内 GPU 云算力平台）
- **来源：** 恒源云
- **链接：** https://gpushare.com/docs/
- **入库日期：** 2026-07-02
- **一句话说明：** 专注 AI 行业的共享算力平台，提供 GPU 云主机与个人数据 OSS；支持包年/包月/包周/按量，实例数据目录 `/hy-tmp`，关机 10 天自动释放。

## 为什么值得保留

- **个人数据 OSS**：自研 `oss` CLI 可在开实例前上传压缩包至个人空间，节省运行中传数费用；实例内 `oss cp` 拉取。
- **竞价/空闲算力生态**：行业横评常将恒源云与 AutoDL 并列为高性价比选项。
- **灵活计费周期**：包周适合中期实验；按量适合短期调试。

## 平台要点（文档 2026-07 摘录）

### 实例与存储

| 路径 | 说明 |
|------|------|
| `/` | 系统盘约 **20GB** |
| `/hy-tmp` | 数据目录；同物理机多实例共享，容量用 `df -hT` 查询 |

### 工作流

1. 注册 → 用 **oss 工具**上传压缩包到个人数据（可不开实例）
2. 云市场选主机 → 创建实例（官方镜像：TF/PyTorch/MXNet/Paddle 等）
3. JupyterLab / SSH → `cd /hy-tmp` → `oss login` → `oss cp` 下载数据
4. 训练结束可 `shutdown` 关机；**停止 10 天**自动释放

### 资源配比

CPU/内存按租用 GPU 数占机器总卡数比例分配（例：8 卡机租 1 卡 → 1/8 CPU 与内存）。

## 对 wiki 的映射

- 实体页：[gpushare.md](../../wiki/entities/gpushare.md)
- 统一选型：[china-gpu-cloud-platforms.md](../../wiki/comparisons/china-gpu-cloud-platforms.md)
