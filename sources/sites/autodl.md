# AutoDL（算力市场 / 炼丹平台）

> 来源归档

- **标题：** AutoDL 帮助文档与算力市场
- **类型：** site（国内 GPU 云算力平台）
- **来源：** AutoDL
- **链接：** https://www.autodl.com/docs/ 、https://www.autodl.com/market/list
- **入库日期：** 2026-07-02
- **一句话说明：** 面向深度学习与 AI 研发的国内 GPU 容器云：按物理主机租用 1–N 卡实例，内置 JupyterLab / SSH / VSCode Remote-SSH，提供社区镜像、公网网盘与炼丹会员体系。

## 为什么值得保留

- **机器人学习常见算力入口**：本库旧版资源地图与 `sources/train.md` 已列出 AutoDL；Isaac Lab / 大规模 PPO 训练常需 **24GB+ 显存 × 多卡**，个人工作站不足时云实例是现实选项。
- **文档体系成熟**：官方帮助文档覆盖快速开始、GPU 选型、计费、数据上传、VSCode/PyCharm 远程、守护进程与实例保留策略，适合作为「如何租卡跑训练」的操作参考。
- 统一选型：[china-gpu-cloud-platforms.md](../../wiki/comparisons/china-gpu-cloud-platforms.md)

## 平台要点（文档 2026-07 摘录）

### 产品形态

- **容器实例**：Docker 隔离的 GPU 开发机；**GPU 非共享**，按选定物理主机创建 N 卡实例。
- **资源配比**：CPU / 内存按 **GPU 数量成比例**分配（例：某主机 8 核 CPU、32GB 内存 / GPU → 租 2 卡即 16 核、64GB）。
- **严禁挖矿**：官方声明一经发现封号。

### 存储布局

| 名称 | 路径 | 说明 |
|------|------|------|
| 系统盘 | `/`（除特殊挂载外） | 约 30GB 本地 SSD；关机保留；可进镜像 |
| 数据盘 | `/root/autodl-tmp` | 50GB 起可扩容；高 IO；不进镜像 |
| 文件存储 | `/root/autodl-fs` | 同地区实例间共享；免费 20GB 起 |
| 公共数据 | `/root/autodl-pub` | 只读公共数据集/模型 |

### 典型工作流

1. 控制台「租用新实例」→ 选计费方式、地区、GPU 型号与数量、镜像。
2. 实例 **运行中** 开始计费；不用及时关机。
3. **JupyterLab** 上传数据 / 开终端训练；或 **VSCode Remote-SSH**（长训建议 `screen`/`tmux` 守护）。
4. 关机后环境与数据保留；**连续关机 15 天**实例释放（见实例数据保留说明）。

### GPU 选型要点（官方文档）

- **按任务选卡**：Pascal（1080Ti 等）入门小模型；Volta/Turing（V100、2080Ti）混合精度；Ampere（3090/4090/A100）主流训练；文档按架构列表 TensorCore / CUDA 版本要求。
- **CPU 同样重要**：DataLoader 瓶颈时官方推荐 **NVIDIA DALI** 等加速数据管线。
- **多卡数量**：文档建议按「24h 内完成一次实验」估算；8 卡为经典复现配置。
- **安培及以上**（3060、3090、4090、A100 等）需 **CUDA 11.1+**。

### 生态与会员

- 注册送 **炼丹会员**（1 个月）；学生认证可升级。
- 常用文档链：公网网盘、Git 克隆、开守护进程、FileZilla、PyCharm。
- 容器内 **不支持 Docker**；需 Docker 须联系客服租裸金属（整机包月）。

## 对 wiki 的映射

- 实体页：[autodl.md](../../wiki/entities/autodl.md)
- 统一选型：[china-gpu-cloud-platforms.md](../../wiki/comparisons/china-gpu-cloud-platforms.md)
- 交叉：[isaac-lab.md](../../wiki/entities/isaac-lab.md)、[robot-training-stack-layers-technology-map.md](../../wiki/overview/robot-training-stack-layers-technology-map.md)
