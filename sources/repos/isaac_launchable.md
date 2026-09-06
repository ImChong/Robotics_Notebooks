# isaac-sim/isaac-launchable

> 来源归档

- **标题：** Isaac Launchable
- **类型：** repo / Brev Launchable 模板
- **来源：** NVIDIA Isaac Sim 团队
- **链接：** https://github.com/isaac-sim/isaac-launchable
- **Brev Deploy：** https://brev.nvidia.com/launchable/deploy/now?launchableID=env-35JP2ywERLgqtD0b0MIeK1HnF46
- **许可证：** Other（GitHub API：`NOASSERTION`；部署需接受 [Isaac Sim Additional Software License](https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/)）
- **入库日期：** 2026-09-06
- **一句话说明：** 浏览器内跑 Isaac Lab + Isaac Sim 的 Brev Launchable 模板：VS Code 开发 tab + Kit App Streaming `/viewer` 视口；Docker 钉 **Lab 3.0.0-beta2-post1**、**Sim 6.0.1**；仅学习用途，按小时计费。
- **代码：** https://github.com/isaac-sim/isaac-launchable（**已开源** 仓库可 fork；Isaac/Sim 组件受 NVIDIA 许可约束）
- **沉淀到 wiki：** [`wiki/entities/isaac-launchable.md`](../../wiki/entities/isaac-launchable.md)

---

## 栈与容器（README，2026-09-06）

| 组件 | 版本 / 说明 |
|------|-------------|
| Isaac Lab | 3.0.0-beta2-post1 |
| Isaac Sim | 6.0.1 |
| VS Code | 独立容器（Secure Links 或 `VSCODE_PASSWORD`） |
| Kit App Streaming | 基于 [web-viewer-sample](https://github.com/NVIDIA-Omniverse/web-viewer-sample) |
| 编排 | `isaac-lab/docker-compose.yml`（`ENV=brev` 或 `localhost`） |

运行中容器名（排障）：`isaac-lab-nginx`、`isaac-lab-vscode`、`isaac-lab-viewer`。

## Quickstart（官方 Deploy）

1. README **Deploy Now** → Brev Launchable `env-35JP2ywERLgqtD0b0MIeK1HnF46`
2. 实例 **running + built + setup 完成** 后，打开 Secure Links 分享的 VS Code URL
3. 需 UI 时：同 host 新开 tab 访问 **`/viewer`**（例：`ec2.*.amazonaws.com/viewer`）
4. **仅保持一个 viewer tab**；不用时 **stop 实例** 省 credits

## 常用命令

```bash
# 仅 Isaac Sim UI（headless + streaming）
/isaac-sim/runheadless.sh
# 等 console 出现 app ready → 打开 /viewer

# Isaac Lab 训练（无 UI）
./isaaclab.sh train --rl_library skrl --task Isaac-Ant-v0 --headless

# Isaac Lab 回放（要视口）
./isaaclab.sh play --rl_library skrl --task Isaac-Ant-v0 --livestream 2
# 等 Simulation App Startup Complete → /viewer
```

## 自定义 Launchable（Brev 控制台）

Setup script 核心：

```bash
git clone https://github.com/isaac-sim/isaac-launchable
cd isaac-launchable/isaac-lab
docker compose up -d
```

- Secure Link：端口 **80**，名 `isaac`
- 流媒体端口：**1024、47998、49100**
- GPU 需 **RT core**；**Crusoe 实例不兼容**；AWS 已测
- 可选 `export VSCODE_PASSWORD=...`

## 本地 Docker

1. 安装 `nvidia-container-toolkit`
2. `docker-compose.yml` 中 `ENV=brev` → `ENV=localhost`
3. `cd isaac-lab && docker compose up -d`
4. 浏览器访问 `localhost`

## 对 wiki 的映射

- 实体页 → [`wiki/entities/isaac-launchable.md`](../../wiki/entities/isaac-launchable.md)
- 平台 → [`wiki/entities/nvidia-brev.md`](../../wiki/entities/nvidia-brev.md)
- 课程云入口 → [`wiki/entities/nvidia-getting-started-isaac-lab.md`](../../wiki/entities/nvidia-getting-started-isaac-lab.md)、[`wiki/entities/nvidia-physical-ai-learning.md`](../../wiki/entities/nvidia-physical-ai-learning.md)
- 底座 → [`wiki/entities/isaac-lab.md`](../../wiki/entities/isaac-lab.md)、[`wiki/entities/isaac-sim.md`](../../wiki/entities/isaac-sim.md)
