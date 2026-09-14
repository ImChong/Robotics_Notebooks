# IsaacLab_SteadyTray（SteadyTray 定制 IsaacLab fork）

> 来源归档

- **标题：** IsaacLab_SteadyTray
- **类型：** repo
- **来源：** SteadyTray 作者（AllenHuangGit）
- **链接：** <https://github.com/AllenHuangGit/IsaacLab_SteadyTray>
- **入库日期：** 2026-09-14
- **一句话说明：** SteadyTray 训练所依赖的 IsaacLab 定制 fork；通过 Docker 挂载与主仓 `steadytray` 扩展包联用。
- **主仓：** [`steadytray.md`](./steadytray.md)
- **沉淀到 wiki：** [`wiki/entities/paper-notebook-steadytray.md`](../../wiki/entities/paper-notebook-steadytray.md)

---

## 复现关系

README 要求 **先构建本 fork 的 Docker 镜像**，再将 `SteadyTray` 仓库 bind-mount 到 `/workspace/SteadyTray`，于容器内 `pip install -e source/steadytray`。
