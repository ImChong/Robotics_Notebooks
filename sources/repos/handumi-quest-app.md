# HandUMI Quest App（robonet-ai/handumi-quest-app）

> 来源归档

- **标题：** HandUMI Quest App
- **类型：** repo（Meta Quest 头显应用）
- **链接：** https://github.com/robonet-ai/handumi-quest-app
- **Release APK：** https://github.com/robonet-ai/handumi-quest-app/releases（文档示例 v0.2.1）
- **配套软件：** https://github.com/robonet-ai/handumi-sw
- **机构：** RoboNet AI
- **许可证：** 仓库页截至 2026-07-27 **未标注 SPDX**（以仓内 LICENSE / Releases 为准）
- **入库日期：** 2026-07-27
- **一句话说明：** Unity 重建的 Meta Quest 应用，向 handumi-sw 流式传输 HMD + 双手柄遥测，作为 HandUMI 的 Quest 追踪后端（PICO 则走 XRoboToolkit）。
- **沉淀到 wiki：** [handumi](../../wiki/entities/handumi.md)

---

## 工程要点（文档站 setup）

1. 开启 Developer Mode → USB `adb` 授权 → `adb install -r handumi-quest-app-*.apk`
2. 将 `adb shell ip route` 得到的地址写入 `configs/rig.yaml` 的 `meta_quest.connection.quest_ip`
3. 从 Library → Unknown Sources 启动应用并保持前台
4. 工作站侧：`python -m handumi.tracking.meta_quest --config configs/rig.yaml` 或经 `handumi record --device meta`

---

## 对 wiki 的映射

- [handumi](../../wiki/entities/handumi.md) · [handumi-sw](./handumi-sw.md) · [handumi-hw](./handumi-hw.md)
