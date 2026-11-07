# HachimoDock 嘉立创硬件资料 / Hardware replication resources

- **类型：** site
- **项目页：** https://oshwhub.com/eda_gqvzlprk/project_cnbmkbjc
- **代码：** https://github.com/YizhengWw/HachimoDock
- **版本：** 页面显示更新于 2026-10-30 15:17:54；2026-11-07 浏览器读取。
- **核对范围：** 通读可见项目描述、整机物料清单、三种方案、使用指南、FAQ、许可段落与附件/模型清单。未下载检查 STEP 几何、设计工程或观看装配视频，不据此声称硬件复刻已验证。
- **通读：** 已完成
- **入库日期：** 2026-11-07
- **沉淀到 wiki：** [HachimoDock](../../wiki/entities/hachimodock.md)

## 已确认资料

页面标注复刻成本 ￥150；描述内有整机物料表，列出微雪 ESP32-P4-WIFI6 / WIFI6-M 开发板、2.8 英寸 480×640 MIPI-DSI 屏、排母、摇杆、三机械轴、扬声器、螺钉与热熔铜螺母。平台独立 BOM 区却显示「暂无BOM」：不能把文字物料表误说成可直接下单的 EDA BOM。

3D 模型区列有 `屏幕lens 3d打印版.STEP` 与 `桌搭-整机外壳3d打印版.STEP`，附件区列有 `拼装视频.mp4`。FAQ 说明外壳可委托打印，一套几十元；采购、最小包装、打板、结构件与运费均需按实际核算，￥150 不是当前到手报价。

## 版本与许可冲突

页面头部及软件 FAQ 仍标 GPL 3.0 / GPL-3.0-only；当前 GitHub LICENSE 已明确为自定义非商业源码可见许可。两处不一致，不能将硬件页旧文案扩展为当前软件的商用授权，也不能推断所有历史版本均已被撤销许可。硬件、结构件与素材应分别核对随附声明。

页面旧方案（Radxa / Raspberry Pi）已经标为废弃，部分 MQTT、CLI 文案仍在；当前实现以 GitHub 的 ESP32-P4 + USB 文档和实际能力为准。

## 相关来源

- [源码与运行入口归档](../repos/hachimodock.md)
- [当前许可证](https://github.com/YizhengWw/HachimoDock/blob/7fbe729aa52911da5fdd7e8c3c0578d4a4d8f60f/LICENSE)
