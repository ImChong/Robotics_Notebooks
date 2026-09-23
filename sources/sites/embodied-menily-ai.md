# 具身产业库在线站点（embodied.menily.ai）

- **标题：** 具身产业库 · 在线查询
- **类型：** site / GitHub Pages 静态站
- **URL：** <https://embodied.menily.ai>
- **代码：** <https://github.com/MasashiToda1/embodied-industry-db>（**已开源**；前端在仓库 `docs/`，数据 JSON 由 `scripts/build_site.py` 在 Pages 部署时生成）
- **入库日期：** 2026-09-23
- **一句话说明：** 具身产业库的只读浏览层：时间线、主体卡片、技术轴收敛热力、关系图谱与商业信号看板；纯静态、无框架。

## 五个视图（README / 站点核查 2026-09-23）

| 视图 | 用途 |
|------|------|
| 时间线 | 按日期排列可核查事件 |
| 主体卡片 | 编译自 `registry/` + `events/` 的主体页 |
| 轴 / 收敛 | 受控词表轴上的取值分布与收敛趋势（流图 + 热力） |
| 图谱 | 实线仅画事件中出现的关系（投资/客户/供应），可点回事件；共享技术栈为可选虚线叠加，默认关闭 |
| 商业信号 | 谁在投 / 谁在买 / 场景 / 公开价格（只加已披露金额、不换汇） |

## 本地预览

```bash
git clone https://github.com/MasashiToda1/embodied-industry-db
cd embodied-industry-db && make setup
make site-serve   # http://127.0.0.1:8800
```

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 站点 Footer / 源码链接 | 指向同一 GitHub 仓库 |
| 数据生成 | 部署时现场编译 JSON，大文件不入 git（与本库 site-data 策略类似） |
| 编辑入口 | 数据变更走仓库 `events/` + PR；无 Wiki 式在线编辑 |

## 对 wiki 的映射

- [具身产业库实体页](../../wiki/entities/embodied-industry-db.md)
- [仓库归档](../repos/embodied-industry-db.md)
