# PR #411 图谱布局验证截图

本地验证命令：

```bash
cd docs && python3 -m http.server 8765
node scripts/screenshot_graph_layout_verify.cjs http://127.0.0.1:8765/graph.html
```

## 截图

| 场景 | 文件 |
|------|------|
| 首屏加载完成（587 节点居中分布） | [graph-layout-loaded.png](./graph-layout-loaded.png) |
| 点击空白后（无右下角偏移） | [graph-layout-after-blank-click.png](./graph-layout-after-blank-click.png) |

## 自动判据（`graph-layout-verify.json`）

- 首屏：`cornerCluster=false`，质心约 (610, 398)
- 点击空白后：`cornerBR=false`，质心约 (719, 393)

## PR 正文可粘贴的链接（合并前用分支 raw URL）

```
https://github.com/ImChong/Robotics_Notebooks/blob/claude/zen-volta-LTCDm/docs/checklists/verification/pr-411-graph-layout/graph-layout-loaded.png
https://github.com/ImChong/Robotics_Notebooks/blob/claude/zen-volta-LTCDm/docs/checklists/verification/pr-411-graph-layout/graph-layout-after-blank-click.png
```
