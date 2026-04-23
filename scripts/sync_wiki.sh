#!/bin/bash
# scripts/sync_wiki.sh - 自动化 Wiki 维护与统计同步脚本
set -e

# 检查是否提供了描述
if [ -z "$1" ]; then
  echo "❌ 错误: 请提供操作描述 (DESC)"
  echo "用法: ./scripts/sync_wiki.sh \"描述内容\""
  exit 1
fi

DESC=$1

echo "--- 📂 步骤 1: 更新索引 (make catalog) ---"
python3 scripts/generate_page_catalog.py

echo "--- 📊 步骤 2: 生成图谱与主页统计 (make graph) ---"
python3 scripts/generate_link_graph.py
python3 scripts/generate_home_stats.py
cp exports/link-graph.json docs/exports/link-graph.json
cp exports/graph-stats.json docs/exports/graph-stats.json 2>/dev/null || true
cp exports/home-stats.json docs/exports/home-stats.json 2>/dev/null || true

echo "--- 🚀 步骤 3: 导出全站数据 (make export) ---"
python3 scripts/export_minimal.py

echo "--- 📝 步骤 4: 记录变更日志 (make log) ---"
python3 scripts/append_log.py ingest "$DESC"

echo "--- ✅ 同步完成! ---"
echo "主页节点/连接数已更新。你可以运行 'git status' 检查变更并提交。"
