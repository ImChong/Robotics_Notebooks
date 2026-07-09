/**
 * Wiki 节点类型展示标签（中文 / 英文双语与纯中文）。
 * - 节点类型 UI：formatBilingual → 「概念 (Concept)」
 * - 更新记录时间线：formatChinese → 「概念」
 */
(function () {
  var ZH = {
    concept: '概念',
    method: '方法',
    task: '任务',
    comparison: '对比',
    entity: '实体',
    query: '查询',
    formalization: '形式化',
    overview: '总览',
    reference: '参考',
    roadmap: '路线',
    roadmap_page: '路线',
    wiki_page: '知识页',
    entity_page: '实体',
    reference_page: '参考',
    tech_map_node: '技术节点',
    detail_page: '详情页',
    '': '知识页'
  };

  var EN = {
    concept: 'Concept',
    method: 'Method',
    task: 'Task',
    comparison: 'Comparison',
    entity: 'Entity',
    query: 'Query',
    formalization: 'Formalization',
    overview: 'Overview',
    reference: 'Reference',
    roadmap: 'Roadmap',
    roadmap_page: 'Roadmap',
    wiki_page: 'Wiki',
    entity_page: 'Entity',
    reference_page: 'Reference',
    tech_map_node: 'Tech Node',
    detail_page: 'Detail Page',
    '': 'Wiki'
  };

  function normalizeType(type) {
    if (type == null || type === '') return '';
    return String(type);
  }

  function formatChinese(type) {
    var key = normalizeType(type);
    return ZH[key] || (key ? key : ZH['']);
  }

  function formatBilingual(type) {
    var key = normalizeType(type);
    var zh = formatChinese(key);
    var en = EN[key] || (key ? key : EN['']);
    if (!en || zh === en) return zh;
    return zh + ' (' + en + ')';
  }

  window.RNWikiTypeLabels = {
    ZH: ZH,
    EN: EN,
    formatChinese: formatChinese,
    formatBilingual: formatBilingual
  };
})();
