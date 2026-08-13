// Verify graph filter accordion: only one of 维度/路线/机构 open; default 按社区;
// collapsed headers pin above/below the expanded pane.
const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

const OUT_DIR = path.resolve(__dirname, '..', '.cursor-artifacts', 'screenshots');
const ART_DIR = '/opt/cursor/artifacts/screenshots';
const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  '/usr/local/bin/google-chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);
const exe = CHROME_CANDIDATES.find((p) => fs.existsSync(p));
if (!exe) {
  console.error('No Chrome/Chromium found. Set CHROME_PATH.');
  process.exit(1);
}
const d3Body = fs.readFileSync(path.resolve(__dirname, '..', 'node_modules', 'd3', 'dist', 'd3.min.js'));

function copyToArtifacts(src, name) {
  try {
    fs.mkdirSync(ART_DIR, { recursive: true });
    const dest = path.join(ART_DIR, name);
    fs.copyFileSync(src, dest);
    return dest;
  } catch (_) {
    return null;
  }
}

async function sectionLayout(page) {
  return page.evaluate(() => {
    const ids = [
      'filter-dimension-section',
      'filter-depth-section',
      'filter-institution-section',
    ];
    const container = document.getElementById('filter-dimension-sections');
    const cRect = container.getBoundingClientRect();
    const items = ids.map((id) => {
      const el = document.getElementById(id);
      const r = el.getBoundingClientRect();
      const body = el.querySelector(
        ':scope > .filter-dimension-list, :scope > .filter-depth-chips, :scope > .filter-institution-list'
      );
      return {
        id,
        open: !!el.open,
        top: r.top,
        bottom: r.bottom,
        height: r.height,
        summaryH: el.querySelector('summary')?.getBoundingClientRect().height || 0,
        bodyClientH: body ? body.clientHeight : 0,
        bodyScrollH: body ? body.scrollHeight : 0,
        bodyCanScroll: !!(body && body.scrollHeight > body.clientHeight + 1),
        bodyOverflowY: body ? getComputedStyle(body).overflowY : '',
        chromeVisible: (() => {
          const chrome = el.querySelector(':scope > .filter-scroll-chrome');
          if (!chrome || chrome.hidden) return false;
          const thumb = chrome.querySelector('.filter-scroll-thumb');
          return !!(thumb && thumb.getBoundingClientRect().height >= 20);
        })(),
      };
    });
    return {
      containerTop: cRect.top,
      containerBottom: cRect.bottom,
      containerHeight: cRect.height,
      openCount: items.filter((i) => i.open).length,
      items,
    };
  });
}

function assertExclusive(layout, expectedOpenId) {
  if (layout.openCount !== 1) {
    throw new Error(`Expected exactly 1 open section, got ${layout.openCount}`);
  }
  const open = layout.items.find((i) => i.open);
  if (!open || open.id !== expectedOpenId) {
    throw new Error(`Expected open=${expectedOpenId}, got ${open && open.id}`);
  }
  // Expanded pane should meaningfully fill remaining space (taller than a summary row)
  // 路线视图除外：chip 紧凑换行，不吃满手风琴剩余高度
  if (expectedOpenId !== 'filter-depth-section' && open.height < 120) {
    throw new Error(`Open section too short to fill accordion: ${open.height}`);
  }
  if (expectedOpenId === 'filter-depth-section' && open.bodyClientH > open.bodyScrollH + 8) {
    throw new Error(
      `路线视图 chips should stay compact (client=${open.bodyClientH}, scroll=${open.bodyScrollH})`
    );
  }
  // Long lists (community / institution) must be height-clamped so the scrollbar appears
  if (
    (expectedOpenId === 'filter-dimension-section' ||
      expectedOpenId === 'filter-institution-section') &&
    !open.bodyCanScroll
  ) {
    throw new Error(
      `Open ${expectedOpenId} should scroll (client=${open.bodyClientH}, scroll=${open.bodyScrollH})`
    );
  }
  if (
    expectedOpenId !== 'filter-depth-section' &&
    open.bodyOverflowY !== 'auto' &&
    open.bodyOverflowY !== 'scroll'
  ) {
    throw new Error(`Open pane overflow-y should allow scroll, got ${open.bodyOverflowY}`);
  }
  // Body must be height-clamped inside the open section (not content-sized then clipped)
  if (
    expectedOpenId !== 'filter-depth-section' &&
    open.bodyClientH > open.height - open.summaryH + 2
  ) {
    throw new Error(
      `Open pane not clamped: bodyClient=${open.bodyClientH} section=${open.height} summary=${open.summaryH}`
    );
  }
  if (
    (expectedOpenId === 'filter-dimension-section' ||
      expectedOpenId === 'filter-institution-section') &&
    !open.chromeVisible
  ) {
    throw new Error(`Open ${expectedOpenId} should show custom scroll thumb`);
  }
  const openIdx = layout.items.findIndex((i) => i.id === expectedOpenId);
  layout.items.forEach((item, idx) => {
    if (item.open) return;
    // Collapsed ≈ summary height
    if (item.height > item.summaryH + 8) {
      throw new Error(`Collapsed ${item.id} still tall: ${item.height}`);
    }
    if (idx < openIdx) {
      // above open → near container top (stacked from top)
      if (item.top > layout.containerTop + 4) {
        // allow stacking of multiple above; first should touch top
        if (idx === 0 && item.top > layout.containerTop + 2) {
          throw new Error(`${item.id} should pin near container top`);
        }
      }
    } else {
      // below open → near container bottom chain
      if (idx === layout.items.length - 1) {
        if (Math.abs(item.bottom - layout.containerBottom) > 3) {
          throw new Error(`${item.id} should pin near container bottom`);
        }
      }
    }
  });
  // Container should be filled: first top ≈ container top, last bottom ≈ container bottom
  const first = layout.items[0];
  const last = layout.items[layout.items.length - 1];
  if (Math.abs(first.top - layout.containerTop) > 2) {
    throw new Error('First section should start at container top');
  }
  if (Math.abs(last.bottom - layout.containerBottom) > 3) {
    throw new Error('Last section should end at container bottom');
  }
}

(async () => {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--ignore-certificate-errors'],
    ignoreHTTPSErrors: true,
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
    await page.setRequestInterception(true);
    page.on('request', (req) => {
      if (req.url().includes('cdn.jsdelivr.net/npm/d3')) {
        req.respond({ status: 200, contentType: 'application/javascript', body: d3Body });
      } else {
        req.continue();
      }
    });
    const base = process.env.GRAPH_BASE_URL || 'http://127.0.0.1:8765/graph.html';
    await page.goto(base, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
      const el = document.getElementById('graph-node-count');
      return el && el.textContent && !el.textContent.includes('加载中');
    }, { timeout: 25000 }).catch(() => {});
    await new Promise((r) => setTimeout(r, 3500));
    await page.evaluate(() => {
      const ld = document.getElementById('graph-loading');
      if (ld) ld.style.display = 'none';
    });

    await page.click('#filter-toggle');
    await new Promise((r) => setTimeout(r, 500));

    let layout = await sectionLayout(page);
    console.log('Default layout:', JSON.stringify(layout, null, 2));
    assertExclusive(layout, 'filter-dimension-section');

    const defOut = path.join(OUT_DIR, 'graph-filter-accordion-community.png');
    await page.screenshot({ path: defOut, fullPage: false });
    copyToArtifacts(defOut, 'graph-filter-accordion-community.png');
    console.log('Saved:', defOut);

    // Open 路线视图 → dimension+institution collapse; institution pins bottom
    await page.click('#filter-depth-section > summary');
    await new Promise((r) => setTimeout(r, 350));
    layout = await sectionLayout(page);
    console.log('Depth open layout:', JSON.stringify(layout, null, 2));
    assertExclusive(layout, 'filter-depth-section');
    const dim = layout.items[0];
    const inst = layout.items[2];
    if (dim.open || inst.open) throw new Error('Others should be closed when depth opens');
    if (Math.abs(dim.top - layout.containerTop) > 2) {
      throw new Error('按社区 should pin to top when 路线视图 open');
    }
    if (Math.abs(inst.bottom - layout.containerBottom) > 3) {
      throw new Error('研究机构 should pin to bottom when 路线视图 open');
    }
    const depthOpen = layout.items[1];
    if (depthOpen.bodyClientH > depthOpen.bodyScrollH + 8) {
      throw new Error(
        `路线视图 chips stretched: client=${depthOpen.bodyClientH} scroll=${depthOpen.bodyScrollH}`
      );
    }
    if (depthOpen.height > layout.containerHeight - dim.summaryH - inst.summaryH - 24) {
      throw new Error(
        `路线视图 should not consume remaining accordion height: ${depthOpen.height} / ${layout.containerHeight}`
      );
    }

    const depthOut = path.join(OUT_DIR, 'graph-filter-accordion-depth.png');
    await page.screenshot({ path: depthOut, fullPage: false });
    copyToArtifacts(depthOut, 'graph-filter-accordion-depth.png');
    console.log('Saved:', depthOut);

    // Open 研究机构 → first two pin to top
    await page.click('#filter-institution-section > summary');
    await new Promise((r) => setTimeout(r, 350));
    layout = await sectionLayout(page);
    console.log('Institution open layout:', JSON.stringify(layout, null, 2));
    assertExclusive(layout, 'filter-institution-section');
    if (layout.items[0].open || layout.items[1].open) {
      throw new Error('Others should be closed when institution opens');
    }
    if (Math.abs(layout.items[0].top - layout.containerTop) > 2) {
      throw new Error('按社区 should remain at top');
    }
    if (layout.items[1].top < layout.items[0].bottom - 1) {
      throw new Error('路线视图 should stack under 按社区 at top');
    }

    const instOut = path.join(OUT_DIR, 'graph-filter-accordion-institution.png');
    await page.screenshot({ path: instOut, fullPage: false });
    copyToArtifacts(instOut, 'graph-filter-accordion-institution.png');
    console.log('Saved:', instOut);

    // Clicking open summary must not collapse to zero
    await page.click('#filter-institution-section > summary');
    await new Promise((r) => setTimeout(r, 200));
    layout = await sectionLayout(page);
    if (layout.openCount !== 1 || !layout.items[2].open) {
      throw new Error('Clicking open summary should keep institution expanded');
    }

    console.log('OK: filter accordion exclusive + pin top/bottom verified');
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
