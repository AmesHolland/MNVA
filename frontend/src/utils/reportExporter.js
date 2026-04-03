// src/utils/reportExporter.js
// ============================================================
// 结构化报告导出器 v2
// 核心修复：html2canvas 不渲染 off-screen 或 hidden 元素
// 改为使用全屏可见遮罩层 + 可见报告容器
// ============================================================

import * as echarts from 'echarts'
import html2pdf from 'html2pdf.js'

// ===================== 1. 离屏图表捕获 =====================

async function ensureWorldMap() {
  if (!echarts.getMap('world')) {
    const res = await fetch('https://cdn.jsdelivr.net/npm/echarts@4.9.0/map/json/world.json')
    echarts.registerMap('world', await res.json())
  }
}

/**
 * ECharts getDataURL() 不依赖 DOM 可见性，
 * 所以图表捕获仍可用低 z-index 的方式
 */
async function captureChart(option, width = 900, height = 500, waitMs = 500) {
  const el = document.createElement('div')
  el.style.cssText = `width:${width}px;height:${height}px;position:fixed;left:0;top:0;z-index:-1;opacity:0;pointer-events:none;`
  document.body.appendChild(el)

  const chart = echarts.init(el, null, { renderer: 'canvas' })
  chart.setOption(option, true)
  await new Promise(r => setTimeout(r, waitMs))

  const url = chart.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#ffffff' })
  chart.dispose()
  document.body.removeChild(el)
  return url
}

// ===================== 2. 图表 Option 构建 =====================

function buildGlobalMapOption(rawData) {
  return {
    backgroundColor: '#ffffff',
    geo: {
      map: 'world', roam: false, zoom: 1.2,
      itemStyle: { areaColor: '#e8ecf1', borderColor: '#fff', borderWidth: 1 },
      emphasis: { disabled: true }
    },
    series: [{
      type: 'effectScatter', coordinateSystem: 'geo',
      data: rawData.map(item => ({
        name: item.topic_name,
        value: [item.lon, item.lat, item.intensity],
        itemStyle: { color: '#F56C6C', shadowBlur: 5, shadowColor: '#F56C6C', opacity: 0.8 }
      })),
      symbolSize: val => Math.max(8, Math.min(val[2] * 4, 28)),
      showEffectOn: 'render',
      rippleEffect: { brushType: 'stroke', scale: 3 }
    }]
  }
}

function buildDeepDiveMapOption(mapData) {
  const colorMap = {
    'Technology': '#409EFF', 'Politics': '#E6A23C', 'Military': '#F56C6C',
    'Governance': '#909399', 'Environment': '#67C23A', 'Resources': '#eacc69'
  }
  const seriesData = mapData.map(item => ({
    name: item.name,
    value: [item.lon, item.lat],
    itemStyle: { color: colorMap[item.type] || '#409EFF', opacity: 1 }
  }))
  const avgLon = seriesData.reduce((s, d) => s + d.value[0], 0) / (seriesData.length || 1)
  const avgLat = seriesData.reduce((s, d) => s + d.value[1], 0) / (seriesData.length || 1)

  return {
    backgroundColor: '#ffffff',
    geo: {
      map: 'world', roam: false, zoom: 3, center: [avgLon, avgLat],
      itemStyle: { areaColor: '#e8ecf1', borderColor: '#fff' },
      emphasis: { disabled: true }
    },
    series: [{
      type: 'effectScatter', coordinateSystem: 'geo',
      data: seriesData, symbolSize: 12,
      showEffectOn: 'render', rippleEffect: { scale: 4 }
    }]
  }
}

function buildRelationGraphOption(graphData) {
  const { nodes, links } = graphData
  const colorMap = {
    'Conflict': '#F56C6C', 'Cooperation': '#67C23A', 'Diplomacy': '#E6A23C',
    'Trade': '#409EFF', 'Other': '#909399'
  }
  return {
    backgroundColor: '#ffffff',
    series: [{
      type: 'graph', layout: 'circular',
      circular: { rotateLabel: false },
      data: nodes.map(n => ({
        name: n.id, symbolSize: 42,
        itemStyle: { color: '#2a5caa', borderColor: '#fff', borderWidth: 2 },
        label: { show: true, position: 'bottom', color: '#303133', fontWeight: 'bold', fontSize: 11 }
      })),
      links: links.map(l => ({
        source: l.source, target: l.target,
        lineStyle: {
          color: colorMap[l.type] || '#909399',
          width: Math.min(Math.max(l.value || 1, 1), 5),
          curveness: 0.3, opacity: 0.7
        },
        label: {
          show: true, formatter: l.type || '', fontSize: 9,
          backgroundColor: colorMap[l.type] || '#909399',
          color: '#fff', padding: [2, 5], borderRadius: 10, rotate: 0
        }
      })),
      roam: false, edgeSymbol: ['none', 'arrow'], edgeSymbolSize: [4, 10]
    }]
  }
}

// ===================== 3. 采集所有图表 =====================

async function captureAllCharts(tasks, onProgress) {
  await ensureWorldMap()
  const images = {}

  for (const [taskId, task] of Object.entries(tasks)) {
    const viz = task.visualization_data
    if (!viz) continue

    if (task.agent_name === 'Global_Monitor_Agent' && viz.geo_dynamic_data?.length > 0) {
      onProgress?.('Rendering global map...')
      images[`${taskId}__global_map`] = await captureChart(buildGlobalMapOption(viz.geo_dynamic_data), 860, 440)
    }
    if (task.agent_name === 'Deep_Dive_Agent' && viz.map_chart?.length > 0) {
      onProgress?.('Rendering entity map...')
      images[`${taskId}__deep_dive_map`] = await captureChart(buildDeepDiveMapOption(viz.map_chart), 860, 440)
    }
    if (task.agent_name === 'Relation_Miner_Agent' && viz.graph_chart?.nodes) {
      onProgress?.('Rendering relation graph...')
      images[`${taskId}__relation_graph`] = await captureChart(buildRelationGraphOption(viz.graph_chart), 760, 560, 800)
    }
  }
  return images
}

// ===================== 4. Section → Chart 关联 =====================

function getRelatedChartKeys(section, chartImages) {
  const keys = []
  const claims = section.claims || section.content_claims || []
  const taskIds = new Set()
  claims.forEach(c => { if (c.source_subtask) taskIds.add(c.source_subtask) })
  for (const chartKey of Object.keys(chartImages)) {
    for (const tid of taskIds) {
      if (chartKey.startsWith(tid + '__')) keys.push(chartKey)
    }
  }
  return keys
}

// ===================== 5. HTML 构建 =====================

function esc(s) {
  if (!s) return ''
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function chartLabel(key) {
  if (key.includes('global_map')) return 'Spatiotemporal Event Distribution'
  if (key.includes('deep_dive_map')) return 'Entity Trajectory Map'
  if (key.includes('relation_graph')) return 'Entity Interaction Network'
  return 'Visualization'
}

function buildReportHTML(results, chartImages) {
  const report = results.report
  const phases = results.phase_data_list || []
  const pool = results.evidence_pool || []
  const sections = report.sections || []
  const usedKeys = new Set()
  // 🔥 新增：用于全局跨章节去重的 Set
  const globalRenderedCharts = new Set()
  let html = ''

  // ---------- 封面 ----------
  html += `
    <div style="text-align:center; padding:50px 0 36px 0; border-bottom:3px double #1e3a8a; margin-bottom:36px;">
      <div style="font-size:11px; color:#94a3b8; letter-spacing:3px; text-transform:uppercase; margin-bottom:18px;">
        EventHub · Multi-Agent Visual Analytics Report
      </div>
      <div style="font-size:22px; color:#1e3a8a; font-weight:700; line-height:1.4; margin:0 0 14px 0;">
        ${esc(report.report_title)}
      </div>
      <div style="font-size:11px; color:#94a3b8; margin-top:18px;">
        Generated: ${new Date().toLocaleDateString('en-US', { year:'numeric', month:'long', day:'numeric' })}
        &nbsp;·&nbsp; ${pool.length} source records
        ${phases.length > 0 ? `&nbsp;·&nbsp; ${phases.length} event phases` : ''}
      </div>
    </div>`

  // ---------- Executive Summary ----------
  if (report.executive_summary) {
    html += `
    <div style="background:#f8fafc; border-left:4px solid #3b82f6; padding:14px 18px; border-radius:0 6px 6px 0; margin-bottom:28px;">
      <div style="font-size:11px; color:#3b82f6; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Executive Summary</div>
      <div style="font-size:13px; line-height:1.9; text-align:justify; color:#444;">${esc(report.executive_summary)}</div>
    </div>`
  }

  // ---------- Phase Overview ----------
  if (phases.length > 0) {
    const colors = ['#3b82f6','#8b5cf6','#06b6d4','#10b981','#f59e0b','#ef4444']
    html += `<div style="font-size:15px; color:#1e3a8a; font-weight:700; border-bottom:2px solid #e2e8f0; padding-bottom:8px; margin:28px 0 14px 0;">Event Phase Overview</div>`
    phases.forEach((p, i) => {
      const c = colors[i % colors.length]
      html += `
      <div style="display:flex; gap:12px; align-items:flex-start; padding:10px 14px; margin-bottom:8px; background:#fff; border:1px solid #e2e8f0; border-radius:6px; border-left:4px solid ${c}; page-break-inside:avoid;">
        <div style="min-width:28px; height:28px; background:${c}; color:#fff; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:12px; flex-shrink:0; line-height:28px; text-align:center;">${p.phase_index}</div>
        <div style="flex:1;">
          <div style="font-size:13px; font-weight:700; color:#303133;">${esc(p.phase_name)}</div>
          <div style="font-size:10px; color:${c}; font-family:Consolas,monospace; margin:3px 0 5px 0;">${esc(p.phase_time_range || '')}</div>
          <div style="font-size:12px; color:#64748b; line-height:1.6;">${esc(p.phase_summary)}</div>
        </div>
      </div>`
    })
  }

  // ---------- 正文 Sections ----------
  sections.forEach((section, si) => {
    // 字号微调到16px，略微缩小 margin-bottom
    html += `<div style="font-size:16px; color:#1e3a8a; font-weight:700; border-bottom:2px solid #e2e8f0; padding-bottom:6px; margin:28px 0 10px 0;">${si + 1}. ${esc(section.section_title)}</div>`
    // 字号放大到 14.5px，行高收紧到 1.65，颜色加深至 #333，增加段落底部间距
    html += `<div style="font-size:14.5px; line-height:1.65; text-align:justify; color:#333; margin-bottom:16px;">`

    // ... 原代码 ...
    const claims = section.claims || section.content_claims || []
    claims.forEach(claim => {
      const text = esc(claim.content || claim.statement || '')
      if (!text) return

      // 🔥 获取 ID 数组并拼接成字符串 (例如: "DOC_1, DOC_2")
      const sourceIds = claim.source_ids || []
      // 如果 ID 太长，建议用 .map(id => id.slice(-4)) 截取后几位，或者在前面做索引映射
      const idText = esc(sourceIds.join(', '))

      if (claim.is_subjective_insight) {
        html += `<span style="color:#2a5caa;">${text}</span>`
        // 用 idText 替换原来的 n
        if (sourceIds.length > 0) {
          html += `<sup style="font-size:9px; font-weight:700; background:#fdf6ec; color:#e6a23c; padding:2px 4px; border-radius:3px; margin:0 3px; vertical-align:text-top;">AI: ${idText}</sup>`
        }
      } else {
        html += `<span>${text}</span>`
        // 用 idText 替换原来的 n
        if (sourceIds.length > 0) {
          html += `<sup style="font-size:9px; font-weight:700; background:#ecf5ff; color:#409EFF; padding:2px 4px; border-radius:3px; margin:0 3px; vertical-align:text-top;">${idText}</sup>`
        }
      }
      html += ' '
    })
    // ... 原代码 ...
    html += `</div>`

    // 嵌入关联图表
    // 嵌入关联图表
  const relKeys = getRelatedChartKeys(section, chartImages)
  relKeys.forEach(key => {
    usedKeys.add(key) // 这个保留，为了过滤最后附录里“完全没用过”的图

    // 🔥 判断这张图是否在之前的章节已经渲染过了
    if (!globalRenderedCharts.has(key)) {
      globalRenderedCharts.add(key) // 标记为已渲染

      html += `
      <div style="text-align:center; margin:18px 0; page-break-inside:avoid; break-inside:avoid;">
        <img src="${chartImages[key]}" style="width:100%; border:1px solid #e2e8f0; border-radius:4px;" />
        <div style="font-size:11px; color:#909399; text-align:center; margin:8px 0 16px 0; font-style:italic;">Figure: ${esc(chartLabel(key))}</div>
      </div>`
    }
  })
  })

  // ---------- 未关联的图表 → 附录 ----------
  const unusedKeys = Object.keys(chartImages).filter(k => !usedKeys.has(k))
  if (unusedKeys.length > 0) {
    html += `<div style="font-size:15px; color:#1e3a8a; font-weight:700; border-bottom:2px solid #e2e8f0; padding-bottom:8px; margin:32px 0 14px 0;">Visual Analysis Appendix</div>`
    unusedKeys.forEach(key => {
      html += `
      <div style="text-align:center; margin-bottom:24px; page-break-inside:avoid;">
        <img src="${chartImages[key]}" style="width:100%; border:1px solid #e2e8f0; border-radius:4px;" />
        <div style="font-size:11px; color:#909399; text-align:center; margin:8px 0 16px 0; font-style:italic;">${esc(chartLabel(key))}</div>
      </div>`
    })
  }

  // ---------- Conclusion ----------
  html += `
    <div style="background:#f0fdf4; border-left:4px solid #10b981; padding:14px 18px; border-radius:0 6px 6px 0; margin-top:32px;">
      <div style="font-size:11px; color:#10b981; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Conclusion</div>
      <div style="font-size:13px; line-height:1.9; text-align:justify; color:#444;">${esc(report.conclusion)}</div>
    </div>`

  // ---------- 证据索引 ----------
  if (pool.length > 0) {
    html += `
    <div style="page-break-before:always; padding-top:20px;">
      <div style="font-size:15px; color:#1e3a8a; font-weight:700; border-bottom:2px solid #e2e8f0; padding-bottom:8px; margin-bottom:14px;">Appendix: Evidence Source Index (${pool.length} records)</div>
      <table style="width:100%; border-collapse:collapse; font-size:9px; line-height:1.4;">
        <thead>
          <tr style="background:#f1f5f9;">
            <th style="border:1px solid #e2e8f0; padding:4px 6px; text-align:left; color:#475569;">ID</th>
            <th style="border:1px solid #e2e8f0; padding:4px 6px; text-align:left; color:#475569;">Title</th>
            <th style="border:1px solid #e2e8f0; padding:4px 6px; text-align:left; color:#475569;">Source</th>
            <th style="border:1px solid #e2e8f0; padding:4px 6px; text-align:left; color:#475569;">Date</th>
            <th style="border:1px solid #e2e8f0; padding:4px 6px; text-align:left; color:#475569;">Region</th>
          </tr>
        </thead>
        <tbody>`
    pool.forEach((item, i) => {
      html += `
          <tr style="background:${i % 2 === 0 ? '#fff' : '#fafbfc'};">
            <td style="border:1px solid #e2e8f0; padding:3px 6px; font-family:Consolas,monospace; color:#64748b;">${esc(item.DOC_ID)}</td>
            <td style="border:1px solid #e2e8f0; padding:3px 6px; color:#303133;">${esc(item.title || '')}</td>
            <td style="border:1px solid #e2e8f0; padding:3px 6px; color:#64748b;">${esc(item.source || '')}</td>
            <td style="border:1px solid #e2e8f0; padding:3px 6px; color:#64748b;">${esc(item.publish_date || '')}</td>
            <td style="border:1px solid #e2e8f0; padding:3px 6px; color:#64748b;">${esc((item.country || '') + (item.region ? '·' + item.region : ''))}</td>
          </tr>`
    })
    html += `</tbody></table></div>`
  }

  // ---------- 页脚 ----------
  html += `
    <div style="text-align:center; margin-top:48px; padding-top:16px; border-top:1px solid #e2e8f0; font-size:10px; color:#c0c4cc;">
      Generated by EventHub Multi-Agent Visual Analytics System · ${new Date().toISOString().split('T')[0]}
    </div>`

  return html
}

// ===================== 6. 主入口 =====================

/**
 * 导出结构化 PDF 报告
 *
 * 用法（Dashboard.vue）：
 *   import { exportStructuredReport } from '../utils/reportExporter'
 *   await exportStructuredReport(store.analysisResults, (msg) => { exportStatus.value = msg })
 *
 * @param {Object} analysisResults - store.analysisResults
 * @param {Function} onProgress - 进度回调 (msg: string) => void
 */
// ===================== 6. 主入口 (全新架构：Iframe 原生打印流) =====================

export async function exportStructuredReport(analysisResults, onProgress) {
  if (!analysisResults?.report) {
    throw new Error('No report data available.')
  }

  // Step 1: 依然离屏渲染获取 ECharts 的 base64 图片 (你的原代码)
  onProgress?.('Capturing charts...')
  const chartImages = await captureAllCharts(analysisResults.tasks || {}, onProgress)

  // Step 2: 构建 HTML 字符串 (你的原代码)
  onProgress?.('Building report...')
  const reportHTML = buildReportHTML(analysisResults, chartImages)

  // Step 3: 🔥 核心改变 —— 创建一个隐藏的 iframe 作为纯净的渲染容器
  onProgress?.('Preparing PDF engine...')
  const iframe = document.createElement('iframe')
  // 将 iframe 隐藏，但不使用 display:none (防止某些浏览器不渲染)
  iframe.style.cssText = 'position:fixed; right:200%; bottom:200%; width:800px; height:1000px; border:none;'
  document.body.appendChild(iframe)

  const doc = iframe.contentWindow.document

  // Step 4: 写入纯净的 HTML 文档，并注入【完美打印的 CSS】
  doc.open()
  doc.write(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>${analysisResults.report.report_title || 'EventHub_Report'}</title>
      <style>
        /* 基础重置样式 */
        body {
          font-family: -apple-system, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
          color: #333;
          line-height: 1.8;
          padding: 0;
          margin: 0;
          background: #fff;
        }
        
        /* 报表主体容器 */
        .report-container {
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
        }

        /* * 🔥 打印机专属 CSS 控制 
         */
        @media print {
          /* 强制打印背景色（你的报告里有很多彩色小标签和背景框） */
          * {
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
          }
          
          /* 设置纸张大小和边距 */
          @page {
            size: A4 portrait;
            margin: 15mm; /* 留出好看的页边距 */
          }
          
          /* 防止图片和特定块被跨页截断 */
          img, table, tr, .avoid-break {
            page-break-inside: avoid;
            break-inside: avoid;
          }
          
          /* 强制换页标记（如果你需要在某些章节前强制换页，可以在 buildReportHTML 里加上这个 class） */
          .page-break {
            page-break-before: always;
            break-before: always;
          }
        }
      </style>
    </head>
    <body>
      <div class="report-container">
        ${reportHTML}
      </div>
    </body>
    </html>
  `)
  doc.close()

  // Step 5: 等待 iframe 内容加载完成并唤起打印
  onProgress?.('Ready! Please save as PDF in the dialog.')

  // 稍微延迟确保 DOM 树和 base64 图片完全挂载到 iframe 中
  setTimeout(() => {
    try {
      iframe.contentWindow.focus()
      iframe.contentWindow.print()
    } catch (e) {
      console.error('Print failed:', e)
    } finally {
      // 打印对话框关闭后（无论用户是保存还是取消），清理 iframe
      setTimeout(() => {
        if (document.body.contains(iframe)) {
          document.body.removeChild(iframe)
        }
        onProgress?.('Done!')
      }, 1000)
    }
  }, 500) // 500ms 延迟通常足够让本地内容就绪
}