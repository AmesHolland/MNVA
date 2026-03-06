<script setup>
import { ref, watch, onMounted, onUnmounted, shallowRef } from 'vue'
import * as echarts from 'echarts'
import { useChatStore } from '../../store/chatStore'

// 定义响应式的视图模式变量，默认为 'timeline'
const viewMode = ref('timeline');

const props = defineProps({
  chartType: String,
  chartData: [Array, Object]
})

const store = useChatStore()
const chartRef = ref(null)
let chartInstance = shallowRef(null)
let resizeObserver = null

// 【核心缓存】：用于存储渲染时最原始的 series data，方便后续刷选对比
let originalSeriesData = []

const renderChart = async () => {
  if (!chartInstance.value || !props.chartData || props.chartData.length === 0) return
  chartInstance.value.showLoading()
  let option = {}

  // 每次重新渲染前清空缓存，防止图表切换时数据污染
  originalSeriesData = []

  // ==========================================
  // 1. Global Monitor 全局地图
  // ==========================================
  if (props.chartType === 'global_map') {
    if (!echarts.getMap('world')) {
      const res = await fetch('https://cdn.jsdelivr.net/npm/echarts@4.9.0/map/json/world.json')
      echarts.registerMap('world', await res.json())
    }

    const mode = viewMode.value;
    const rawData = props.chartData;

    // --- 公共的 Tooltip 配置 ---
    const commonTooltip = {
      trigger: 'item',
      formatter: (params) => {
        // [lon, lat, intensity, topic, summary, source_ids, date]
        const [lon, lat, intensity, topic, summary, source_ids, date] = params.value;
        let tip = `
          <div style="font-weight:bold; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-bottom: 5px;">
            🗓️ ${date} | ${topic} (intensity: ${intensity})
          </div>
          <div style="max-width: 250px; white-space: normal; line-height: 1.4;">${summary}</div>
          <div style="font-size: 12px; color: #666; margin-top: 5px;">📍 [${lat.toFixed(2)}, ${lon.toFixed(2)}]</div>
        `;
        if (source_ids && source_ids.length > 0) {
          tip += `<div style="margin-top:8px; padding-top:8px; border-top:1px dashed #ebeef5; color:#409EFF; font-size:12px; font-weight:bold; cursor: pointer;">
                    👆Click the scatter point to view ${source_ids.length} pieces of original intelligence
                  </div>`;
        }
        return tip;
      }
    };

    const commonGeo = {
      map: 'world', roam: true, zoom: 1.2,
      itemStyle: { areaColor: '#e4e7ed', borderColor: '#ffffff' },
      emphasis: { itemStyle: { areaColor: '#d1d6e0' } }
    };

    if (mode === 'all') {
      // 【修改】：缓存原始数据供刷选使用
      originalSeriesData = rawData.map(item => ({
        name: item.topic_name,
        // 索引说明: 0:lon, 1:lat, 2:intensity, 3:topic, 4:summary, 5:source_ids, 6:date
        value: [item.lon, item.lat, item.intensity, item.topic_name, item.summary, item.source_ids || [], item.date],
        itemStyle: { color: '#F56C6C', shadowBlur: 5, shadowColor: '#F56C6C', opacity: 0.7 }
      }));

      option = {
        backgroundColor: 'transparent',
        tooltip: commonTooltip,
        geo: commonGeo,
        series: [{
          name: '全局热点分布', type: 'effectScatter', coordinateSystem: 'geo',
          data: originalSeriesData, // 使用缓存的原始数据
          symbolSize: (val) => Math.max(8, Math.min(val[2] * 4, 30)),
          showEffectOn: 'emphasis',
          rippleEffect: { brushType: 'stroke', scale: 3 }
        }],
        // 【新增 1】：添加右上方工具栏，开启矩形和多边形选框
        toolbox: {
          show: true,
          left: 'left', // 水平位置：可选 'left'/'center'/'right'、像素值（如 20）、百分比（如 '10%'）
          top: 'top',    // 垂直位置：可选 'top'/'middle'/'bottom'、像素值（如 20）、百分比（如 '10%'）
          feature: {
            brush: {
              type: ['rect', 'polygon', 'clear'] // 允许矩形、多边形选框和清除按钮
            }
          },
          iconStyle: { borderColor: '#409EFF' } // 契合你的 UI 主色调
        },

        // 【新增 2】：配置刷选行为，绑定到 geo 坐标系
        brush: {
          geoIndex: 'all', // 告诉刷选组件它在针对地理坐标系操作
          brushLink: 'all',
          outOfBrush: {
            colorAlpha: 0.2 // 框选区域外的数据点变暗（自带 Dimming 效果！）
          },
          brushStyle: {
            borderWidth: 1,
            color: 'rgba(64,158,255,0.2)', // 选框内部的颜色
            borderColor: '#409EFF'
          }
        },


      };
    }
    else if (mode === 'timeline') {
      // Timeline 模式自带时间过滤，为了避免与刷选冲突，此处不缓存 originalSeriesData
      const sortedRawData = [...rawData].sort((a, b) => new Date(a.date) - new Date(b.date));
      const uniqueDates = [...new Set(sortedRawData.map(item => item.date))];

      const timelineOptions = uniqueDates.map(currentDate => {
        const dayData = sortedRawData.filter(item => item.date === currentDate);
        return {
          title: { text: `Trend Evolution: ${currentDate}`, left: 'center', textStyle: { color: '#333', fontSize: 16 } },
          series: [{
            name: 'Hot Events', type: 'effectScatter', coordinateSystem: 'geo',
            data: dayData.map(item => ({
              name: item.topic_name,
              value: [item.lon, item.lat, item.intensity, item.topic_name, item.summary, item.source_ids || [], item.date]
            })),
            symbolSize: (val) => Math.max(8, Math.min(val[2] * 4, 30)),
            showEffectOn: 'render', rippleEffect: { brushType: 'stroke', scale: 3 },
            itemStyle: { color: '#F56C6C', shadowBlur: 10, shadowColor: '#F56C6C', opacity: 0.9 }
          }]
        };
      });

      option = {
        baseOption: {
          backgroundColor: 'transparent',
          timeline: {
            axisType: 'category', autoPlay: true, playInterval: 2000,
            data: uniqueDates, bottom: 10, label: { formatter: (s) => s.substring(5) },
            itemStyle: { color: '#004E52' }, checkpointStyle: { color: '#F56C6C', borderColor: '#fff' }
          },
          tooltip: commonTooltip, geo: commonGeo
        },
        options: timelineOptions
      };
    }
  }
  // ==========================================
  // 2.1 Deep Dive: Radar
  // ==========================================
  else if (props.chartType === 'radar') {
    const dataObj = props.chartData || { military: 0, diplomatic: 0, media: 0, Technology:0, Scientific:0 }
    option = {
      title: { text: 'Multi-dimensional Behavioral Intensity Assessment', left: 'center', top: 0, textStyle: { fontSize: 14, color: '#606266', fontWeight: 'normal' } },
      tooltip: {
        trigger: 'item',
        formatter: (params) => {
          return `
            <div style="font-weight:bold; border-bottom:1px solid #ccc; padding-bottom:5px; margin-bottom:5px;">Comprehensive Intensity Score</div>
            Military Operations: <span style="font-weight:bold;color:#F56C6C">${params.value[0].toFixed(1)}</span> / 5<br/>
            Diplomatic Pressure: <span style="font-weight:bold;color:#E6A23C">${params.value[1].toFixed(1)}</span> / 5<br/>
            Public Opinion Campaigns: <span style="font-weight:bold;color:#409EFF">${params.value[2].toFixed(1)}</span> / 5<br/>
            Technology: <span style="font-weight:bold;color:#F56C6C">${params.value[3].toFixed(1)}</span> / 5<br/>
            Scientific Expedition: <span style="font-weight:bold;color:#E6A23C">${params.value[4].toFixed(1)}</span> / 5<br/>
          `
        }
      },
      radar: {
        indicator: [
          { name: 'Military', max: 5 }, { name: 'Diplomatic', max: 5 }, { name: 'Media', max: 5 }, { name: 'Technology', max: 5 }, { name: 'Scientific Expedition', max: 5 }
        ],
        radius: '60%', center: ['50%', '55%'], axisName: { color: '#303133', fontWeight: 'bold' },
        splitArea: { areaStyle: { color: ['rgba(245, 108, 108, 0.02)', 'rgba(245, 108, 108, 0.05)', 'rgba(245, 108, 108, 0.1)', 'rgba(245, 108, 108, 0.15)'] } }
      },
      series: [{
        name: 'Behavioral Intensity Profile', type: 'radar',
        data: [{
          value: [dataObj.military, dataObj.diplomatic, dataObj.media, dataObj.Technology, dataObj.Scientific],
          name: 'Intensity Score', areaStyle: { color: 'rgba(245, 108, 108, 0.5)' },
          lineStyle: { color: '#F56C6C', width: 2 }, itemStyle: { color: '#F56C6C' }
        }]
      }],

    }
  }
  // ==========================================
  // 2.2 Deep Dive: Map
  // ==========================================
  else if (props.chartType === 'deep_dive_map') {
    if (!echarts.getMap('world')) {
      const res = await fetch('https://cdn.jsdelivr.net/npm/echarts@4.9.0/map/json/world.json')
      echarts.registerMap('world', await res.json())
    }
    const colorMap = { 'Patrol': '#409EFF', 'Drill': '#E6A23C', 'Conflict': '#F56C6C', 'Statement': '#909399', 'Visit': '#67C23A' }

    // 【修改】：缓存原始数据，固定 source_ids 为索引 5
    originalSeriesData = props.chartData.map(item => ({
      name: item.name,
      value: [item.lon, item.lat, item.type, item.summary, item.date, item.source_ids || []],
      itemStyle: { color: colorMap[item.type] || '#409EFF', opacity: 1 }
    }))

    option = {
      tooltip: {
        trigger: 'item',
        formatter: (p) => {
          const [lon, lat, type, summary, date, source_ids] = p.value
          let tip = `<b>${p.name}</b><br/>时间: ${date}<br/>行为: <span style="color:${colorMap[type]}">${type}</span><br/>摘要: ${summary}`
          if (source_ids && source_ids.length > 0) {
            tip += `<div style="margin-top:8px; padding-top:8px; border-top:1px dashed #ebeef5; color:#409EFF; font-size:12px; font-weight:bold;">👆Click the scatter point to view ${source_ids.length} pieces of original intelligence</div>`
          }
          return tip
        }
      },
      geo: { map: 'world', roam: true, zoom: 3, center: originalSeriesData.length > 0 ? [originalSeriesData[0].value[0], originalSeriesData[0].value[1]] : null, itemStyle: { areaColor: '#e4e7ed', borderColor: '#ffffff' } },
      series: [{ name: '具体行动', type: 'effectScatter', coordinateSystem: 'geo', data: originalSeriesData, symbolSize: 12, showEffectOn: 'emphasis', rippleEffect: { scale: 4 } }]
    }
  }
  // ==========================================
  // 3.1 Relation Miner: Force Graph
  // ==========================================
  else if (props.chartType === 'relation_graph') {
    const { nodes, links } = props.chartData
    const relationColorMap = { 'Conflict': '#F56C6C', 'Cooperation': '#67C23A', 'Diplomacy': '#E6A23C', 'Trade': '#409EFF', 'Other': '#909399' }
    const graphNodes = nodes.map(n => ({ name: n.id, symbolSize: 40, itemStyle: { color: '#2a5caa', borderColor: '#fff', borderWidth: 2 }, label: { show: true, position: 'bottom', color: '#303133', fontWeight: 'bold' } }))

    // 【修改】：缓存带有 active_dates 的连线数据
    originalSeriesData = links.map(l => ({
      source: l.source, target: l.target, value: l.value,
      source_ids: l.source_ids || [],
      active_dates: l.active_dates || [], // 刷选依赖此字段
      label: { show: true, formatter: l.label, fontSize: 10, color: '#666' },
      lineStyle: { color: relationColorMap[l.type] || '#909399', width: Math.min(Math.max(l.value, 1), 5), curveness: 0.2, opacity: 1 },
      tooltip: l.tooltip
    }))

    option = {
      tooltip: {
        formatter: (p) => {
          if (p.dataType === 'edge') {
            let tip = `<div style="max-width:250px; white-space:pre-wrap;"><b>${p.data.source} ➔ ${p.data.target}</b><br/><br/>${p.data.tooltip.replace(/\n/g, '<br/>')}</div>`
            if (p.data.source_ids && p.data.source_ids.length > 0) {
              tip += `<div style="margin-top:8px; padding-top:8px; border-top:1px dashed #ebeef5; color:#409EFF; font-size:12px; font-weight:bold;">👆 Click the connection line to view ${p.data.source_ids.length} pieces of original intelligence</div>`
            }
            return tip
          }
          return p.name
        }
      },
      series: [{ type: 'graph', layout: 'force', data: graphNodes, links: originalSeriesData, roam: true, edgeSymbol: ['none', 'arrow'], edgeSymbolSize: [4, 10], force: { repulsion: 800, edgeLength: [100, 200], gravity: 0.1 } }]
    }
  }
  // ==========================================
  // 3.2 Relation Miner: Sankey
  // ==========================================
  else if (props.chartType === 'sankey') {
    const links = props.chartData
    const uniqueNodeNames = [...new Set(links.flatMap(l => [l.source, l.target]))]
    const sankeyNodes = uniqueNodeNames.map(name => ({ name }))

    option = {
      tooltip: { trigger: 'item', triggerOn: 'mousemove' },
      series: [{
        type: 'sankey', data: sankeyNodes, links: links,
        emphasis: { focus: 'adjacency' }, nodeAlign: 'left',
        lineStyle: { color: 'source', curveness: 0.5, opacity: 0.4 },
        label: { color: '#303133', fontSize: 12, fontWeight: 'bold' },
        itemStyle: { borderWidth: 1, borderColor: '#aaa' }
      }]
    }
  }

  chartInstance.value.setOption(option, true)
  chartInstance.value.hideLoading()

  // 挂载全局点击事件
  chartInstance.value.off('click')
  chartInstance.value.on('click', (params) => {
    let sIds = []
    if (params.seriesType === 'effectScatter' || params.seriesType === 'scatter') {
      sIds = params.value[5] // 两张地图均已对齐 source_ids 存放在 index 5
    } else if (params.dataType === 'edge') {
      sIds = params.data.source_ids
    }
    if (sIds && sIds.length > 0) {
      store.openEvidence(sIds)
    }
  })

  // ==========================================
  // 【新增 3】：在底部挂载 brushEnd 事件监听
  // ==========================================
  chartInstance.value.off('brushEnd') // 防抖，避免重复绑定
  chartInstance.value.on('brushEnd', (params) => {
    const areas = params.areas
    if (!areas || areas.length === 0) {
      console.log('用户清除了选框')
      return
    }

    // 获取第一个选框的数据
    const currentArea = areas[0]

    // 场景 A：如果你想获取选框真实的【经纬度边界】(以矩形为例)
    if (currentArea.brushType === 'rect') {
      const [[x1, x2], [y1, y2]] = currentArea.range // 拿到屏幕像素的 X 和 Y 范围

      // 使用 convertFromPixel 将像素转为经纬度 [lon, lat]
      // 注意传入 geo 坐标系，x1, y2 对应左下角，x2, y1 对应右上角
      const bottomLeft = chartInstance.value.convertFromPixel({ geoIndex: 0 }, [x1, y2])
      const topRight = chartInstance.value.convertFromPixel({ geoIndex: 0 }, [x2, y1])

      console.log(`选区经纬度范围:
        左下角: 经度 ${bottomLeft[0].toFixed(2)}, 纬度 ${bottomLeft[1].toFixed(2)}
        右上角: 经度 ${topRight[0].toFixed(2)}, 纬度 ${topRight[1].toFixed(2)}
      `)
    }
    // === 处理多边形 (Polygon) 边界 ===
  if (currentArea.brushType === 'polygon') {
    // 拿到多边形所有顶点的像素坐标数组
    const pixelPoints = currentArea.range

    // 将每一个像素点转换为真实的 [经度, 纬度]
    const geoPolygon = pixelPoints.map(pixel => {
      // 传入 geoIndex: 0 表示使用第一个地理坐标系进行转换
      const coord = chartInstance.value.convertFromPixel({ geoIndex: 0 }, pixel)
      // 保留两位小数，使得数据更干净
      return [Number(coord[0].toFixed(2)), Number(coord[1].toFixed(2))]
    })

    console.log('框选的海洋多边形经纬度集合:', geoPolygon)
    // 输出结果示例: [[118.5, 23.2], [122.1, 24.5], [120.3, 21.8], ...]

    // 你可以将这个 geoPolygon 数组发送给后端，
    // 后端可以使用 Elasticsearch 的 geo_polygon 查询，或者 PostGIS 的 ST_Within 进行空间检索
  }
  //
  // // 可选：将数据输出到页面（比如展示在某个 div 中）
  // const outputDom = document.getElementById('brush-output');
  // if (outputDom) {
  //   outputDom.innerHTML = `
  //     <div>多边形顶点数：${currentArea.type === 'polygon' ? currentArea.area[0].path.length : '非多边形'}</div>
  //     <div>选中热点数量：${selectedData.length}</div>
  //     <div>选中数据：${JSON.stringify(selectedData, null, 2)}</div>
  //   `;
  // }

    // 场景 B：如果你更关心【哪些数据点(热点事件)】被框中了
    // params.areas 不包含具体数据，数据在 echarts 实例里，可以通过 getOption() 结合当前高亮状态获取
    // 或者最简单的，用 brushSelected 事件去抓 payload 里的 dataIndex
  })

  // 可选：监听 brushSelected 获取被框中的具体散点数据
  // chartInstance.value.off('brushSelected')
  // chartInstance.value.on('brushSelected', (params) => {
  //   const brushComponent = params.batch[0]
  //   if (brushComponent.selected.length === 0) return
  //
  //   // 这里的 dataIndex 就是被框中的点在 originalSeriesData 里的索引
  //   const selectedIndices = brushComponent.selected[0].dataIndex
  //   if (selectedIndices.length > 0) {
  //     const selectedData = selectedIndices.map(index => originalSeriesData[index])
  //     console.log('被框中的热点事件数据:', selectedData)
  //     // 这里可以触发 Store 更新，比如 store.updateSpatialFilter(selectedData)
  //   }
  // })
}

// ==========================================
// 【合并的核心魔术】：监听 Store 执行 Dimming
// ==========================================
watch(() => store.brushState, (newBrush) => {
  console.log("store.brushState change")
  if (!chartInstance.value || originalSeriesData.length === 0) return
  // 如果处于 Timeline 模式，跳过 Dimming，以免与 Timeline 的自动轮播数据冲突
  if (props.chartType === 'global_map' && viewMode.value === 'timeline') return

  const [brushStart, brushEnd] = newBrush.timeRange || [0, Infinity]
  console.log("newBrush.timeRange" + newBrush.timeRange)
  const activeLabels = newBrush.spatialLabels || []

  const matchesLabels = (textStr) => {
    if (activeLabels.length === 0) return true
    const text = textStr.toLowerCase()
    return activeLabels.some(label => text.includes(label.toLowerCase()))
  }

  // 1. 地图刷选更新
  if (props.chartType === 'global_map' || props.chartType === 'deep_dive_map') {
    const updatedData = originalSeriesData.map(item => {
      // 全局地图在 6，深挖地图在 4
      const dateStr = props.chartType === 'global_map' ? item.value[6] : item.value[4]
      const itemTime = new Date(dateStr).getTime()

      const isTimeMatch = itemTime >= brushStart && itemTime <= brushEnd
      const searchStr = item.name + " " + (item.value[3] || '') + " " + (item.value[4] || '')
      const isLabelMatch = matchesLabels(searchStr)

      const isHighlight = isTimeMatch //  && isLabelMatch

      return {
        ...item,
        itemStyle: isHighlight
          ? { ...item.itemStyle, opacity: (props.chartType === 'global_map' ? 0.7 : 1), shadowBlur: 10 }
          : { ...item.itemStyle, opacity: 0.15, shadowBlur: 0, color: '#cccccc' }
      }
    })
    chartInstance.value.setOption({ series: [{ data: updatedData }] })
  }

  // 2. 关系网连线刷选更新
  else if (props.chartType === 'relation_graph') {
    const updatedLinks = originalSeriesData.map(link => {
      const hasTimeMatch = link.active_dates.some(dateStr => {
        const t = new Date(dateStr).getTime()
        return t >= brushStart && t <= brushEnd
      })
      const isLabelMatch = matchesLabels(link.source + " " + link.target + " " + link.label.formatter)
      const isHighlight = hasTimeMatch && isLabelMatch

      return {
        ...link,
        lineStyle: isHighlight
          ? { ...link.lineStyle, opacity: 1 }
          : { ...link.lineStyle, opacity: 0.1, color: '#e4e7ed' }
      }
    })
    chartInstance.value.setOption({ series: [{ links: updatedLinks }] })
  }
}, { deep: true })

onMounted(() => {
  chartInstance.value = echarts.init(chartRef.value)
  resizeObserver = new ResizeObserver(() => chartInstance.value?.resize())
  resizeObserver.observe(chartRef.value)
  renderChart()
})

watch(() => props.chartData, () => renderChart(), { deep: true })

watch(viewMode, () => {
  if (props.chartType === 'global_map') {
    renderChart();
  }
});

onUnmounted(() => { if (resizeObserver) resizeObserver.disconnect(); chartInstance.value?.dispose() })
</script>

<template>
  <div class="map-controls" v-if="props.chartType === 'global_map' ">
    <div
      class="control-btn"
      :class="{ active: viewMode === 'all' }"
      @click="viewMode = 'all'"
    >
      🗺️ Global Overview
    </div>
    <div
      class="control-btn"
      :class="{ active: viewMode === 'timeline' }"
      @click="viewMode = 'timeline'"
    >
      ⏱️ Dynamic Evolution
    </div>
  </div>
  <div class="echarts-container" ref="chartRef"></div>
</template>

<style scoped>
.echarts-container { width: 100%; height: 100%; min-height: 300px; cursor: pointer; }
.map-controls {
  position: absolute;
  top: 15px;
  right: 15px;
  z-index: 10;
  display: flex;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 6px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  overflow: hidden;
  border: 1px solid #dcdfe6;
}

.control-btn {
  padding: 6px 12px;
  font-size: 13px;
  cursor: pointer;
  color: #606266;
  user-select: none;
  transition: background-color 0.3s, color 0.3s;
}

.control-btn:first-child {
  border-right: 1px solid #dcdfe6;
}

.control-btn:hover {
  background-color: #f5f7fa;
}

.control-btn.active {
  background-color: #409EFF;
  color: #ffffff;
  font-weight: bold;
}
</style>