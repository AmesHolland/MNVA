<script setup>
import { ref, watch, onMounted, onUnmounted, shallowRef } from 'vue'
import * as echarts from 'echarts'
import { useChatStore } from '../../store/chatStore' // 【新增】引入 Store

// 【新增】：定义响应式的视图模式变量，默认为 'timeline'
const viewMode = ref('timeline');



const props = defineProps({
  chartType: String,
  chartData: [Array, Object]
})

const store = useChatStore() // 【新增】初始化 Store
const chartRef = ref(null)
let chartInstance = shallowRef(null)
let resizeObserver = null

const renderChart = async () => {
  if (!chartInstance.value || !props.chartData || props.chartData.length === 0) return
  chartInstance.value.showLoading()
  let option = {}

  // ==========================================
  // 1. Global Monitor 全局地图
  // ==========================================
  if (props.chartType === 'global_map') {
  if (!echarts.getMap('world')) {
    const res = await fetch('https://cdn.jsdelivr.net/npm/echarts@4.9.0/map/json/world.json')
    echarts.registerMap('world', await res.json())
  }

  // 获取视图模式：'all' 表示展示所有热点，'timeline' 表示按时间播放
  const mode = viewMode.value;                  // 替换为这行
  const rawData = props.chartData;

  // --- 公共的 Tooltip 配置 ---
  const commonTooltip = {
    trigger: 'item',
    formatter: (params) => {
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

  // --- 公共的地图基础配置 ---
  const commonGeo = {
    map: 'world',
    roam: true,
    zoom: 1.2,
    itemStyle: { areaColor: '#e4e7ed', borderColor: '#ffffff' },
    emphasis: { itemStyle: { areaColor: '#d1d6e0' } }
  };

  // ==========================================
  // 模式 1: 全局聚合视图 (展示所有热点)
  // ==========================================
  if (mode === 'all') {
    const allScatterData = rawData.map(item => ({
      name: item.topic_name,
      value: [item.lon, item.lat, item.intensity, item.topic_name, item.summary, item.source_ids || [], item.date]
    }));

    option = {
      backgroundColor: 'transparent',
      tooltip: commonTooltip,
      geo: commonGeo,
      series: [{
        name: '全局热点分布',
        type: 'effectScatter',
        coordinateSystem: 'geo',
        data: allScatterData,
        // 根据烈度计算大小
        symbolSize: (val) => Math.max(8, Math.min(val[2] * 4, 30)),
        showEffectOn: 'emphasis', // 聚合模式下，为了避免满屏乱闪，可以改为鼠标 hover 时才泛起涟漪 ('emphasis')，或者保留 'render'
        rippleEffect: { brushType: 'stroke', scale: 3 },
        itemStyle: {
          color: '#F56C6C',
          shadowBlur: 5,
          shadowColor: '#F56C6C',
          opacity: 0.7 // 增加透明度，这样多天在同一个地点的点叠加在一起时，颜色会更深，形成类似热力图的效果
        }
      }]
    };
  }

  // ==========================================
  // 模式 2: 动态演化视图 (带时间轴)
  // ==========================================
  else if (mode === 'timeline') {
    const sortedRawData = [...rawData].sort((a, b) => new Date(a.date) - new Date(b.date));
    const uniqueDates = [...new Set(sortedRawData.map(item => item.date))];

    const timelineOptions = uniqueDates.map(currentDate => {
      const dayData = sortedRawData.filter(item => item.date === currentDate);
      return {
        title: { text: `Trend Evolution: ${currentDate}`, left: 'center', textStyle: { color: '#333', fontSize: 16 } },
        series: [{
          name: 'Hot Events',
          type: 'effectScatter',
          coordinateSystem: 'geo',
          data: dayData.map(item => ({
            name: item.topic_name,
            value: [item.lon, item.lat, item.intensity, item.topic_name, item.summary, item.source_ids || [], item.date]
          })),
          symbolSize: (val) => Math.max(8, Math.min(val[2] * 4, 30)),
          showEffectOn: 'render',
          rippleEffect: { brushType: 'stroke', scale: 3 },
          itemStyle: { color: '#F56C6C', shadowBlur: 10, shadowColor: '#F56C6C', opacity: 0.9 }
        }]
      };
    });

    option = {
      baseOption: {
        backgroundColor: 'transparent',
        timeline: {
          axisType: 'category',
          autoPlay: true,
          playInterval: 2000,
          data: uniqueDates,
          bottom: 10,
          label: { formatter: (s) => s.substring(5) },
          itemStyle: { color: '#004E52' },
          checkpointStyle: { color: '#F56C6C', borderColor: '#fff' }
        },
        tooltip: commonTooltip,
        geo: commonGeo
      },
      options: timelineOptions
    };
  }
}
  // 路由 2.1: Deep Dive 的雷达画像图
  // ==========================================
  else if (props.chartType === 'radar') {
    // chartData 格式: {"military": 4.5, "diplomatic": 3.0, "media": 2.5}
    const dataObj = props.chartData || { military: 0, diplomatic: 0, media: 0, Technology:0, Scientific:0 }

    option = {
      // 【新增】：添加一个内敛的标题，提升仪表盘的专业感
      title: {
        text: 'Multi-dimensional Behavioral Intensity Assessment',
        left: 'center',
        top: 0,
        textStyle: { fontSize: 14, color: '#606266', fontWeight: 'normal' }
      },
      tooltip: {
        trigger: 'item',
        // 【新增】：定制化 Tooltip，使其更易读
        formatter: (params) => {
          return `
            <div style="font-weight:bold; border-bottom:1px solid #ccc; padding-bottom:5px; margin-bottom:5px;">
              Comprehensive Intensity Score
            </div>
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
          { name: 'Military', max: 5 },
          { name: 'Diplomatic', max: 5 },
          { name: 'Media', max: 5 },
          { name: 'Technology', max: 5 },
          { name: 'Scientific Expedition', max: 5 }
        ],
        radius: '60%', // 稍微留出边缘空间给标题
        center: ['50%', '55%'],
        axisName: { color: '#303133', fontWeight: 'bold' },
        // 【优化】：将背景网格调整为渐变感，提升质感
        splitArea: {
          areaStyle: {
            color: ['rgba(245, 108, 108, 0.02)', 'rgba(245, 108, 108, 0.05)', 'rgba(245, 108, 108, 0.1)', 'rgba(245, 108, 108, 0.15)']
          }
        }
      },
      series: [{
        name: 'Behavioral Intensity Profile',
        type: 'radar',
        data: [{
          value: [dataObj.military, dataObj.diplomatic, dataObj.media],
          name: 'Intensity Score',
          // 【Optimization】: Use warning red color system to represent "intensity", which has more visual impact than blue
          areaStyle: { color: 'rgba(245, 108, 108, 0.5)' },
          lineStyle: { color: '#F56C6C', width: 2 },
          itemStyle: { color: '#F56C6C' }
        }]
      }]
    }
  }
  // ==========================================
  // 路由 2.2: Deep Dive 的微观轨迹地图
  // ==========================================
  else if (props.chartType === 'deep_dive_map') {
    if (!echarts.getMap('world')) {
      const res = await fetch('https://cdn.jsdelivr.net/npm/echarts@4.9.0/map/json/world.json')
      echarts.registerMap('world', await res.json())
    }
    const colorMap = { 'Patrol': '#409EFF', 'Drill': '#E6A23C', 'Conflict': '#F56C6C', 'Statement': '#909399', 'Visit': '#67C23A' }
    const scatterData = props.chartData.map(item => ({
      name: item.name,
      // 【修改】：将 source_ids 放在数组的第 6 位 (索引 5)
      value: [item.lon, item.lat, item.type, item.summary, item.date, item.source_ids || []],
      itemStyle: { color: colorMap[item.type] || '#409EFF' }
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
      geo: { map: 'world', roam: true, zoom: 3, center: scatterData.length > 0 ? [scatterData[0].value[0], scatterData[0].value[1]] : null, itemStyle: { areaColor: '#e4e7ed', borderColor: '#ffffff' } },
      series: [{ name: '具体行动', type: 'effectScatter', coordinateSystem: 'geo', data: scatterData, symbolSize: 12, showEffectOn: 'emphasis', rippleEffect: { scale: 4 } }]
    }
  }
  // ==========================================
  // 路由 3.1: Relation Miner 的力导向关系网
  // ==========================================
  else if (props.chartType === 'relation_graph') {
    const { nodes, links } = props.chartData
    const relationColorMap = { 'Conflict': '#F56C6C', 'Cooperation': '#67C23A', 'Diplomacy': '#E6A23C', 'Trade': '#409EFF', 'Other': '#909399' }
    const graphNodes = nodes.map(n => ({ name: n.id, symbolSize: 40, itemStyle: { color: '#2a5caa', borderColor: '#fff', borderWidth: 2 }, label: { show: true, position: 'bottom', color: '#303133', fontWeight: 'bold' } }))

    const graphLinks = links.map(l => ({
      source: l.source, target: l.target, value: l.value,
      source_ids: l.source_ids || [], // 【新增】：保留来源 ID
      label: { show: true, formatter: l.label, fontSize: 10, color: '#666' },
      lineStyle: { color: relationColorMap[l.type] || '#909399', width: Math.min(Math.max(l.value, 1), 5), curveness: 0.2 },
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
      series: [{ type: 'graph', layout: 'force', data: graphNodes, links: graphLinks, roam: true, edgeSymbol: ['none', 'arrow'], edgeSymbolSize: [4, 10], force: { repulsion: 800, edgeLength: [100, 200], gravity: 0.1 } }]
    }
  }
  // ==========================================
  // 路由 3.2: Relation Miner 的因果桑基图
  // ==========================================
  else if (props.chartType === 'sankey') {
    // 后端 sankey_chart 传过来的是 [{source, target, value}]
    const links = props.chartData

    // ECharts 的桑基图必须明确声明 nodes 数组，我们需要从 links 中动态提取去重后的实体
    const uniqueNodeNames = [...new Set(links.flatMap(l => [l.source, l.target]))]
    const sankeyNodes = uniqueNodeNames.map(name => ({ name }))

    option = {
      tooltip: { trigger: 'item', triggerOn: 'mousemove' },
      series: [{
        type: 'sankey',
        data: sankeyNodes,
        links: links,
        emphasis: { focus: 'adjacency' }, // 高亮当前相关的能量流
        nodeAlign: 'left', // 节点对齐方式
        lineStyle: {
          color: 'source', // 线条颜色采用源节点的颜色
          curveness: 0.5,
          opacity: 0.4
        },
        label: {
          color: '#303133',
          fontSize: 12,
          fontWeight: 'bold'
        },
        itemStyle: {
          borderWidth: 1,
          borderColor: '#aaa'
        }
      }]
    }
  }

  chartInstance.value.setOption(option, true)
  chartInstance.value.hideLoading()

  // ==========================================
  // 【最核心】：全局挂载 ECharts 点击事件！
  // ==========================================
  chartInstance.value.off('click') // 防止重复绑定
  chartInstance.value.on('click', (params) => {
    let sIds = []

    // 如果点击的是地图上的散点 (我们把 source_ids 存在了 value 数组的第 6 位)
    if (params.seriesType === 'effectScatter' || params.seriesType === 'scatter') {
      sIds = params.value[5]
    }
    // 如果点击的是力导向图的连线
    else if (params.dataType === 'edge') {
      sIds = params.data.source_ids
    }

    // 如果该数据点有来源 ID，唤起右侧抽屉
    if (sIds && sIds.length > 0) {
      store.openEvidence(sIds)
    }
  })
}

onMounted(() => {
  chartInstance.value = echarts.init(chartRef.value)
  resizeObserver = new ResizeObserver(() => chartInstance.value?.resize())
  resizeObserver.observe(chartRef.value)
  renderChart()
})

watch(() => props.chartData, () => renderChart(), { deep: true })

watch(viewMode, () => {
  if (props.chartType === 'global_map') {
    renderChart(); // 重新执行你上面的 ECharts Option 生成逻辑并 setOption
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
  <div class="echarts-container" ref="chartRef"></div></template>
<style scoped>
.echarts-container { width: 100%; height: 100%; min-height: 300px; cursor: pointer; }
/* 【新增】：控制组件的样式 */
.map-controls {
  position: absolute;
  top: 15px;
  right: 15px;
  z-index: 10; /* 确保悬浮在 ECharts 画布上方 */
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
  background-color: #409EFF; /* Element UI 的经典蓝色 */
  color: #ffffff;
  font-weight: bold;
}
</style>