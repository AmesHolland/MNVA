<script setup>
import { ref, watch, onMounted, onUnmounted, shallowRef } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  chartType: String, // 标识当前需要渲染什么图，如 'global_map', 'sankey', 'relation_network'
  chartData: [Array, Object] // 后端传来的具体数据
})

const chartRef = ref(null)
let chartInstance = shallowRef(null)
let resizeObserver = null

// 核心渲染逻辑路由
const renderChart = async () => {
  if (!chartInstance.value || !props.chartData || props.chartData.length === 0) return

  chartInstance.value.showLoading()

  let option = {}

  // ==========================================
  // 路由 1: Global Monitor 的全局态势地图
  // ==========================================
  if (props.chartType === 'global_map') {
    // 1. 动态加载世界地图 GeoJSON
    if (!echarts.getMap('world')) {
      try {
        const res = await fetch('https://cdn.jsdelivr.net/npm/echarts@4.9.0/map/json/world.json')
        const worldJson = await res.json()
        echarts.registerMap('world', worldJson)
      } catch (e) {
        console.error('地图数据加载失败', e)
      }
    }

    // 2. 将后端的 GeoPoint 转换为 ECharts 需要的 [经度, 纬度, 烈度, 主题, 摘要]
    const scatterData = props.chartData.map(item => ({
      name: item.topic_name,
      value: [
        item.lon,
        item.lat,
        item.intensity,
        item.topic_name,
        item.summary
      ]
    }))

    // 3. 配置 ECharts Option
    option = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        formatter: (params) => {
          const [lon, lat, intensity, topic, summary] = params.value
          return `
            <div style="font-weight:bold; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-bottom: 5px;">
              ${topic} (烈度: ${intensity})
            </div>
            <div style="max-width: 200px; white-space: normal;">${summary}</div>
            <div style="font-size: 12px; color: #666; margin-top: 5px;">📍 [${lat.toFixed(2)}, ${lon.toFixed(2)}]</div>
          `
        }
      },
      geo: {
        map: 'world',
        roam: true, // 允许鼠标缩放和平移
        zoom: 1.2,
        itemStyle: {
          areaColor: '#e4e7ed', // 陆地颜色
          borderColor: '#ffffff', // 国界线颜色
          borderWidth: 1
        },
        emphasis: { itemStyle: { areaColor: '#d3dce6' } }
      },
      series: [
        {
          name: '热点事件',
          type: 'effectScatter', // 涟漪特效散点图
          coordinateSystem: 'geo',
          data: scatterData,
          symbolSize: (val) => {
            // 根据 intensity 动态计算气泡大小，防止过大或过小
            return Math.max(8, Math.min(val[2] * 2, 30))
          },
          showEffectOn: 'render',
          rippleEffect: { brushType: 'stroke', scale: 3 },
          itemStyle: {
            color: '#F56C6C', // 危险/高亮红色
            shadowBlur: 10,
            shadowColor: '#F56C6C'
          }
        }
      ]
    }
  }
  // 预留其他图表的坑位...
  // ==========================================
  // 路由 2.1: Deep Dive 的雷达画像图
  // ==========================================
  else if (props.chartType === 'radar') {
    // chartData 格式: {"military": 4.5, "diplomatic": 3.0, "media": 2.5}
    const dataObj = props.chartData || { military: 0, diplomatic: 0, media: 0 }

    option = {
      tooltip: { trigger: 'item' },
      radar: {
        indicator: [
          { name: '军事行动 (Military)', max: 5 },
          { name: '外交施压 (Diplomatic)', max: 5 },
          { name: '舆论造势 (Media)', max: 5 }
        ],
        radius: '65%', // 控制雷达图大小
        axisName: { color: '#606266', fontWeight: 'bold' },
        splitArea: { areaStyle: { color: ['#f4f4f5', '#ebeef5', '#e4e7ed', '#d3dce6'] } }
      },
      series: [{
        name: '行为烈度画像',
        type: 'radar',
        data: [{
          value: [dataObj.military, dataObj.diplomatic, dataObj.media],
          name: '烈度评分',
          areaStyle: { color: 'rgba(64, 158, 255, 0.4)' },
          lineStyle: { color: '#409EFF', width: 2 },
          itemStyle: { color: '#409EFF' }
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

    // 映射不同的行为类型到不同颜色 (颜色字典)
    const colorMap = {
      'Patrol': '#409EFF', // 蓝色-巡逻
      'Drill': '#E6A23C',  // 橙色-演习
      'Conflict': '#F56C6C',// 红色-冲突
      'Statement': '#909399',// 灰色-声明
      'Visit': '#67C23A'   // 绿色-访问
    }

    const scatterData = props.chartData.map(item => ({
      name: item.name,
      value: [item.lon, item.lat, item.type, item.summary, item.date],
      itemStyle: { color: colorMap[item.type] || '#409EFF' }
    }))

    option = {
      tooltip: {
        trigger: 'item',
        formatter: (p) => {
          const [lon, lat, type, summary, date] = p.value
          return `<b>${p.name}</b><br/>时间: ${date}<br/>行为: <span style="color:${colorMap[type]}">${type}</span><br/>摘要: ${summary}`
        }
      },
      geo: {
        map: 'world', roam: true, zoom: 3, // 微观地图可以默认放大一点
        center: scatterData.length > 0 ? [scatterData[0].value[0], scatterData[0].value[1]] : null, // 视角居中到第一个事件
        itemStyle: { areaColor: '#e4e7ed', borderColor: '#ffffff' }
      },
      series: [{
        name: '具体行动',
        type: 'effectScatter',
        coordinateSystem: 'geo',
        data: scatterData,
        symbolSize: 12,
        showEffectOn: 'emphasis', // 鼠标悬停时才显示波纹，保持界面清爽
        rippleEffect: { scale: 4 }
      }]
    }
  }
  // ==========================================
  // 路由 3.1: Relation Miner 的力导向关系网
  // ==========================================
  else if (props.chartType === 'relation_graph') {
    const { nodes, links } = props.chartData

    // 预设关系类型颜色字典
    const relationColorMap = {
      'Conflict': '#F56C6C',   // 冲突-红色
      'Cooperation': '#67C23A',// 合作-绿色
      'Diplomacy': '#E6A23C',  // 外交-橙色
      'Trade': '#409EFF',      // 贸易-蓝色
      'Other': '#909399'       // 其他-灰色
    }

    // 处理节点样式
    const graphNodes = nodes.map(n => ({
      name: n.id,
      symbolSize: 40, // 节点基础大小
      itemStyle: { color: '#2a5caa', borderColor: '#fff', borderWidth: 2 },
      label: { show: true, position: 'bottom', color: '#303133', fontWeight: 'bold' }
    }))

    // 处理连线样式
    const graphLinks = links.map(l => ({
      source: l.source,
      target: l.target,
      value: l.value,
      label: { show: true, formatter: l.label, fontSize: 10, color: '#666' },
      lineStyle: {
        color: relationColorMap[l.type] || '#909399',
        width: Math.min(Math.max(l.value, 1), 5), // 根据权重控制粗细(1~5px)
        curveness: 0.2 // 增加曲率，防止双向关系重叠
      },
      tooltip: l.tooltip // 绑定悬停详情
    }))

    option = {
      tooltip: {
        formatter: (p) => {
          if (p.dataType === 'edge') {
            // 连线 tooltip：支持换行展示多条细节
            return `<div style="max-width:250px; white-space:pre-wrap;"><b>${p.data.source} ➔ ${p.data.target}</b><br/><br/>${p.data.tooltip.replace(/\n/g, '<br/>')}</div>`
          }
          return p.name // 节点 tooltip
        }
      },
      series: [{
        type: 'graph',
        layout: 'force',
        data: graphNodes,
        links: graphLinks,
        roam: true, // 开启鼠标缩放和漫游
        edgeSymbol: ['none', 'arrow'], // 线条两端，终点为箭头
        edgeSymbolSize: [4, 10],
        force: {
          repulsion: 800, // 节点间的排斥力，越大越散开
          edgeLength: [100, 200], // 连线的长度范围
          gravity: 0.1 // 节点受到的向心力
        }
      }]
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

  // 渲染并隐藏 Loading
  chartInstance.value.setOption(option, true)
  chartInstance.value.hideLoading()
}

// 初始化图表实例与监听器
onMounted(() => {
  chartInstance.value = echarts.init(chartRef.value)

  // 监听容器大小变化，自动重绘图表尺寸
  resizeObserver = new ResizeObserver(() => {
    chartInstance.value?.resize()
  })
  resizeObserver.observe(chartRef.value)

  renderChart()
})

// 监听后端数据的更新（当 LLM 重新生成时，自动触发重绘）
watch(() => props.chartData, () => {
  renderChart()
}, { deep: true })

onUnmounted(() => {
  if (resizeObserver) resizeObserver.disconnect()
  chartInstance.value?.dispose()
})
</script>

<template>
  <div class="echarts-container" ref="chartRef"></div>
</template>

<style scoped>
.echarts-container {
  width: 100%;
  height: 100%;
  min-height: 300px;
}
</style>