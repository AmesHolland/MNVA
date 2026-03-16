<script setup>
import { ref, watch, onMounted, onUnmounted, shallowRef } from 'vue'
import * as echarts from 'echarts'
import { useChatStore } from '../../store/chatStore'

// 定义响应式的视图模式变量，默认为 'timeline'
const viewMode = ref('all');

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

/**
 * 将原始的扁平化海洋新闻态势数据转换为 ECharts 山脊图 Option
 * @param {Array} rawData - 原始数据
 * @returns {Object} ECharts Option
 */
function buildRidgelineOption(rawData) {
  if (!rawData || rawData.length === 0) return {};

  const dates = [...new Set(rawData.map(item => item.date))].sort();
  const topics = [...new Set(rawData.map(item => item.topic_name))];

  // --- 修改 1：在 dataMap 中不仅存 count，还要把 source_ids 存下来 ---
  const dataMap = {};
  topics.forEach(topic => { dataMap[topic] = {}; });

  rawData.forEach(item => {
    dataMap[item.topic_name][item.date] = {
      count: item.count || 1,
      source_ids: item.source_ids || [] // 🌟 把原始新闻的 ID 数组带过来
    };
  });

  // 🌟 关键参数：山峰的间距。
  // 注意：如果你的新闻篇数普遍较小（比如 1~5 篇），这个偏移量千万别设太大（如 15），否则山峰之间够不着，就不会有重叠效果。
  // 建议将此值设置为你数据平均最大值的 1/2 或 1/3，山峰就会漂亮地交错在一起。
  const OVERLAP_OFFSET = 1;
  const colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc'];

  const series = [];

  topics.forEach((topic, index) => {
    // 越往后的主题，基准线越高
    const baseOffset = index * OVERLAP_OFFSET;
    // const topicData = dates.map(date => dataMap[topic][date] || 0);

    // 1. 构造【隐形基座】系列
    series.push({
      name: `${topic}_base`,
      type: 'line',
      stack: `group_${topic}`,
      data: dates.map((_, i) => {
        if (i === 0) {
          return {
            value: baseOffset,
            // 🌟 【修复核心】：强行给第一个点一个圆点作为文字锚点，并使其完全透明！
            symbol: 'circle',
            symbolSize: 1,
            itemStyle: { color: 'transparent' },

            label: {
              show: true,
              position: [10, -20], // 这里的偏移量你可以根据实际视觉效果微调
              formatter: topic,
              color: '#333',
              fontSize: 11,
              fontWeight: 'bold',
              backgroundColor: 'rgba(255, 255, 255, 0.7)',
              padding: [2, 4],
              borderRadius: 3
            }
          };
        }
        return baseOffset;
      }),
      // 其他点依然保持无图形，保证性能
      symbol: 'none',
      lineStyle: { width: 0 },
      tooltip: { show: false },
      animation: false
    });
    // 2. 构造【真实山峰】系列
    const topicData = dates.map(date => {
      const info = dataMap[topic][date];
      if (info && info.count > 0) {
        return {
          value: info.count,
          source_ids: info.source_ids // 🌟 核心：将 ID 数组藏在数据对象里
        };
      }
      return 0; // 没有数据的日子补 0
    });

    series.push({
      name: topic,
      type: 'line',
      stack: `group_${topic}`,
      data: topicData,

      // 🌟 【交互核心】：替换掉原来的 symbol: 'none'
      showSymbol: false, // 平时隐藏折点，保持曲线顺滑
      symbol: 'circle',  // 鼠标 Hover 时，显示一个圆点作为点击热区！
      symbolSize: 8,     // Hover 时圆点的大小

      smooth: true,
      z: topics.length - index,
      lineStyle: { color: '#fff', width: 1.5 },
      areaStyle: {
        color: colors[index % colors.length],
        opacity: 0.85
      }
    });
  });

  return {
    tooltip: {
      trigger: 'axis',
      formatter: function (params) {
        // 过滤掉我们用于垫高的 _base 系列，且过滤掉没有真实数据的点
        const validParams = params.filter(p => !p.seriesName.endsWith('_base') && (p.value > 0 || p.data?.value > 0));
        if (validParams.length === 0) return '';

        let html = `<div style="font-weight:bold;margin-bottom:5px;">${params[0].axisValue}</div>`;
        let totalSources = 0; // 统计这个时间点所有主题的新闻总数

        [...validParams].reverse().forEach(param => {
           // 处理可能携带对象的 data
           const actualValue = param.data?.value !== undefined ? param.data.value : param.value;
           const sourceIds = param.data?.source_ids || [];
           totalSources += sourceIds.length;

           html += `${param.marker} ${param.seriesName}: <b>${actualValue}</b> 篇<br/>`;
        });

        // 🌟 如果存在源数据，给予醒目的点击引导
        if (totalSources > 0) {
          html += `<div style="margin-top:8px; padding-top:8px; border-top:1px dashed #ebeef5; color:#409EFF; font-size:12px; font-weight:bold;">
                     Click to see the Source News
                   </div>`;
        }
        return html;
      }
    },
    grid: { top: 30, bottom: 30, left: 0, right: 0 },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: dates,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: '#999' }
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: { show: false },
      min: 'dataMin'
    },
    series: series
  };
}

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
          <div style="white-space: normal; line-height: 1.4;">${summary}</div>
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
              type: ['rect', 'polygon', 'keep','clear'], // 允许矩形、多边形选框和清除按钮
              // title: { rect: '矩形圈选', polygon: '自由圈选', clear: '清除圈选' }
            }
          },
          iconStyle: { borderColor: '#409EFF' } // 契合你的 UI 主色调
        },

        // 【新增 2】：配置刷选行为，绑定到 geo 坐标系
        brush: {
          geoIndex: 'all', // 告诉刷选组件它在针对地理坐标系操作
          brushLink: 'all',
          brushMode: 'multiple', // 【核心破局点】：开启多选模式！
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
    const colorMap = { 'Technology': '#409EFF',
      'Politics': '#E6A23C',
      'Military': '#F56C6C',
      'Governance': '#909399',
      'Environment': '#67C23A',
      'Resources': '#eacc69'}

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
          let tip = `<b>${p.name}</b><br/>Time: ${date}<br/>Action: <span style="color:${colorMap[type]}">${type}</span><br/>Summary: ${summary}`
          if (source_ids && source_ids.length > 0) {
            tip += `<div style="margin-top:8px; padding-top:8px; border-top:1px dashed #ebeef5; color:#409EFF; font-size:12px; font-weight:bold;">👆Click the scatter point to view ${source_ids.length} pieces of original intelligence</div>`
          }
          return tip
        }
      },
      geo: { map: 'world', roam: true, zoom: 3, center: originalSeriesData.length > 0 ? [originalSeriesData[0].value[0], originalSeriesData[0].value[1]] : null, itemStyle: { areaColor: '#e4e7ed', borderColor: '#ffffff' } },
      series: [{ name: 'Action', type: 'effectScatter', coordinateSystem: 'geo', data: originalSeriesData, symbolSize: 12, showEffectOn: 'emphasis', rippleEffect: { scale: 4 } }]
    }
  }
  // ==========================================
  // 3.1 Relation Miner: Force Graph
  // ==========================================
  else if (props.chartType === 'relation_graph') {
  const { nodes, links } = props.chartData
  const relationColorMap = { 'Conflict': '#F56C6C', 'Cooperation': '#67C23A', 'Diplomacy': '#E6A23C', 'Trade': '#409EFF', 'Other': '#909399' }

  const graphNodes = nodes.map(n => ({
    name: n.id,
    symbolSize: 40,
    itemStyle: { color: '#2a5caa', borderColor: '#fff', borderWidth: 2, shadowBlur: 10, shadowColor: 'rgba(42, 92, 170, 0.3)' }, // 加一点微弱的高级阴影
    label: { show: true, position: 'bottom', color: '#303133', fontWeight: 'bold' }
  }))

  // 【修改 1】：为连线文字加上白色的“胶囊背景”，防止与线条或其他文字重叠
  originalSeriesData = links.map(l => ({
    source: l.source,
    target: l.target,
    value: l.value,
    source_ids: l.source_ids || [],
    active_dates: l.active_dates || [],
    label: {
      show: true,
      formatter: l.label,
      fontSize: 11,
      color: '#555',
      backgroundColor: 'rgba(255, 255, 255, 0.85)', // 🌟 核心：半透明白色背景
      padding: [3, 6],                              // 🌟 核心：给文字一点呼吸空间
      borderRadius: 4,                              // 🌟 核心：圆角变成小胶囊
      borderWidth: 1,
      borderColor: relationColorMap[l.type] || '#ebeef5' // 边框颜色和连线颜色一致
    },
    lineStyle: {
      color: relationColorMap[l.type] || '#909399',
      width: Math.min(Math.max(l.value, 1), 5),
      curveness: 0.2,
      opacity: 0.7
    },
    tooltip: l.tooltip // 保存原始长文本
  }))

  option = {
    // 【修改 2】：彻底重构 Tooltip，限制最大宽度，并允许内部滚动
    tooltip: {
      enterable: true, // 🌟 核心：允许鼠标进入 tooltip 内部（这样才能滚动！）
      confine: true,   // 🌟 核心：防止 tooltip 超出 ECharts 容器边缘
      extraCssText: 'max-width: 320px; white-space: normal; word-break: break-word; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border-radius: 8px; padding: 12px;',
      formatter: (p) => {
        if (p.dataType === 'edge') {
          let tip = `
            <div style="font-size: 14px; font-weight: bold; color: #303133; border-bottom: 1px solid #ebeef5; padding-bottom: 8px; margin-bottom: 8px;">
              🔗 ${p.data.source} ➔ ${p.data.target}
            </div>
            <div style="font-size: 13px; color: #606266; line-height: 1.6; max-height: 180px; overflow-y: auto; padding-right: 4px;">
              ${p.data.tooltip.replace(/\n/g, '<br/>')}
            </div>
          `;
          if (p.data.source_ids && p.data.source_ids.length > 0) {
            tip += `
              <div style="margin-top: 10px; padding-top: 10px; border-top: 1px dashed #ebeef5; color: #409EFF; font-size: 12px; font-weight: bold; display: flex; align-items: center;">
                👆 Click the edge to view ${p.data.source_ids.length} pieces of evidence
              </div>
            `;
          }
          return tip;
        }
        // 节点的简易提示
        return `<div style="font-weight:bold;">📍 ${p.name}</div>`;
      }
    },
    series: [{
      type: 'graph',
      layout: 'force',
      data: graphNodes,
      links: originalSeriesData,
      roam: true, // 允许整个画布缩放和平移

      // 🌟 核心新增：允许单个节点被拖拽！
      draggable: true,
      edgeSymbol: ['none', 'arrow'],
      edgeSymbolSize: [4, 10],
      force: {
        repulsion: 1200, // 🌟 修改：把排斥力从 800 调大到 1200，让节点之间离得更远，给文字留出空间
        edgeLength: [150, 250], // 🌟 修改：把最小连线长度拉长
        gravity: 0.1
      }
    }]
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
  else if (props.chartType === 'ridgeline_plot') {
    // 调用我们刚写的函数
    option = buildRidgelineOption(props.chartData);
  }
  // 假设你把 globalData 和 deepDiveData 都传进来了
  else if (props.chartType === 'unified_map') {
  if (!echarts.getMap('world')) {
      const res = await fetch('https://cdn.jsdelivr.net/npm/echarts@4.9.0/map/json/world.json')
      echarts.registerMap('world', await res.json())
  }

  const globalData = props.chartData.globalData || [];
  const deepDiveData = props.chartData.deepDiveData || [];

  // 1. 整理全局底噪数据 (Context)
  const globalSeriesData = globalData.map(item => ({
    name: item.topic_name,
    value: [item.lon, item.lat, item.intensity, item.topic_name, item.summary, item.source_ids || [], item.date],
    // 【关键】：作为背景，颜色要弱化，透明度调低
    itemStyle: { color: '#5a94ec', opacity: 0.5, shadowBlur: 0 }
  }));

  // 2. 整理微观轨迹数据 (Focus)
  const colorMap = { 'Technology': '#409EFF', 'Politics': '#E6A23C', 'Military': '#F56C6C', 'Governance': '#909399', 'Environment': '#67C23A', 'Resources': '#eacc69'};
  const deepDiveSeriesData = deepDiveData.map(item => ({
    name: item.name,
    value: [item.lon, item.lat, item.type, item.summary, item.date, item.source_ids || []],
    // 【关键】：作为焦点，颜色要极其鲜艳，且带发光和涟漪特效
    itemStyle: { color: colorMap[item.type] || '#409EFF', opacity: 1, shadowBlur: 10 }
  }));

  // 3. （可选超神操作）将微观实体的点连成轨迹线
  const trajectoryLines = [];
  for (let i = 0; i < deepDiveSeriesData.length - 1; i++) {
    trajectoryLines.push({
      coords: [
        [deepDiveSeriesData[i].value[0], deepDiveSeriesData[i].value[1]],
        [deepDiveSeriesData[i+1].value[0], deepDiveSeriesData[i+1].value[1]]
      ]
    });
  }

  option = {
    // 【核心交互】：开启 ECharts 原生图例，用户点击图例就能隐藏/显示对应图层！
    legend: {
      show: true,
      top: 10,
      left: 'center',
      data: ['Macro Context (Global)', 'Micro Trajectory (Entity)'],
      textStyle: { fontWeight: 'bold' }
    },
    tooltip: { /* 根据 params.seriesName 做不同的 tooltip 渲染即可 */ },
    // 【新增 1】：添加右上方工具栏，开启矩形和多边形选框
        toolbox: {
          show: true,
          left: 'left', // 水平位置：可选 'left'/'center'/'right'、像素值（如 20）、百分比（如 '10%'）
          top: 'top',    // 垂直位置：可选 'top'/'middle'/'bottom'、像素值（如 20）、百分比（如 '10%'）
          feature: {
            brush: {
              type: ['rect', 'polygon', 'keep','clear'], // 允许矩形、多边形选框和清除按钮
              // title: { rect: '矩形圈选', polygon: '自由圈选', clear: '清除圈选' }
            }
          },
          iconStyle: { borderColor: '#409EFF' } // 契合你的 UI 主色调
        },

        // 【新增 2】：配置刷选行为，绑定到 geo 坐标系
        brush: {
          geoIndex: 'all', // 告诉刷选组件它在针对地理坐标系操作
          brushLink: 'all',
          brushMode: 'multiple', // 【核心破局点】：开启多选模式！
          outOfBrush: {
            colorAlpha: 0.2 // 框选区域外的数据点变暗（自带 Dimming 效果！）
          },
          brushStyle: {
            borderWidth: 1,
            color: 'rgba(64,158,255,0.2)', // 选框内部的颜色
            borderColor: '#409EFF'
          }
        },
    geo: {
      map: 'world', roam: true, zoom: 1.5,
      itemStyle: { areaColor: '#e4e7ed', borderColor: '#ffffff' }
    },
    series: [
      {
        name: 'Macro Context (Global)', // 对应图例名字
        type: 'scatter', // 背景用普通散点即可，不要用 effectScatter 抢戏
        coordinateSystem: 'geo',
        data: globalSeriesData,
        symbolSize: (val) => Math.max(5, Math.min(val[2] * 2, 15))
      },
      {
        name: 'Micro Trajectory (Entity)', // 对应图例名字
        type: 'effectScatter', // 焦点实体用涟漪散点
        coordinateSystem: 'geo',
        data: deepDiveSeriesData,
        symbolSize: 12,
        showEffectOn: 'render',
        rippleEffect: { scale: 4, brushType: 'stroke' },
        zlevel: 2 // 确保微观点永远覆盖在宏观点上面
      },
      {
        name: 'Micro Trajectory (Entity)', // 名字一样，和散点共享同一个图例开关
        type: 'lines',
        coordinateSystem: 'geo',
        data: trajectoryLines,
        lineStyle: { color: '#F56C6C', width: 2, type: 'dashed', opacity: 0.6 },
        zlevel: 1
      }
    ]
  };
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
    } else {
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
      // 清除 store 中的空间标签
      store.updateBrushState(store.brushState.timeRange, [])
      return
    }

    // 获取第一个选框的数据
    const currentArea = areas[0]
    let coordinates = []

    // 场景 A：矩形选框
    if (currentArea.brushType === 'rect') {
      const [[x1, x2], [y1, y2]] = currentArea.range
      // 采样 5 个点：4个角 + 中心点
      const points = [
        [x1, y2], // 左下
        [x2, y2], // 右下
        [x2, y1], // 右上
        [x1, y1], // 左上
        [(x1+x2)/2, (y1+y2)/2] // 中心
      ]

      coordinates = points.map(p => {
        const coord = chartInstance.value.convertFromPixel({ geoIndex: 0 }, p)
        return [Number(coord[0].toFixed(2)), Number(coord[1].toFixed(2))]
      })
    }
    // 场景 B：多边形选框
    else if (currentArea.brushType === 'polygon') {
      const pixelPoints = currentArea.range
      // 采样：如果点太多，每隔几个取一个
      const step = Math.ceil(pixelPoints.length / 10)
      const sampledPixels = pixelPoints.filter((_, i) => i % step === 0)

      coordinates = sampledPixels.map(pixel => {
        const coord = chartInstance.value.convertFromPixel({ geoIndex: 0 }, pixel)
        return [Number(coord[0].toFixed(2)), Number(coord[1].toFixed(2))]
      })
    }

    console.log('采样坐标:', coordinates)

    // 调用 Store 的 Action 进行反解
    if (coordinates.length > 0) {
       store.resolveMapBrush(coordinates)
    }
  })
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
  z-index: 1;
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
  background-color: #375fc9;
  color: #ffffff;
  font-weight: bold;
}
</style>