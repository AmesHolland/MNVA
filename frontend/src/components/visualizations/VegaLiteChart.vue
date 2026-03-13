<script setup>
import { ref, watch, onMounted, shallowRef } from 'vue'
import vegaEmbed from 'vega-embed'
import { useChatStore } from '../../store/chatStore' // 【新增】引入 Store

const props = defineProps({
  chartType: String,
  chartData: Array
})

const store = useChatStore() // 【新增】初始化 Store
const chartRef = ref(null)

/**
 * 预处理河流图数据：增加前后 0 点并补齐缺失主题数据
 * @param {Array} data 原始数据数组
 * @returns {Array} 处理后的平滑数据
 */
function preprocessThemeRiverData(data) {
  if (!data || data.length === 0) return [];

  // 1. 提取所有唯一的主题名
  const allTopics = [...new Set(data.map(d => d.topic_name))];

  // 2. 提取所有日期并排序
  const sortedDates = [...new Set(data.map(d => d.date))].sort();

  // 计算前一天和后一天的日期字符串 (YYYY-MM-DD)
  const firstDate = new Date(sortedDates[0]);
  const lastDate = new Date(sortedDates[sortedDates.length - 1]);

  const prevDate = new Date(firstDate.getTime() - 24 * 60 * 60 * 1000).toISOString().split('T')[0];
  const nextDate = new Date(lastDate.getTime() + 24 * 60 * 60 * 1000).toISOString().split('T')[0];

  // 3. 构建完整的时间轴
  const fullTimeline = [prevDate, ...sortedDates, nextDate];

  const processedData = [];

  // 4. 遍历时间轴，确保每个日期每个主题都有值
  fullTimeline.forEach(date => {
    allTopics.forEach(topic => {
      // 在原数据中查找
      const found = data.find(d => d.date === date && d.topic_name === topic);

      if (found) {
        processedData.push({ ...found });
      } else {
        // 如果没找到（包括我们新增的前后日期），补 0
        processedData.push({
          date: date,
          topic_name: topic,
          count: 0,
          source_ids: [] // 保证字段完整性，避免 Hint 计算报错
        });
      }
    });
  });

  return processedData;
}

// 在你的代码中这样调用：
// const cleanData = preprocessThemeRiverData(props.chartData);

const renderChart = async () => {
  if (!chartRef.value || !props.chartData || props.chartData.length === 0) return

  let spec = {}

  // ==========================================
  // 1. Ridgeline Plot (峰峦图) - Replaces ThemeRiver
  // ==========================================
  if (props.chartType === 'ridgeline_plot') {
    spec = {
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      data: { values: props.chartData },
      width: "container", // 让宽度自适应父容器
      // height: 200, // 移除固定总高度，改用 step 控制行高
      // 【修改点 1】：告诉 Vega-Lite 高度也跟随容器
      height: "container",
      // 【修改点 2】：强制图表缩放以适应外部容器尺寸
      autosize: { type: "fit", contains: "padding" },
      title: null, // 移除标题，节省空间
      mark: {
        type: "area",
        interpolate: "monotone",
        fillOpacity: 0.8,
        stroke: "white",
        strokeWidth: 0.5
      },
      encoding: {
        x: {
          field: "date",
          type: "temporal",
          axis: { title: null, format: "%m-%d", grid: false, tickCount: 5 } // 减少刻度数量，避免拥挤
        },
        y: {
          field: "count",
          type: "quantitative",
          axis: null,
          scale: { range: [25, 0] } // 【关键】：控制每个波峰的最大高度 (像素)，反转范围以正确向上生长
        },
        row: {
          field: "topic_name",
          type: "nominal",
          header: {
            title: null,
            labelAngle: 0,
            labelAlign: "left",
            labelPadding: 2,
            labelFontSize: 11,
            labelFontWeight: "bold"
          },
          // spacing: -18 // 【关键】：负间距实现堆叠效果，数值绝对值越接近 range[0] 堆叠越紧密
        },
        color: {
          field: "topic_name",
          type: "nominal",
          scale: { scheme: "tableau10" },
          legend: null
        },
        tooltip: [
          { field: "date", type: "temporal", title: "Date", format: "%Y-%m-%d" },
          { field: "topic_name", type: "nominal", title: "Topic" },
          { field: "count", type: "quantitative", title: "Article Count" }
        ]
      },
      config: {
        view: { stroke: null }, // 移除边框
        axis: { domain: false }, // 移除轴线
        // 【修改点 3】：覆盖默认的分面 step 行为
        facet: { spacing: 0 }
      }
    }
  }
  // ==========================================
  // 2. Scatter Timeline 散点分类时间轴
  // ==========================================
  // 3. 真正的 Gantt Chart (甘特图 - 事件持续时间跨度)
  // ==========================================
  else if (props.chartType === 'gantt_chart') {
    spec = {
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      data: { values: props.chartData },
      width: "container",
      height: "container",
      // 【修改】：使用 bar 类型绘制甘特图的横条，增加圆角让视觉更现代
      mark: { type: "bar", cornerRadius: 4, cursor: "pointer", stroke: "#fff", strokeWidth: 1 },
      transform: [
        { calculate: "datum.source_ids && length(datum.source_ids) > 0 ? '👆 Click the bar to view ' + length(datum.source_ids) + ' pieces of intelligence' : 'No direct sources'", as: "hint" },
        // 【Core Visual Fix】: For single-day events (start == end), add 24 hours (86400000 milliseconds) to the end time for visibility on the Gantt chart
        { calculate: "datum.start === datum.end ? datum.start + 86400000 : datum.end", as: "render_end" }
      ],
      encoding: {
        // Y-axis: Action Category
        y: {
          field: "category",
          type: "nominal",
          axis: { title: "Action Category", grid: true, tickBand: "extent" },
          sort: "-x" // Sort Y-axis categories by time
        },
        // X-axis start point
        x: {
          field: "start",
          type: "temporal",
          axis: { title: "Timeline Span", format: "%Y-%m-%d", grid: true, gridDash: [4, 4] }
        },
        // X-axis end point (using our fixed render_end)
        x2: { field: "render_end" },

        // Color mapping: Map color depth based on military intensity (red color system)
        color: {
          field: "intensity",
          type: "quantitative",
          scale: { scheme: "reds", domain: [0, 5] },
          legend: { title: "Politics Intensity (1-5)" }
        },
        // Interactive highlight
        opacity: {
          condition: { param: "hover", empty: false, value: 1 },
          value: 0.7
        },
        tooltip: [
          { field: "summary", type: "nominal", title: "Event Summary" },
          { field: "start", type: "temporal", title: "Start Date", format: "%Y-%m-%d" },
          { field: "end", type: "temporal", title: "End Date", format: "%Y-%m-%d" },
          { field: "intensity", type: "quantitative", title: "Military Intensity" },
          { field: "hint", type: "nominal", title: "Intelligence Source Tracing" }
        ]
      },
      params: [{
        name: "hover",
        select: { type: "point", on: "mouseover", clear: "mouseout" }
      }]
    }
  }

  // ==========================================
  // 【最核心】：渲染并挂载 Vega 点击事件！
  // ==========================================
  try {
    const result = await vegaEmbed(chartRef.value, spec, { actions: false, theme: 'vox' })

    // 监听图表内部图元（marks）的点击事件
    result.view.addEventListener('click', (event, item) => {
      // 确保点击的是有效数据点，且包含了 source_ids
      if (item && item.datum && item.datum.source_ids && item.datum.source_ids.length > 0) {
        store.openEvidence(item.datum.source_ids);
      }
    });
  } catch (err) {
    console.error("Vega-Lite Rendering failed: ", err)
  }
}

onMounted(() => renderChart())
watch(() => props.chartData, () => renderChart(), { deep: true })
</script>

<template><div class="vega-container" ref="chartRef"></div></template>
<style scoped>.vega-container { width: 100%; height: 100%; min-height: 250px; }</style>