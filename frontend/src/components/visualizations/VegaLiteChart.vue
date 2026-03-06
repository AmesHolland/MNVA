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
  // 1. Theme River 主题河流图
  // ==========================================
  if (props.chartType === 'theme_river') {
  // 假设 props.chartData 传入的是 trend_river_data 数组
  spec = {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    // 你的 preprocess 最好能确保日期格式为标准 ISO (YYYY-MM-DD)
    // data: { values: preprocessThemeRiverData(props.chartData) },
    data: { values: props.chartData },
    width: "container",
    height: "container",
    // 【VIS加分项】：增加一个选择器，使得鼠标悬停时高亮当前主题，其余变暗
    params: [{
      name: "hover",
      select: { type: "point", fields: ["topic_name"], on: "mouseover", clear: "mouseout" }
    }],
    mark: {
      type: "area",
      interpolate: "basis", // 'monotone' 比 'basis' 拟合更平滑，且不会越界
      tooltip: true,
      cursor: "pointer"
    },
    transform: [
      { calculate: "datum.source_ids && length(datum.source_ids) > 0 ? '👆 Click the ripple to view ' + length(datum.source_ids) + ' pieces of intelligence' : 'No direct sources'", as: "hint" }
    ],
    encoding: {
      x: {
        field: "date",
        type: "temporal",
        axis: { title: "Timeline Evolution", format: "%m-%d", grid: false }
      },
      y: {
        field: "count",
        type: "quantitative",
        stack: "center", // center 模式构成真正的河流图 (Streamgraph)
        axis: null,
        impute: { value: 0 } // 防止断层
      },
      color: {
        field: "topic_name",
        type: "nominal",
        scale: { scheme: "tableau10" }, // tableau10 的配色在学术论文中显得更专业高级
        legend: { orient: "top", title: null } // 将图例放到上方，节省横向空间
      },
      // 加入高亮交互逻辑
      opacity: {
        condition: { param: "hover", empty: false, value: 1 },
        value: 0.8
      },
      tooltip: [
        { field: "date", type: "temporal", title: "Occurrence Date", format: "%Y-%m-%d" },
        { field: "topic_name", type: "nominal", title: "Macro Topic" },
        { field: "count", type: "quantitative", title: "Number of Related News" },
        { field: "hint", type: "nominal", title: "Intelligence Source Tracing" }
      ]
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
          legend: { title: "Military Intensity (1-5)" }
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