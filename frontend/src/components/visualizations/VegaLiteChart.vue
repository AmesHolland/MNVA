<script setup>
import { ref, watch, onMounted, shallowRef } from 'vue'
import vegaEmbed from 'vega-embed'

const props = defineProps({
  chartType: String, // 如 'theme_river'
  chartData: Array   // 后端传来的 TimePoint 数组
})

const chartRef = ref(null)

const renderChart = async () => {
  if (!chartRef.value || !props.chartData || props.chartData.length === 0) return

  let spec = {}

  // ==========================================
  // 路由 1: Global Monitor 的演化河流图
  // ==========================================
  if (props.chartType === 'theme_river') {
    spec = {
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      description: "主题演化流图 (Theme River)",
      data: { values: props.chartData },
      width: "container", // 自适应容器宽度
      height: "container",
      mark: { type: "area", interpolate: "monotone", tooltip: true },
      encoding: {
        x: {
          field: "date",
          type: "temporal",
          axis: { title: "时间演进", format: "%m-%d", labelAngle: -45, grid: false }
        },
        y: {
          field: "count",
          type: "quantitative",
          stack: "center", // 这是形成 ThemeRiver (河流图) 的关键配置！
          axis: null // 河流图通常隐藏 Y 轴
        },
        color: {
          field: "topic_name",
          type: "nominal",
          scale: { scheme: "category10" },
          legend: { title: "热点主题", orient: "top" }
        }
      }
    }
  }
  // ==========================================
  // 路由 2: Deep Dive 的散点时间轴 (替代甘特图)
  // ==========================================
  else if (props.chartType === 'scatter_timeline') {
    spec = {
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      description: "实体行为散点时间轴",
      data: { values: props.chartData },
      width: "container",
      height: "container",
      mark: { type: "circle", opacity: 0.8, stroke: "black", strokeWidth: 0.5 },
      encoding: {
        x: {
          field: "x",
          type: "temporal",
          axis: { title: "时间", format: "%m-%d", grid: true }
        },
        y: {
          field: "y",
          type: "nominal",
          axis: { title: "行为分类", grid: true }
        },
        size: {
          field: "color", // 你后端的 color 字段实际上是烈度数值
          type: "quantitative",
          scale: { range: [50, 400] }, // 控制气泡大小范围
          legend: { title: "烈度" }
        },
        color: {
          field: "y",
          type: "nominal",
          legend: null // 颜色与 Y 轴分类绑定，无需额外图例
        },
        tooltip: [
          {field: "x", type: "temporal", title: "日期", format: "%Y-%m-%d"},
          {field: "y", type: "nominal", title: "分类"},
          {field: "tooltip", type: "nominal", title: "摘要"}
        ]
      }
    }
  }

  // 调用 vega-embed 渲染图表
  try {
    await vegaEmbed(chartRef.value, spec, { actions: false, theme: 'vox' })
  } catch (err) {
    console.error("Vega-Lite 渲染失败:", err)
  }
}

onMounted(() => {
  renderChart()
})

watch(() => props.chartData, () => {
  renderChart()
}, { deep: true })
</script>

<template>
  <div class="vega-container" ref="chartRef"></div>
</template>

<style scoped>
.vega-container {
  width: 100%;
  height: 100%;
  min-height: 250px;
}
</style>