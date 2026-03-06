<script setup>
import { ref, watch, onMounted, shallowRef } from 'vue'
import * as echarts from 'echarts'
import { useChatStore } from '../../store/chatStore'

const store = useChatStore()
const timelineRef = ref(null)
let chartInstance = shallowRef(null)
let resizeObserver = null

// --- 1. 模拟与整合标签数据 (Spatial & Semantic) ---
// 实际应用中，这里应替换为 store.analysisResults.data_profile 的真实数据
const spatialLabels = ref([
  { name: '辽宁', active: false }, { name: '山东', active: false },
  { name: '江苏', active: false }, { name: '上海', active: false },
  { name: '浙江', active: false }, { name: '福建', active: false },
  { name: '广东', active: false }, { name: '海南', active: false },
  { name: '香港', active: false }, { name: '澳门', active: false },
  { name: '南海', active: false, isFocus: true }, // 假设是 Anchor 的焦点
  { name: '太平洋', active: false }
])

const topicLabels = ref([
  { name: '海上风电', active: false }, { name: '联合军演', active: false, isFocus: true },
  { name: '渔业纠纷', active: false }, { name: '深海采矿', active: false }
])

// 标签点击事件 (触发软刷选)
const toggleLabel = (label) => {
  label.active = !label.active
  // TODO: 这里可以调用 store 方法，通知下方图表让非相关的散点变灰
}

// --- 2. Timeline 图表渲染 ---
const renderTimeline = () => {
  if (!chartInstance.value) return

  // 1. 生成 mock 背景数据（统一用时间戳）
  const mockBackgroundData = []
  let baseDate = new Date('2015-01-01').getTime()
  const oneDay = 24 * 3600 * 1000
  for (let i = 0; i < 365 * 11; i++) {
    const currentTime = baseDate + i * oneDay
    const volume = Math.max(10, Math.floor(Math.random() * 50) + (i % 100 === 0 ? 200 : 0))
    mockBackgroundData.push([currentTime, volume])
  }

  // 2. 处理 Anchor 蓝图数据
  const blueprint = store.analysisResults?.spatiotemporal_blueprint
  const markAreaRegions = []
  const phaseColors = ['rgba(103, 194, 58, 0.2)', 'rgba(230, 162, 60, 0.2)', 'rgba(245, 108, 108, 0.2)']

  if (blueprint && blueprint.phases) {
    blueprint.phases.forEach((phase, index) => {
      // 严格校验格式
      if (!phase.time_range || !phase.time_range.includes(' to ')) return;
      const [startStr, endStr] = phase.time_range.split(' to ');
      const startDate = new Date(startStr);
      const endDate = new Date(endStr);
      // 过滤无效日期
      if (isNaN(startDate.getTime()) || isNaN(endDate.getTime())) return;

      markAreaRegions.push({
        name: `Phase ${phase.phase_id}: ${phase.phase_name}`,
        itemStyle: { color: phaseColors[index % phaseColors.length] },
        label: { position: 'insideTop', color: '#606266', fontWeight: 'bold' },
        xAxis: [startDate.getTime(), endDate.getTime()] // 时间戳格式
      })
    })
  }

  // 3. ECharts 配置
  const option = {
    grid: { top: 30, right: 20, bottom: 40, left: 40 },
    tooltip: { trigger: 'axis', position: 'top' },
    xAxis: {
      type: 'time',
      boundaryGap: false,
      axisLine: { lineStyle: { color: '#e4e7ed' } },
      axisLabel: { color: '#909399' },
      splitLine: { show: false }
    },
    yAxis: {
      type: 'value',
      show: false,
      max: 'dataMax'
    },
    dataZoom: [
      {
        type: 'slider',
        show: true,
        height: 20,
        bottom: 5,
        startValue: new Date('2025-08-01').getTime(), // 时间戳
        endValue: new Date('2025-12-31').getTime(),   // 时间戳
        borderColor: 'transparent',
        backgroundColor: '#f5f7fa',
        fillerColor: 'rgba(64, 158, 255, 0.2)',
        handleStyle: { color: '#409EFF' }
      },
      { type: 'inside' }
    ],
    series: [
      {
        name: '历史新闻底噪',
        type: 'bar',
        barWidth: 2,
        itemStyle: { color: '#dcdfe6' },
        data: mockBackgroundData,
        markArea: {
          silent: false,
          data: markAreaRegions
        }
      }
    ]
  }

  chartInstance.value.setOption(option, true)

  // 监听时间轴缩放事件
  chartInstance.value.on('datazoom', (params) => {
    // 业务逻辑...
  })
}

onMounted(() => {
  chartInstance.value = echarts.init(timelineRef.value)
  resizeObserver = new ResizeObserver(() => chartInstance.value?.resize())
  resizeObserver.observe(timelineRef.value)
  renderTimeline()
})

// 当蓝图更新时，重新渲染时间轴
watch(() => store.analysisResults?.spatiotemporal_blueprint, () => {
  renderTimeline()
}, { deep: true })
</script>

<template>
  <div class="spatiotemporal-navigator">

    <div class="labels-panel">

      <div class="label-group">
        <span class="group-title">📍 空间作用域 (Spatial):</span>
        <button
          v-for="item in spatialLabels" :key="item.name"
          :class="['label-btn', { active: item.active, 'is-focus': item.isFocus }]"
          @click="toggleLabel(item)"
        >
          <span v-if="item.isFocus" class="star">★</span>
          {{ item.name }}
        </button>
      </div>

      <div class="label-group">
        <span class="group-title">🏷️ 语义话题 (Topics):</span>
        <button
          v-for="item in topicLabels" :key="item.name"
          :class="['label-btn topic-btn', { active: item.active, 'is-focus': item.isFocus }]"
          @click="toggleLabel(item)"
        >
          {{ item.name }}
        </button>
      </div>

    </div>

    <div class="timeline-container" ref="timelineRef"></div>

  </div>
</template>

<style scoped>
.spatiotemporal-navigator {
  background: #ffffff;
  border: 1px solid #e4e7ed;
  border-radius: 8px 8px 0 0; /* 底部圆角取消 */
  box-shadow: 0 -2px 12px rgba(0,0,0,0.03); /* 阴影向上 */
  padding: 15px 20px;
  /* margin-bottom: 20px; 移除原有底部margin */
  display: flex;
  flex-direction: column;
  gap: 15px;
}

/* 标签面板样式 */
.labels-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.label-group {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.group-title {
  font-size: 0.85rem;
  color: #606266;
  font-weight: 500;
  margin-right: 5px;
}

.label-btn {
  background: #f4f4f5;
  border: 1px solid #e9e9eb;
  color: #606266;
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  gap: 4px;
}

.label-btn:hover { background: #eef5fe; color: #409EFF; }
.label-btn.active { background: #409EFF; color: #ffffff; border-color: #409EFF; }

/* 重点标识样式 (Anchor Focus) */
.label-btn.is-focus { border-color: #E6A23C; box-shadow: 0 0 4px rgba(230, 162, 60, 0.3); }
.star { color: #E6A23C; }
.label-btn.active .star { color: #FFD591; }

.topic-btn.active { background: #67C23A; border-color: #67C23A; }

/* Timeline 容器 */
.timeline-container {
  width: 100%;
  height: 120px; /* 控制时间轴的高度 */
}
</style>