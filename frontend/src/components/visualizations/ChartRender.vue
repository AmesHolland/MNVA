<script setup>
import EchartsCanvas from './EchartsCanvas.vue'
import VegaLiteChart from './VegaLiteChart.vue'
import { ref, nextTick, computed } from 'vue'

defineProps({ task: Object, hideMap: Object })

// 全屏
const isFullscreen = ref(false)
const toggleFullscreen = () => {
  isFullscreen.value = !isFullscreen.value
  nextTick(() => {
    window.dispatchEvent(new Event('resize'))
  })
}

// 右侧详情面板
const selectedEdge = ref(null)

const relationColorMap = {
  'Conflict': '#F56C6C',
  'Cooperation': '#67C23A',
  'Diplomacy': '#E6A23C',
  'Trade': '#409EFF',
  'Other': '#909399'
}

const parsedEvents = computed(() => {
  if (!selectedEdge.value?.tooltip) return []
  return selectedEdge.value.tooltip
    .split('\n')
    .filter(Boolean)
    .map(line => {
      const match = line.match(/\[(\d{4}-\d{2}-\d{2})\](.+)/)
      return match
        ? { date: match[1], text: match[2].trim() }
        : { date: '', text: line }
    })
})

const onEdgeClick = (edgeData) => {
  selectedEdge.value = edgeData
}
</script>

<template>

  <template v-if="task.agent_name === 'Global_Monitor_Agent'">
    <div class="chart-box" v-if="!hideMap">
      <div class="chart-header"> Macro Spatiotemporal Map</div>
      <div class="chart-content">
        <EchartsCanvas chartType="global_map" :chartData="task.visualization_data.geo_dynamic_data" />
      </div>
    </div>



<!--    <div class="chart-box" >-->
<!--      <div class="chart-header"> Topic Evolution Ridgeline</div>-->
<!--      <div class="chart-content">-->
<!--        <VegaLiteChart chartType="theme_river" :chartData="task.visualization_data.ridgeline_data" />-->
<!--      </div>-->
<!--    </div>-->
  </template>

  <template v-if="task.agent_name === 'Deep_Dive_Agent'">
<!--    <div class="chart-box">-->
<!--      <div class="chart-header"><span class="icon">📍</span> Entity Trajectory Map</div>-->
<!--      <div class="chart-content">-->
<!--        <EchartsCanvas chartType="deep_dive_map" :chartData="task.visualization_data.map_chart" />-->
<!--      </div>-->
<!--    </div>-->

<!--    <div class="chart-box">-->
<!--      <div class="chart-header">Domain Behavior Timeline</div>-->
<!--      <div class="chart-content" >-->
<!--        <VegaLiteChart chartType="gantt_chart" :chartData="task.visualization_data.gantt_chart" />-->
<!--      </div>-->
<!--    </div>-->
  </template>

  <template v-if="task.agent_name === 'Relation_Miner_Agent'">
  <div class="chart-box" :class="{ 'is-fullscreen': isFullscreen }">
    <div class="chart-header">
      <span>Interaction Network Graph</span>
      <button class="fullscreen-btn" @click="toggleFullscreen">
        {{ isFullscreen ? '⛌' : '⛶' }}
      </button>
    </div>

    <div class="chart-content">
      <EchartsCanvas
        :key="isFullscreen"
        chartType="relation_graph"
        :chartData="task.visualization_data.graph_chart"
        @edge-click="onEdgeClick"
      />

      <transition name="slide">
        <div v-if="selectedEdge" class="detail-panel">
          <div class="panel-header">
            <span>{{ selectedEdge.source }} → {{ selectedEdge.target }}</span>
            <button @click="selectedEdge = null">✕</button>
          </div>
          <div class="panel-type-badge"
            :style="{ background: relationColorMap[selectedEdge._type] }">
            {{ selectedEdge._type }}
          </div>
          <div class="event-list">
            <div v-for="(event, i) in parsedEvents" :key="i" class="event-item">
              <span class="event-date">{{ event.date }}</span>
              <span class="event-text">{{ event.text }}</span>
            </div>
          </div>
        </div>
      </transition>
    </div>
  </div>
</template>

</template>

<style scoped>
/* 这里只保留单张卡片的样式，排版布局交给父组件 Dashboard */
.chart-box {
  background: #ffffff;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  height: 500px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.03);
  transition: box-shadow 0.3s ease, transform 0.3s ease;
}

.chart-box:hover {
  box-shadow: 0 6px 16px rgba(0,0,0,0.08);
  transform: translateY(-2px);
}

.chart-header {
  font-size: 0.95rem; font-weight: 600; color: #303133;
  padding-bottom: 12px; margin-bottom: 12px;
  border-bottom: 1px dashed #ebeef5;
  display: flex; align-items: center;
}
.chart-header .icon { margin-right: 8px; font-size: 1.1rem; }
.chart-content { flex: 1; position: relative; width: 100%; }
/* 修改 Header 布局，让标题和按钮左右两端对齐 */
.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

/* 按钮样式 */
.fullscreen-btn {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 1.1rem;
  color: #909399;
  transition: all 0.2s;
  padding: 0 4px;
  line-height: 1;
}

.fullscreen-btn:hover {
  color: #409EFF;
  transform: scale(1.1);
}

/* 🌟 核心：全屏状态的 CSS 覆写 */
.chart-box.is-fullscreen {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: 3000; /* 确保层级高于抽屉和其他遮罩 */
  margin: 0;
  border-radius: 0;
  background-color: #f5f7fa; /* 全屏后的底色，根据你的系统主题调整 */
  display: flex;
  flex-direction: column;
}

/* 全屏状态下，让图表内容区撑满剩余的垂直空间 */
.chart-box.is-fullscreen .chart-content {
  flex: 1;
  height: 100%;
}

.chart-box.is-fullscreen .chart-header {
  padding: 15px 20px;
  background-color: #fff;
  border-bottom: 1px solid #ebeef5;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
<style scoped>
.detail-panel {
  width: 300px;
  border-left: 1px solid #ebeef5;
  padding: 16px;
  overflow-y: auto;
  background: #fafafa;
}
.panel-header {
  display: flex;
  justify-content: space-between;
  font-weight: bold;
  font-size: 14px;
  margin-bottom: 8px;
}
.panel-type-badge {
  display: inline-block;
  color: #fff;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 12px;
  margin-bottom: 12px;
}
.event-item {
  padding: 8px 0;
  border-bottom: 1px dashed #ebeef5;
  font-size: 13px;
  line-height: 1.6;
}
.event-date {
  color: #909399;
  font-size: 11px;
  display: block;
}
.legend-item {
  border: 2px solid;
  border-radius: 4px;
  padding: 1px 8px;
  font-size: 12px;
  margin-right: 6px;
}
.slide-enter-active, .slide-leave-active { transition: all 0.25s ease; }
.slide-enter-from, .slide-leave-to { transform: translateX(20px); opacity: 0; }
</style>


<!--<script setup>-->
<!--import EchartsCanvas from './EchartsCanvas.vue'-->
<!--import VegaLiteChart from './VegaLiteChart.vue'-->
<!--import TraceableText from "./TraceableText.vue";-->

<!--defineProps({ task: Object })-->
<!--</script>-->

<!--<template>-->
<!--  <div class="task-visual-container">-->
<!--&lt;!&ndash;    <div class="chart-section-title">&ndash;&gt;-->
<!--&lt;!&ndash;&lt;!&ndash;      📊 {{ task.agent_name.replace('_Agent', '') }}: {{ task.summary }}&ndash;&gt;&ndash;&gt;-->
<!--&lt;!&ndash;    <TraceableText :claims="task.summary" />&ndash;&gt;-->
<!--&lt;!&ndash;    </div>&ndash;&gt;-->

<!--    <template v-if="task.agent_name === 'Global_Monitor_Agent'">-->
<!--            <div class="grid-2col">-->
<!--&lt;!&ndash;        <div class="chart-box small"><EchartsCanvas chartType="radar" :chartData="task.visualization_data.radar_chart" /></div>&ndash;&gt;-->
<!--&lt;!&ndash;        <div class="chart-box small"><VegaLiteChart chartType="gantt_chart" :chartData="task.visualization_data.gantt_chart" /></div>&ndash;&gt;-->
<!--      <div class="chart-box medium" style="position: relative"><EchartsCanvas chartType="global_map" :chartData="task.visualization_data.geo_dynamic_data" /></div>-->
<!--      <div class="chart-box medium">-->
<!--        <EchartsCanvas chartType="ridgeline_plot" :chartData="task.visualization_data.ridgeline_data" />-->
<!--      </div>-->
<!--            </div>-->
<!--      &lt;!&ndash; Task 1: Replace ThemeRiver with Ridgeline Plot &ndash;&gt;-->

<!--    </template>-->

<!--    <template v-if="task.agent_name === 'Deep_Dive_Agent'">-->
<!--      <div class="chart-box large" style="position: relative"><EchartsCanvas chartType="deep_dive_map" :chartData="task.visualization_data.map_chart" /></div>-->
<!--      <div class="grid-2col">-->
<!--&lt;!&ndash;        <div class="chart-box small"><EchartsCanvas chartType="radar" :chartData="task.visualization_data.radar_chart" /></div>&ndash;&gt;-->
<!--        <div class="chart-box small"><VegaLiteChart chartType="gantt_chart" :chartData="task.visualization_data.gantt_chart" /></div>-->
<!--      </div>-->
<!--    </template>-->

<!--    <template v-if="task.agent_name === 'Relation_Miner_Agent'">-->

<!--      <div class="grid-2col-asymmetric">-->
<!--        <div class="chart-box large"><EchartsCanvas chartType="relation_graph" :chartData="task.visualization_data.graph_chart" /></div>-->
<!--&lt;!&ndash;        <div class="chart-box large"><EchartsCanvas chartType="sankey" :chartData="task.visualization_data.sankey_chart" /></div>&ndash;&gt;-->
<!--      </div>-->
<!--    </template>-->
<!--  </div>-->
<!--</template>-->

<!--<style scoped>-->
<!--.task-visual-container { margin-bottom: 5px; background: white; border: 1px solid #ebeef5; border-radius: 8px; padding: 20px; box-shadow: 0 2px 12px 0 rgba(0,0,0,0.02); }-->
<!--.chart-section-title { font-size: 1rem; color: #606266; margin-bottom: 15px; font-weight: 500; }-->
<!--.chart-box { background: #f9fafc; border-radius: 6px; margin-bottom: 15px; }-->
<!--.large { height: 450px; }-->
<!--.medium { height: 350px; }-->
<!--.small { height: 300px; }-->
<!--.grid-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }-->
<!--.grid-2col-asymmetric { display: grid; grid-template-columns: 6fr 4fr; gap: 15px; }-->
<!--</style>-->