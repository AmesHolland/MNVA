<script setup>
import EchartsCanvas from './EchartsCanvas.vue'
import VegaLiteChart from './VegaLiteChart.vue'
import TraceableText from "./TraceableText.vue";

defineProps({ task: Object })
</script>

<template>
  <div class="task-visual-container">
    <div class="chart-section-title">
<!--      📊 {{ task.agent_name.replace('_Agent', '') }}: {{ task.summary }}-->
    <TraceableText :claims="task.summary" />
    </div>

    <template v-if="task.agent_name === 'Global_Monitor_Agent'">
      <div class="chart-box large" style="position: relative"><EchartsCanvas chartType="global_map" :chartData="task.visualization_data.geo_dynamic_data" /></div>
      <div class="chart-box medium"><VegaLiteChart chartType="theme_river" :chartData="task.visualization_data.trend_river_data" /></div>
    </template>

    <template v-if="task.agent_name === 'Deep_Dive_Agent'">
      <div class="chart-box large" style="position: relative"><EchartsCanvas chartType="global_map" :chartData="task.visualization_data.map_chart" /></div>
      <div class="grid-2col">
        <div class="chart-box small"><EchartsCanvas chartType="radar" :chartData="task.visualization_data.radar_chart" /></div>
        <div class="chart-box small"><VegaLiteChart chartType="gantt_chart" :chartData="task.visualization_data.gantt_chart" /></div>
      </div>
    </template>

    <template v-if="task.agent_name === 'Relation_Miner_Agent'">
      <div class="grid-2col-asymmetric">
        <div class="chart-box large"><EchartsCanvas chartType="relation_graph" :chartData="task.visualization_data.graph_chart" /></div>
        <div class="chart-box large"><EchartsCanvas chartType="sankey" :chartData="task.visualization_data.sankey_chart" /></div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.task-visual-container { margin-bottom: 30px; background: white; border: 1px solid #ebeef5; border-radius: 8px; padding: 20px; box-shadow: 0 2px 12px 0 rgba(0,0,0,0.02); }
.chart-section-title { font-size: 1rem; color: #606266; margin-bottom: 15px; font-weight: 500; }
.chart-box { background: #f9fafc; border-radius: 6px; margin-bottom: 15px; }
.large { height: 450px; }
.medium { height: 350px; }
.small { height: 300px; }
.grid-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
.grid-2col-asymmetric { display: grid; grid-template-columns: 6fr 4fr; gap: 15px; }
</style>