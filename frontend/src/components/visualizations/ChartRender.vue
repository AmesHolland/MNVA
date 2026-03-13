<script setup>
import EchartsCanvas from './EchartsCanvas.vue'
import VegaLiteChart from './VegaLiteChart.vue'
// import TraceableText from "./TraceableText.vue";

defineProps({ task: Object ,
hideMap: Object})
</script>

<template>

  <template v-if="task.agent_name === 'Global_Monitor_Agent'">
    <div class="chart-box" v-if="!hideMap">
      <div class="chart-header"> Macro Spatiotemporal Map</div>
      <div class="chart-content">
        <EchartsCanvas chartType="global_map" :chartData="task.visualization_data.geo_dynamic_data" />
      </div>
    </div>



    <div class="chart-box" >
      <div class="chart-header"> Topic Evolution Ridgeline</div>
      <div class="chart-content">
        <EchartsCanvas chartType="ridgeline_plot" :chartData="task.visualization_data.ridgeline_data" />
      </div>
    </div>
  </template>

  <template v-if="task.agent_name === 'Deep_Dive_Agent'">
<!--    <div class="chart-box">-->
<!--      <div class="chart-header"><span class="icon">📍</span> Entity Trajectory Map</div>-->
<!--      <div class="chart-content">-->
<!--        <EchartsCanvas chartType="deep_dive_map" :chartData="task.visualization_data.map_chart" />-->
<!--      </div>-->
<!--    </div>-->

    <div class="chart-box">
      <div class="chart-header">Domain Behavior Timeline</div>
      <div class="chart-content" >
        <VegaLiteChart chartType="gantt_chart" :chartData="task.visualization_data.gantt_chart" />
      </div>
    </div>
  </template>

  <template v-if="task.agent_name === 'Relation_Miner_Agent'">
    <div class="chart-box">
      <div class="chart-header">Interaction Network Graph</div>
      <div class="chart-content">
        <EchartsCanvas chartType="relation_graph" :chartData="task.visualization_data.graph_chart" />
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