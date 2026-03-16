<script setup>
import { computed, ref } from 'vue'
import { useChatStore } from '../store/chatStore'

// 引入左侧组件
import ChatBox from '../components/chat/ChatBox.vue'
// import ApprovalCard from '../components/chat/ApprovalCard.vue'
import ChatInput from '../components/chat/ChatInput.vue'
import TraceableText from '../components/visualizations/TraceableText.vue'
import EvidenceDrawer from '../components/visualizations/EvidenceDrawer.vue'
// 引入图表分发组件 (记得确保你有这个组件，或者把代码平铺进来)
import ChartRenderer from '../components/visualizations/ChartRender.vue'
import SpatiotemporalNavigator from "../components/visualizations/SpatiotemporalNavigator.vue";
import EchartsCanvas from "../components/visualizations/EchartsCanvas.vue";

const store = useChatStore()

// 核心状态：视图模式切换 ('split' 左右分栏模式 | 'narrative' 瀑布流模式)
const viewMode = ref('split')
// 在 Dashboard.vue 中
const unifiedMapData = computed(() => {
  let globalData = [];
  let deepDiveData = [];

  allUniqueTasks.value.forEach(task => {
    if (task.agent_name === 'Global_Monitor_Agent') {
      globalData = task.visualization_data.geo_dynamic_data || [];
    }
    if (task.agent_name === 'Deep_Dive_Agent') {
      deepDiveData = task.visualization_data.map_chart || [];
    }
  });

  // 如果两者都没有，返回 null 不渲染地图
  if (globalData.length === 0 && deepDiveData.length === 0) return null;

  return { globalData, deepDiveData };
});
// 辅助方法：图表去重引擎
// 无论哪种模式，我们都要确保图表不重复出现
const getUniqueTasks = (sections) => {
  const tasks = store.analysisResults?.tasks || {}
  const seen = new Set()
  const uniqueTasks = []

  // 防御性检查
  if (!sections || !Array.isArray(sections)) return uniqueTasks

  sections.forEach(sec => {
    // 兼容新旧字段名
    const claims = sec.claims || sec.content_claims || []

    claims.forEach(claim => {
      // 🌟 核心修改：直接从句子的溯源属性中提取任务 ID
      const taskId = claim.source_subtask

      // 如果这个任务存在、有效、且没被添加过
      if (taskId && !seen.has(taskId) && tasks[taskId]) {
        seen.add(taskId)
        uniqueTasks.push({ ...tasks[taskId], task_id: taskId })
      }
    })
  })

  return uniqueTasks
}

// 计算属性：提取全部去重后的 Task (用于 Split 模式的右侧)
const allUniqueTasks = computed(() => {
  if (!store.analysisResults?.report?.sections) return []
  return getUniqueTasks(store.analysisResults.report.sections)
})

// 【新增】：切换历史任务
const handleTaskSwitch = (task) => {
  store.switchTask(task)
}
const isHovering = ref(false)
let hoverTimer = null

// 增加防抖，避免鼠标不小心划过时频繁闪烁
const handleMouseEnter = () => {
  clearTimeout(hoverTimer)
  hoverTimer = setTimeout(() => {
    isHovering.value = true
  }, 200)
}

const handleMouseLeave = () => {
  clearTimeout(hoverTimer)
  hoverTimer = setTimeout(() => {
    isHovering.value = false
  }, 200)
}
</script>

<template>
  <div class="app-wrapper">

    <header class="global-top-bar">
      <div class="brand">

        <span class="sys-name">Marine News</span>
        <span class="sys-subname">Multi-Agent Visual Analytics</span>
      </div>

      <div class="global-actions" v-if="store.analysisResults?.report">

        <div class="task-history-panel" v-if="store.taskHistory.length > 0" @mouseenter="handleMouseEnter" @mouseleave="handleMouseLeave">
          <span class="history-label">History:</span>
          <div class="history-list">
            <button
              v-for="(task, idx) in store.taskHistory"
              :key="task.task_id"
              :class="['history-btn', { active: store.analysisResults.report.report_title === task.results.report.report_title }]"
              @click="handleTaskSwitch(task)"
            >
              {{ idx + 1 }}
              <span v-if="task.is_sandbox" class="sandbox-badge">🔍</span>
            </button>
          </div>

          <transition name="fade-slide">
            <div class="dag-hover-board" v-show="isHovering">
              <div class="dag-board-header">
                <h3>Analysis Trace</h3>
                <span class="sub-text">History Pipeline</span>
              </div>

              <div class="dag-timeline">
                <div
                  class="dag-tree"
                  v-for="(task, idx) in store.taskHistory"
                  :key="task.task_id"
                  :class="{ 'is-active': store.analysisResults.report.report_title === task.results.report.report_title }"
                  @click="handleTaskSwitch(task)"
                >
                  <div class="node goal-node">
                    <div class="node-header">
                      <span class="node-type">Goal #{{ idx + 1 }}</span>
                      <span v-if="task.is_sandbox" class="badge-sandbox">Sandbox</span>
                    </div>
                    <div class="node-content" :title="task.query">{{ task.query }}</div>
                    <div class="progress-bar red-bar"></div>
                  </div>

                  <div class="sub-nodes-container" v-if="task.results && task.results.tasks">
                    <div
                      class="node sub-node"
                      v-for="([key, subTask], index) in Object.entries(task.results.tasks)"
                      :key="key"
                    >
                      <div class="sub-node-left">
                        <!-- 用 index + 1 展示顺序号 -->
                        <span class="node-type">Task {{ index + 1 }}</span>
                      </div>
                      <div class="sub-node-right">
                        <span class="node-content">{{ subTask.agent_name.replace('_Agent', '') }}</span>
                        <span class="icon-chart">📊</span>
                      </div>
                      <div class="progress-bar blue-bar"></div>
                    </div>
                  </div>
                  <div class="history-connector" v-if="idx < store.taskHistory.length - 1"></div>
                </div>
              </div>
            </div>
          </transition>
        </div>

        <div class="view-toggle">
          <button :class="['toggle-btn', { active: viewMode === 'split' }]" @click="viewMode = 'split'" title="Split view">
            <span class="icon">◫</span> Dashboard
          </button>
          <button :class="['toggle-btn', { active: viewMode === 'narrative' }]" @click="viewMode = 'narrative'" title="Stream view">
            <span class="icon">📄</span> Narrative
          </button>
        </div>

      </div>
    </header>
    <div class="dashboard-container">

      <aside class="left-panel">
  <!--      <header class="panel-header">-->
  <!--        <h2>Marine News</h2>-->
  <!--        <span class="subtitle">Multi-Agent Visual Analytics</span>-->
  <!--      </header>-->
        <main class="chat-stream">
          <ChatBox />
  <!--        <ApprovalCard class="hitl-card" />-->
        </main>
        <footer class="input-area">
          <ChatInput />
        </footer>
      </aside>

      <section class="right-panel">

        <!-- 场景 A：正式报告已生成 -->
        <template v-if="store.analysisResults?.report">


          <div v-if="viewMode === 'split'" class="split-layout">

            <div class="split-text-pane">
  <!--            <div class="executive-summary-card">-->
  <!--              <strong>Abstract：</strong>{{ store.analysisResults.report.executive_summary }}-->
  <!--            </div>-->
            <article v-for="(section, index) in store.analysisResults.report.sections" :key="'text-'+index" class="text-section">
              <h2 class="section-subtitle">{{ index + 1 }}. {{ section.section_title }}</h2>

              <div class="section-content">
                 <TraceableText :claims="section.claims || section.content_claims" />
              </div>
            </article>
              <div class="conclusion-box">
                <h3>Conclusion</h3>
                <p>{{ store.analysisResults.report.conclusion }}</p>
              </div>
            </div>

<!--            <div class="split-chart-pane">-->
<!--              <template v-for="task in allUniqueTasks" :key="'chart-'+task.task_id">-->
<!--                <ChartRenderer :task="task" />-->
<!--              </template>-->
<!--            </div>-->
            <div class="split-chart-pane">

                <div class="chart-box" style="grid-column: 1 / -1; " v-if="unifiedMapData">
                  <div class="chart-header"> Spatiotemporal Unified Map</div>
                  <div class="chart-content">
                    <EchartsCanvas chartType="unified_map" :chartData="unifiedMapData" />
                  </div>
                </div>

                <template v-for="task in allUniqueTasks" :key="task.task_id">
                    <ChartRenderer :task="task" :hideMap="true" />
                </template>

          </div>

          </div>

          <div v-else-if="viewMode === 'narrative'" class="narrative-layout">
            <div class="report-container">

              <div class="executive-summary-card">
                <span class="quote-icon">❝</span>
                <p>{{ store.analysisResults.report.executive_summary }}</p>
              </div>

              <article
                v-for="(section, index) in store.analysisResults.report.sections"
                :key="'nar-'+index"
                class="doc-section"
              >
              <div class="section-text-block">
                  <h2 class="section-subtitle"><span class="section-index">{{ index + 1 }}</span>{{ section.subtitle }}</h2>
                  <div class="section-paragraphs">
                     <TraceableText :claims="section.content_claims" />
                  </div>
               </div>
                <div class="section-visual-block">
                  <template v-for="task in getUniqueTasks([section])" :key="'nar-chart-'+task.task_id">
                     <ChartRenderer :task="task" />
                  </template>
                </div>
              </article>

              <div class="doc-conclusion">
                <h3>Conclusion</h3>
                <p>{{ store.analysisResults.report.conclusion }}</p>
              </div>
            </div>
          </div>

        </template>

        <!-- 场景 B：初始状态或预览状态 (全屏地图) -->
        <template v-else>
          <div class="full-screen-map-container">
             <!-- 如果有预览数据，传给 EchartsCanvas；否则传空数组，EchartsCanvas 会渲染空底图 -->
             <EchartsCanvas
               chart-type="global_map"
               :chart-data="store.previewData || []"
             />

             <!-- 浮动提示，告诉用户这是预览模式 -->
             <div v-if="store.previewData" class="preview-badge">
               🔍 Data Profiling Preview
             </div>
          </div>
        </template>

  <!--      <SpatiotemporalNavigator class="spatiotemporal-float" />-->
        <div class="spatiotemporal-float">
          <transition name="slide-up">
            <div v-if="store.brushState.timeRange || store.brushState.spatialLabels?.length" class="sandbox-trigger-bar">
              <div class="sandbox-info">
                <span class="icon">🎯</span>
                <span>Local Spatiotemporal Region Locked：</span>

                <strong v-if="store.brushState.timeRange" class="highlight-time">
                  {{ store.formatDate(store.brushState.timeRange[0]) }} to
                  {{ store.formatDate(store.brushState.timeRange[1]) }}
                </strong>
                <strong v-else class="highlight-time">Global Time</strong>

                <span v-if="store.brushState.spatialLabels?.length" class="highlight-tags">
                   | Area: {{ store.brushState.spatialLabels.join(', ') }}
                </span>
              </div>

              <div class="sandbox-actions">
                <button class="cancel-btn" @click="store.clearBrushState" title="清除时空约束">✖</button>

                <button class="re-anchor-btn" @click="store.prepareSandboxAnalysis">
                  ✍️ Convert Selection into Analysis Prompt
                </button>
              </div>
            </div>
          </transition>

          <SpatiotemporalNavigator />
        </div>
      </section>

      <EvidenceDrawer />

    </div>
  </div>
</template>

<style scoped>
.app-wrapper {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  background-color: #f8fafc;
}

/* 统一细长顶栏 */
.global-top-bar {
  height: 48px;
  background-color: #1e3a8a; /* 深海蓝视觉锚点 */
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 24px;
  color: white;
  z-index: 20;
  box-shadow: 0 1px 4px rgba(0,0,0,0.1);
}

.brand { display: flex; align-items: baseline; gap: 8px; }
.logo-icon { font-size: 1.2rem; }
.sys-name { font-size: 1.1rem; font-weight: 600; letter-spacing: 0.5px; }
.sys-subname { font-size: 0.8rem; color: white; font-weight: 400; }

.global-actions { display: flex; align-items: center; gap: 24px; }

/* 保持原有的左侧样式 #93c5fd */

.dashboard-container { display: flex; height: 100vh; width: 100vw; background-color: #f8fafc; /* 极简冷灰底色 */ overflow: hidden; font-family: -apple-system, sans-serif; }
.left-panel { width: 380px; background-color: #ffffff; border-right: 1px solid #e2e8f0; display: flex; flex-direction: column; z-index: 10; }
.panel-header { padding: 20px; border-bottom: 1px solid #e2e8f0; background-color: #ffffff; }
.panel-header h2 { margin: 0; font-size: 1.2rem; color: #334155; }
.panel-header .subtitle { font-size: 0.8rem; color: #64748b; }
.chat-stream { flex: 1; padding: 15px; overflow-y: auto; background-color: #ffffff; } /* 统一白底，消除区块割裂感 */
.input-area { padding: 15px; background-color: #ffffff; border-top: 1px solid #e2e8f0; }

.right-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; background-color: #f8fafc; position: relative; padding-bottom: 180px; }
/* --- 右侧顶部工具栏 --- */
.top-toolbar {
  display: flex; justify-content: space-between; align-items: center;
  padding: 15px 30px; background-color: #1e3a8a; /* 深海蓝视觉锚点 */ border-bottom: none;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1); z-index: 5;
}
/* 报告大标题：因为移入了白底文本区，需要改成深色字体和学术排版 */
.report-title {
  font-size: 1.8rem;
  color: #0f172a; /* 极深灰/黑 */
  font-weight: 700;
  margin-top: 0;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 2px solid #e2e8f0;
  line-height: 1.3;
}
.centered-title { text-align: center; }

/* 切换按钮组 (适配深色背景) */
.view-toggle { display: flex; background: #193276; border-radius: 6px; padding: 4px; border: 1px solid #1e3a8a; }
.toggle-btn {
  border: none; background: transparent; padding: 6px 12px; border-radius: 4px;
  cursor: pointer; color: #94a3b8; font-size: 0.9rem; font-weight: 500; transition: all 0.2s;
}
.toggle-btn:hover { color: #f8fafc; }
.toggle-btn.active { background: #2c4c9b; color: #ffffff; box-shadow: none; }
/* 公共摘要卡片 (学术风) */
.executive-summary-card { background: #ffffff; padding: 15px 20px; border-radius: 6px; color: #475569; border: 1px solid #e2e8f0; border-left: 4px solid #3b82f6; margin-bottom: 20px; line-height: 1.6; font-size: 0.95rem; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }
/* ================== 模式 A: Split Layout (左右分栏) ================== */
.split-layout {
  flex: 1; display: flex; overflow: hidden; /* 核心：让子面板独立滚动 */
}

.text-section { margin-bottom: 40px; }
.section-subtitle { font-size: 1.3rem; color: #334155; border-bottom: 1px solid #e2e8f0; padding-bottom: 10px; margin-bottom: 15px; font-weight: 600; }
.section-content p { font-size: 2rem; line-height: 1.8; color: #475569; margin-bottom: 1em; text-align: justify; }
.conclusion-box { background: #ffffff; padding: 20px; border-radius: 6px; color: #475569; border: 1px solid #e2e8f0; border-left: 4px solid #10b981; margin-top: 40px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }
.conclusion-box h3 { color: #334155; margin-top: 0; }

/* 子 Agent 支撑论点框 */
.agent-summaries-box { background: #f9fafc; border: 1px solid #ebeef5; border-radius: 6px; padding: 15px; margin-top: 20px; }
.agent-summary-title { margin: 0 0 10px 0; font-size: 0.9rem; color: #909399; }
.agent-summary-item { display: flex; gap: 10px; margin-bottom: 10px; align-items: flex-start; }
.agent-summary-item:last-child { margin-bottom: 0; }
.agent-badge { background: #ecf5ff; color: #409EFF; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; white-space: nowrap; border: 1px solid #b3d8ff; }
.agent-text { font-size: 0.9rem; color: #606266; line-height: 1.5; }

/* --- 文本面板与卡片 --- */
.split-text-pane { flex: 4; min-width: 400px; padding: 30px; overflow-y: auto; background-color: transparent; border-right: 1px solid #e2e8f0; }
/* Dashboard.vue */
.split-chart-pane {
  flex: 6;
  padding: 6px; /* 给右侧区域一点四周的呼吸空间 */
  overflow-y: auto;
  background-color: transparent;

  /* 🔥 全局动态卡片网格 🔥 */
  display: grid;
  grid-template-columns: repeat(2, 1fr); /* 强制全局双列 */
  gap: 6px; /* 卡片之间的间距 */
  align-content: start; /* 防止卡片数量少时纵向被强行拉伸变形 */
}

/* 之前教你的魔法依然生效：
   由于这里是 Dashboard，我们需要使用深度选择器 :deep() 来穿透修改子组件的样式。
   如果屏幕上所有渲染出来的图表总数是奇数，最后一张图表会自动横跨两列！
*/
.split-chart-pane :deep(.chart-box:nth-child(even):last-child) {
  grid-column: 1 / -1 ;
}
.spatiotemporal-float {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  z-index: 10; /* 确保悬浮在内容上方 */
  margin: 0; /* 清除原有margin */
  border-radius: 8px 8px 0 0; /* 底部圆角取消，贴合面板底部 */
  border-left: none;
  border-right: none;
  border-bottom: none;
  box-shadow: 0 -2px 12px rgba(0,0,0,0.03); /* 阴影向上，更贴合悬浮效果 */
}

/* ================== 模式 B: Narrative Layout (瀑布流) ================== */
.narrative-layout { flex: 1; overflow-y: auto; scroll-behavior: smooth; }
.report-container { margin: 0 auto; padding: 40px; }
.doc-section { margin-bottom: 60px; }
.section-text-block { max-width: 800px; margin: 0 auto 30px auto; }
.section-index { background: #409EFF; color: white; width: 28px; height: 28px; display: inline-flex; align-items: center; justify-content: center; border-radius: 50%; font-size: 1rem; margin-right: 10px; }
.section-visual-block { width: 100%; }
.doc-conclusion { max-width: 800px; margin: 0 auto; padding: 30px; background: #fdf6ec; border-left: 5px solid #E6A23C; border-radius: 4px; color: #606266; }

/* Dashboard.vue 的 style 区域补充 */

.spatiotemporal-wrapper {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  z-index: 50;
  display: flex;
  flex-direction: column;
}

/* 悬浮触发条样式 */
/* --- 沙盒悬浮条 --- */
.sandbox-trigger-bar {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-top: 1px solid #e2e8f0;
  padding: 10px 24px; display: flex; justify-content: space-between; align-items: center;
  box-shadow: 0 -4px 16px rgba(0, 0, 0, 0.04);
  border-radius: 12px 12px 0 0;
}
.sandbox-info { font-size: 0.9rem; color: #475569; display: flex; align-items: center; gap: 8px; }
.highlight-time { color: #3b82f6; font-family: monospace; font-size: 1rem;}
.highlight-tags { color: #0ea5e9; font-weight: 500; font-size: 0.85rem;}

.re-anchor-btn {
  background: #2563eb; /* 专业主操作蓝 */
  color: white; border: none; padding: 8px 20px; border-radius: 6px; /* 去除大圆角，显得更严谨 */
  font-size: 0.9rem; font-weight: 500; cursor: pointer;
  box-shadow: 0 1px 3px rgba(37, 99, 235, 0.2); transition: all 0.2s;
}
.re-anchor-btn:hover { background: #1d4ed8; transform: translateY(-1px); box-shadow: 0 4px 6px rgba(37, 99, 235, 0.25); }

/* 极度丝滑的滑入动画 */
.slide-up-enter-active, .slide-up-leave-active {
  transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}
.slide-up-enter-from, .slide-up-leave-to {
  opacity: 0;
  transform: translateY(20px);
}
.sandbox-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.cancel-btn {
  background: transparent;
  color: #909399;
  border: none;
  font-size: 1rem;
  cursor: pointer;
  padding: 4px 8px;
  border-radius: 4px;
  transition: all 0.2s;
}
.cancel-btn:hover {
  color: #F56C6C;
  background: #fef0f0;
}


.sandbox-badge {
  position: absolute;
  top: -4px;
  right: -4px;
  font-size: 0.6rem;
  background: #fff;
  border-radius: 50%;
  padding: 1px;
}
/* --- 原有简略面板样式 --- */
.task-history-panel {
  display: flex;
  align-items: center;
  gap: 12px;
  position: relative; /* 关键：作为悬浮窗的定位基准 */
  padding: 8px 12px;
  border-radius: 8px;
  transition: background 0.3s;
}

.task-history-panel:hover {
  background-color: #263a82;
}
.history-label{
  color: white;
}
.history-list {
  display: flex;
  gap: 6px;
}

.history-btn {
  /* 你原有的按钮样式 */
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: 1px solid #dcdfe6;
  background: #263a82;
  color: white;
  font-size: 0.8rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  position: relative;
}
.history-btn.active {
  background: #fff;
  color: #263a82;
  border-color: white;
}

/* --- 新增：DAG 悬浮看板样式 --- */
.dag-hover-board {
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: 10px;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
  padding: 20px;
  min-width: 600px;
  max-width: 90vw;
  overflow-x: auto;
  z-index: 100;
  cursor: default;
}

.dag-board-header {
  margin-bottom: 20px;
  border-bottom: 1px dashed #dcdfe6;
  padding-bottom: 10px;
}
.dag-board-header h3 {
  margin: 0;
  font-size: 16px;
  color: #334155;
}
.dag-board-header .sub-text {
  font-size: 12px;
  color: #64748b;
}

/* Timeline 布局：多个历史任务横向排布 */
.dag-timeline {
  display: flex;
  align-items: flex-start;
  gap: 40px;
  padding-bottom: 10px;
}

/* 单个历史任务的树状结构 */
.dag-tree {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  cursor: pointer;
  transition: transform 0.2s;
}
.dag-tree:hover {
  transform: translateY(-2px);
}
.dag-tree.is-active .node {
  box-shadow: 0 2px 8px rgba(0,0,0,0.25);/*cbox-shadow: 0 0 0 1px #929393;  高亮当前选中的任务流 */
}

/* 节点通用样式 */
.node {
  padding: 10px;
  width: 180px;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.02);
  position: relative;
  z-index: 2;
  overflow: hidden;
}

.node-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 12px;
  font-weight: bold;
  color: #334155;
}

.node-content {
  font-size: 12px;
  color: #64748b;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  word-break: break-all;
}

/* 底部彩色进度条装饰 */
.progress-bar {
  position: absolute;
  bottom: 0;
  left: 0;
  height: 4px;
}
.red-bar { width: 100%; background-color: #94a3b8; }
.blue-bar { width: 100%; background-color: #cbd5e1; }

/* 根节点专属样式 */
.goal-node {
  border-color: #cbd5e1;
}
.goal-node .node-type { color: #475569; }
.badge-sandbox {
  background: #0ea5e9;
  color: white;
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 10px;
}

/* --- 子节点容器：垂直堆叠排列 --- */
.sub-nodes-container {
  display: flex;
  flex-direction: column; /* 改为垂直排列 */
  align-items: center;    /* 居中对齐 */
  gap: 12px;              /* 长条卡片之间的间距 */
  margin-top: 25px;       /* 为 Goal 节点下方的连线留出空间 */
  position: relative;
}

/* 贯穿始终的垂直中心主线 (连接 Goal 到最后一个子节点) */
.sub-nodes-container::before {
  content: '';
  position: absolute;
  top: -25px; /* 向上连接到 Goal 节点底部 */
  bottom: 0;  /* 向下贯穿整个容器 */
  left: 50%;
  width: 2px;
  background-color: #dcdfe6;
  transform: translateX(-50%);
  z-index: 1; /* 放在卡片底层 */
}
/* --- 长条形的子任务卡片 --- */
.node.sub-node {
  display: flex;
  align-items: center;
  justify-content: space-between; /* 左右两端对齐 */
  width: 220px; /* 设置为较宽的长条形 */
  padding: 10px 15px;
  background: white;
  border: 1px solid #dcdfe6;
  border-radius: 6px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  position: relative;
  z-index: 2; /* 确保卡片盖住背后的那条垂直连线 */
}

/* 长条卡片内部的排版 */
.sub-node-left {
  font-size: 12px;
  font-weight: bold;
  color: #303133;
}

.sub-node-right {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  font-weight: 500;
  color: #3b82f6;
}

.sub-node-right .node-content {
  font-weight: 500;
  color: #409eff; /* 探员名字用主题色高亮一下 */
}
/* 绘制子节点上方的分叉线段 */
.sub-node {
  position: relative;
}
.sub-node::before {
  content: '';
  position: absolute;
  top: -16px;
  left: 50%;
  width: 2px;
  height: 16px;
  background-color: #dcdfe6;
  transform: translateX(-50%);
}

/* 绘制连接分叉点的横线 (仅当有多个子节点时需要) */
.sub-nodes-container > .sub-node:first-child::after {
  content: '';
  position: absolute;
  top: -16px;
  left: 50%;
  width: 50%;
  height: 2px;
  background-color: #dcdfe6;
}
.sub-nodes-container > .sub-node:last-child::after {
  content: '';
  position: absolute;
  top: -16px;
  right: 50%;
  width: 50%;
  height: 2px;
  background-color: #dcdfe6;
}
/* 中间节点的横线全覆盖 */
.sub-nodes-container > .sub-node:not(:first-child):not(:last-child)::after {
  content: '';
  position: absolute;
  top: -16px;
  left: 0;
  width: 100%;
  height: 2px;
  background-color: #dcdfe6;
}
/* 如果只有一个子节点，隐藏横线 */
.sub-nodes-container > .sub-node:first-child:last-child::after {
  display: none;
}

/* 历史任务之间的虚线连接（跨越 Timeline） */
.history-connector {
  position: absolute;
  top: 40px; /* 对齐 Goal 节点的高度 */
  right: -40px; /* 跨越 gap 的距离 */
  width: 40px;
  border-top: 2px dashed #c0c4cc;
  z-index: 1;
}

/* Vue 过渡动画 */
.fade-slide-enter-active,
.fade-slide-leave-active {
  transition: opacity 0.3s, transform 0.3s;
}
.fade-slide-enter-from,
.fade-slide-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

/* 全屏地图容器 */
.full-screen-map-container {
  flex: 1;
  width: 100%;
  height: 100%;
  position: relative;
  background-color: #f5f7fa;
}

.preview-badge {
  position: absolute;
  top: 20px;
  left: 50%;
  transform: translateX(-50%);
  background-color: rgba(64, 158, 255, 0.9);
  color: white;
  padding: 6px 16px;
  border-radius: 20px;
  font-size: 0.9rem;
  font-weight: 500;
  box-shadow: 0 4px 12px rgba(64, 158, 255, 0.3);
  z-index: 10;
  pointer-events: none;
  animation: fadeInDown 0.5s ease-out;
}

@keyframes fadeInDown {
  from { opacity: 0; transform: translate(-50%, -20px); }
  to { opacity: 1; transform: translate(-50%, 0); }
}
</style>