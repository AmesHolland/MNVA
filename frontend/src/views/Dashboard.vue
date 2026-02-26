<script setup>
import { computed, ref } from 'vue'
import { useChatStore } from '../store/chatStore'

// 引入左侧组件
import ChatBox from '../components/chat/ChatBox.vue'
import ApprovalCard from '../components/chat/ApprovalCard.vue'
import ChatInput from '../components/chat/ChatInput.vue'
import TraceableText from '../components/visualizations/TraceableText.vue'
import EvidenceDrawer from '../components/visualizations/EvidenceDrawer.vue'
// 引入图表分发组件 (记得确保你有这个组件，或者把代码平铺进来)
import ChartRenderer from '../components/visualizations/ChartRender.vue'

const store = useChatStore()

// 核心状态：视图模式切换 ('split' 左右分栏模式 | 'narrative' 瀑布流模式)
const viewMode = ref('split')

// 辅助方法：图表去重引擎
// 无论哪种模式，我们都要确保图表不重复出现
const getUniqueTasks = (sections) => {
  const tasks = store.analysisResults?.tasks || {}
  const seen = new Set()
  const uniqueTasks = []

  sections.forEach(sec => {
    sec.ref_task_ids?.forEach(id => {
      if (!seen.has(id) && tasks[id]) {
        seen.add(id)
        uniqueTasks.push({ ...tasks[id], task_id: id })
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
</script>

<template>
  <div class="dashboard-container">

    <aside class="left-panel">
      <header class="panel-header">
        <h2>海洋新闻态势感知</h2>
        <span class="subtitle">Multi-Agent Visual Analytics</span>
      </header>
      <main class="chat-stream">
        <ChatBox />
        <ApprovalCard class="hitl-card" />
      </main>
      <footer class="input-area">
        <ChatInput />
      </footer>
    </aside>

    <section class="right-panel">

      <template v-if="store.analysisResults?.report">

        <header class="top-toolbar">
          <div class="report-meta">
            <h1 class="report-title">{{ store.analysisResults.report.report_title }}</h1>
          </div>

          <div class="view-toggle">
            <button
              :class="['toggle-btn', { active: viewMode === 'split' }]"
              @click="viewMode = 'split'"
              title="左右分栏：左侧阅读，右侧看图"
            >
              <span class="icon">◫</span> 仪表盘模式
            </button>
            <button
              :class="['toggle-btn', { active: viewMode === 'narrative' }]"
              @click="viewMode = 'narrative'"
              title="瀑布流：图文穿插阅读"
            >
              <span class="icon">📄</span> 叙事模式
            </button>
          </div>
        </header>

        <div v-if="viewMode === 'split'" class="split-layout">

          <div class="split-text-pane">
            <div class="executive-summary-card">
              <strong>摘要：</strong>{{ store.analysisResults.report.executive_summary }}
            </div>

<!--            <article-->
<!--              v-for="(section, index) in store.analysisResults.report.sections"-->
<!--              :key="'text-'+index"-->
<!--              class="text-section"-->
<!--            >-->
<!--              <h2 class="section-subtitle">{{ index + 1 }}. {{ section.subtitle }}</h2>-->

<!--              <div class="section-content">-->
<!--                <p v-for="(paragraph, pIdx) in section.content.split('\n')" :key="'p-'+pIdx">-->
<!--                  {{ paragraph }}-->
<!--                </p>-->
<!--              </div>-->

<!--              <div class="agent-summaries-box">-->
<!--                <h4 class="agent-summary-title">底层研判支撑：</h4>-->
<!--                <template v-for="taskId in section.ref_task_ids" :key="'sum-'+taskId">-->
<!--                  <div class="agent-summary-item" v-if="store.analysisResults.tasks[taskId]">-->
<!--&lt;!&ndash;                    <span class="agent-badge">{{ store.analysisResults.tasks[taskId].agent_name.replace('_Agent', '') }}</span>&ndash;&gt;-->
<!--                    <span class="agent-text">{{ store.analysisResults.tasks[taskId].summary }}</span>-->
<!--                  </div>-->
<!--                </template>-->
<!--              </div>-->
<!--            </article>-->
          <article v-for="(section, index) in store.analysisResults.report.sections" :key="'text-'+index" class="text-section">
            <h2 class="section-subtitle">{{ index + 1 }}. {{ section.subtitle }}</h2>

            <div class="section-content">
               <TraceableText :claims="section.content_claims" />
            </div>
          </article>
            <div class="conclusion-box">
              <h3>战略结语</h3>
              <p>{{ store.analysisResults.report.conclusion }}</p>
            </div>
          </div>

          <div class="split-chart-pane">
            <template v-for="task in allUniqueTasks" :key="'chart-'+task.task_id">
              <ChartRenderer :task="task" />
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
<!--              <div class="section-text-block">-->
<!--                <h2 class="section-subtitle"><span class="section-index">{{ index + 1 }}</span>{{ section.subtitle }}</h2>-->
<!--                <div class="section-paragraphs">-->
<!--                  <p v-for="(paragraph, pIdx) in section.content.split('\n')" :key="pIdx">{{ paragraph }}</p>-->
<!--                </div>-->
<!--              </div>-->
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
              <h3>战略结语</h3>
              <p>{{ store.analysisResults.report.conclusion }}</p>
            </div>
          </div>
        </div>

      </template>

      <template v-else>
      </template>

    </section>

    <EvidenceDrawer />
  </div>
</template>

<style scoped>
/* 保持原有的左侧样式 */
.dashboard-container { display: flex; height: 100vh; width: 100vw; background-color: #f0f2f5; overflow: hidden; font-family: -apple-system, sans-serif; }
.left-panel { width: 380px; background-color: #ffffff; border-right: 1px solid #e4e7ed; display: flex; flex-direction: column; z-index: 10; }
.panel-header { padding: 20px; border-bottom: 1px solid #ebeef5; background-color: #FAFAFA; }
.panel-header h2 { margin: 0; font-size: 1.2rem; color: #303133; }
.panel-header .subtitle { font-size: 0.8rem; color: #909399; }
.chat-stream { flex: 1; padding: 15px; overflow-y: auto; background-color: #f9fafc; }
.input-area { padding: 15px; background-color: #ffffff; border-top: 1px solid #ebeef5; }

/* 右侧顶部工具栏 */
.right-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; background-color: #f5f7fa; }
.top-toolbar {
  display: flex; justify-content: space-between; align-items: center;
  padding: 15px 30px; background-color: #ffffff; border-bottom: 1px solid #e4e7ed;
  box-shadow: 0 2px 8px rgba(0,0,0,0.02); z-index: 5;
}
.report-title { margin: 0; font-size: 1.4rem; color: #1f2f3d; }

/* 切换按钮组 */
.view-toggle { display: flex; background: #f0f2f5; border-radius: 6px; padding: 4px; }
.toggle-btn {
  border: none; background: transparent; padding: 6px 12px; border-radius: 4px;
  cursor: pointer; color: #909399; font-size: 0.9rem; font-weight: 500; transition: all 0.2s;
}
.toggle-btn:hover { color: #303133; }
.toggle-btn.active { background: #ffffff; color: #409EFF; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }

/* 公共摘要卡片 */
.executive-summary-card { background: #fdf6ec; padding: 15px 20px; border-radius: 6px; color: #E6A23C; border-left: 4px solid #E6A23C; margin-bottom: 20px; line-height: 1.6; font-size: 0.95rem; }

/* ================== 模式 A: Split Layout (左右分栏) ================== */
.split-layout {
  flex: 1; display: flex; overflow: hidden; /* 核心：让子面板独立滚动 */
}

/* 左侧文本面板 */
.split-text-pane {
  flex: 4; /* 占比 40% */
  min-width: 400px; padding: 30px; overflow-y: auto;
  background-color: #ffffff; border-right: 1px solid #e4e7ed;
}

.text-section { margin-bottom: 40px; }
.section-subtitle { font-size: 1.3rem; color: #303133; border-bottom: 2px solid #ebeef5; padding-bottom: 10px; margin-bottom: 15px; }
.section-content p { font-size: 1rem; line-height: 1.8; color: #333; margin-bottom: 1em; text-align: justify; }

/* 子 Agent 支撑论点框 */
.agent-summaries-box { background: #f9fafc; border: 1px solid #ebeef5; border-radius: 6px; padding: 15px; margin-top: 20px; }
.agent-summary-title { margin: 0 0 10px 0; font-size: 0.9rem; color: #909399; }
.agent-summary-item { display: flex; gap: 10px; margin-bottom: 10px; align-items: flex-start; }
.agent-summary-item:last-child { margin-bottom: 0; }
.agent-badge { background: #ecf5ff; color: #409EFF; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; white-space: nowrap; border: 1px solid #b3d8ff; }
.agent-text { font-size: 0.9rem; color: #606266; line-height: 1.5; }

/* 右侧图表面板 */
.split-chart-pane {
  flex: 6; /* 占比 60% */
  padding: 30px; overflow-y: auto; background-color: #f0f2f5;
  display: flex; flex-direction: column; gap: 20px;
}

.conclusion-box { background: #f0f9eb; padding: 20px; border-radius: 6px; color: #67C23A; border-left: 4px solid #67C23A; margin-top: 40px;}

/* ================== 模式 B: Narrative Layout (瀑布流) ================== */
.narrative-layout { flex: 1; overflow-y: auto; scroll-behavior: smooth; }
.report-container { margin: 0 auto; padding: 40px; }
.doc-section { margin-bottom: 60px; }
.section-text-block { max-width: 800px; margin: 0 auto 30px auto; }
.section-index { background: #409EFF; color: white; width: 28px; height: 28px; display: inline-flex; align-items: center; justify-content: center; border-radius: 50%; font-size: 1rem; margin-right: 10px; }
.section-visual-block { width: 100%; }
.doc-conclusion { max-width: 800px; margin: 0 auto; padding: 30px; background: #fdf6ec; border-left: 5px solid #E6A23C; border-radius: 4px; color: #606266; }
</style>