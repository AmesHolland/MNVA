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
  <div class="dashboard-container">

    <aside class="left-panel">
      <header class="panel-header">
        <h2>Marine News</h2>
        <span class="subtitle">Multi-Agent Visual Analytics</span>
      </header>
      <main class="chat-stream">
        <ChatBox />
<!--        <ApprovalCard class="hitl-card" />-->
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

          <!-- 【新增】：任务历史切换面板 -->
          <div
      class="task-history-panel"
      v-if="store.taskHistory.length > 0"
      @mouseenter="handleMouseEnter"
      @mouseleave="handleMouseLeave"
    >
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
                  v-for="(subTask, key) in task.results.tasks"
                  :key="key"
                >
                  <div class="sub-node-left">
                    <span class="node-type">Task {{ key }}</span>
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
            <button
              :class="['toggle-btn', { active: viewMode === 'split' }]"
              @click="viewMode = 'split'"
              title="Split view: Left (text) / Right (charts)"
            >
              <span class="icon">◫</span> Dashboard
            </button>
            <button
              :class="['toggle-btn', { active: viewMode === 'narrative' }]"
              @click="viewMode = 'narrative'"
              title="Stream view: Text & images interleaved"
            >
              <span class="icon">📄</span> Narrative
            </button>
          </div>
        </header>

        <div v-if="viewMode === 'split'" class="split-layout">

          <div class="split-text-pane">
            <div class="executive-summary-card">
              <strong>Abstract：</strong>{{ store.analysisResults.report.executive_summary }}
            </div>
          <article v-for="(section, index) in store.analysisResults.report.sections" :key="'text-'+index" class="text-section">
            <h2 class="section-subtitle">{{ index + 1 }}. {{ section.subtitle }}</h2>

            <div class="section-content">
               <TraceableText :claims="section.content_claims" />
            </div>
          </article>
            <div class="conclusion-box">
              <h3>Conclusion</h3>
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

      <template v-else>
      </template>
<!--      <SpatiotemporalNavigator class="spatiotemporal-float" />-->
      <div class="spatiotemporal-float">
        <transition name="slide-up">
          <div v-if="store.brushState.timeRange || store.brushState.spatialLabels?.length" class="sandbox-trigger-bar">
            <div class="sandbox-info">
              <span class="icon">🎯</span>
              <span>已锁定局部时空：</span>

              <strong v-if="store.brushState.timeRange" class="highlight-time">
                {{ store.formatDate(store.brushState.timeRange[0]) }} 至
                {{ store.formatDate(store.brushState.timeRange[1]) }}
              </strong>
              <strong v-else class="highlight-time">全局时间</strong>

              <span v-if="store.brushState.spatialLabels?.length" class="highlight-tags">
                 | 区域: {{ store.brushState.spatialLabels.join(', ') }}
              </span>
            </div>

            <div class="sandbox-actions">
              <button class="cancel-btn" @click="store.clearBrushState" title="清除时空约束">✖</button>

              <button class="re-anchor-btn" @click="store.prepareSandboxAnalysis">Edit Prompt
              </button>
            </div>
          </div>
        </transition>

        <SpatiotemporalNavigator />
      </div>
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
.right-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background-color: #f5f7fa;
  position: relative; /* 🔴 让子元素绝对定位基于此容器 */
  padding-bottom: 180px; /* 🔴 预留悬浮组件高度（根据实际调整） */
}
.top-toolbar {
  display: flex; justify-content: space-between; align-items: center;
  padding: 15px 30px; background-color: #ffffff; border-bottom: 1px solid #e4e7ed;
  box-shadow: 0 2px 8px rgba(0,0,0,0.02); z-index: 5;
}
.report-title { margin: 0; font-size: 20px; color: #1f2f3d; }

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
.sandbox-trigger-bar {
  background: rgba(253, 246, 236, 0.95); /* 浅橙色背景，带点透明度 */
  backdrop-filter: blur(10px);
  border-top: 1px solid #faecd8;
  padding: 10px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 -4px 16px rgba(230, 162, 60, 0.15); /* 向上投射的阴影 */
  border-radius: 12px 12px 0 0;
}

.sandbox-info {
  font-size: 0.9rem;
  color: #606266;
  display: flex;
  align-items: center;
  gap: 8px;
}

.sandbox-info .icon { font-size: 1.1rem; }
.highlight-time { color: #E6A23C; font-family: monospace; font-size: 1rem;}
.highlight-tags { color: #409EFF; font-weight: 500; font-size: 0.85rem;}

.re-anchor-btn {
  background: linear-gradient(135deg, #FF9A9E 0%, #FECFEF 99%, #FECFEF 100%);
  background: #F56C6C; /* 如果不喜欢渐变，用纯色也很稳重 */
  color: white;
  border: none;
  padding: 8px 20px;
  border-radius: 20px;
  font-size: 0.9rem;
  font-weight: bold;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(245, 108, 108, 0.3);
  transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1);
}

.re-anchor-btn:hover {
  background: #f78989;
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(245, 108, 108, 0.4);
}

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
  background-color: #f5f7fa;
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
  background: #fff;
  color: #606266;
  font-size: 0.8rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  position: relative;
}
.history-btn.active {
  background: #409eff;
  color: white;
  border-color: #409eff;
}

/* --- 新增：DAG 悬浮看板样式 --- */
.dag-hover-board {
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: 10px;
  background: #f8f9fc;
  border: 1px solid #e4e7ed;
  border-radius: 12px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
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
  color: #303133;
}
.dag-board-header .sub-text {
  font-size: 12px;
  color: #909399;
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
  background: white;
  border: 1px solid #dcdfe6;
  border-radius: 6px;
  padding: 10px;
  width: 180px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
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
  color: #303133;
}

.node-content {
  font-size: 12px;
  color: #606266;
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
.red-bar { width: 50%; background-color: #f56c6c; }
.blue-bar { width: 100%; background-color: #409eff; }

/* 根节点专属样式 */
.goal-node {
  border-color: #f56c6c;
}
.goal-node .node-type { color: #f56c6c; }
.badge-sandbox {
  background: #e6a23c;
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
  color: #606266;
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

</style>