<script setup>
import {computed, onMounted, ref} from 'vue'
import { useChatStore } from '../store/chatStore'

// 引入左侧组件
import ChatBox from '../components/chat/ChatBox.vue'
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
// 🌟 1. 新增：记录当前激活的地图视角，默认为全局
const activeMapTab = ref('global');

// 🌟 2. 重构 unifiedMapData：收集全局数据和【多个】深度挖掘数据
const unifiedMapData = computed(() => {
  let globalData = [];
  let deepDiveTasks = []; // 改为数组

  allUniqueTasks.value.forEach(task => {
    if (task.agent_name === 'Global_Monitor_Agent') {
      globalData = task.visualization_data.geo_dynamic_data || [];
    }
    if (task.agent_name === 'Deep_Dive_Agent' && task.visualization_data.map_chart) {
      // 收集每一个有地图数据的 Deep Dive 任务
      deepDiveTasks.push({
        task_id: task.task_id,
        // 取 summary 的前几个字作为 Tab 标签，或者用 Agent 名称
        title: task.summary ? task.summary.substring(0, 20) + '...' : `Entity Focus ${deepDiveTasks.length + 1}`,
        data: task.visualization_data.map_chart
      });
    }
  });

  if (globalData.length === 0 && deepDiveTasks.length === 0) return null;

  // 防御性逻辑：如果当前选中的 deep_dive 任务被清除了，重置回 global 视角
  if (activeMapTab.value !== 'global' && !deepDiveTasks.find(t => t.task_id === activeMapTab.value)) {
    activeMapTab.value = 'global';
  }

  return { globalData, deepDiveTasks };
});

// 🌟 3. 新增计算属性：动态计算传给 Echarts 的 chartType
const currentUnifiedChartType = computed(() => {
  return activeMapTab.value === 'global' ? 'global_map' : 'unified_map';
});

// 🌟 4. 新增计算属性：动态计算传给 Echarts 的 chartData
// 🌟 Dashboard.vue 内部
const currentUnifiedChartData = computed(() => {
  if (!unifiedMapData.value) return null;

  if (activeMapTab.value === 'global') {
    // 全局视角：传原始 globalData，交给 global_map 渲染
    return unifiedMapData.value.globalData;
  } else {
    // Focus 视角：交给 unified_map 渲染
    const targetTask = unifiedMapData.value.deepDiveTasks.find(t => t.task_id === activeMapTab.value);

    return {
      globalData: [], // 👈 核心魔术：故意传空数组！这样背景就不会渲染那些蓝色的散点了
      deepDiveData: targetTask ? targetTask.data : []
    };
  }
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

// ==========================================
// === 🌟 3. 新增：数据上传与仓库管理逻辑 ===
// ==========================================
const isUploadModalOpen = ref(false)
const newDatasetName = ref('')
const selectedFile = ref(null)
const isUploading = ref(false)

// 页面挂载时，自动拉取后端已有的数据集列表
onMounted(() => {
  store.fetchDatasets()
})

// 处理文件拖拽或点击选择
const handleFileSelect = (e) => {
  selectedFile.value = e.target.files[0]
}

// 执行核心上传逻辑
const uploadFile = async () => {
  if (!selectedFile.value) {
    alert("Please select a file to upload.")
    return
  }

  isUploading.value = true
  const formData = new FormData()
  formData.append('file', selectedFile.value)
  if (newDatasetName.value) {
    formData.append('dataset_name', newDatasetName.value)
  }

  try {
    // ⚠️ 请确保这里的端口号(5000)与你 Flask 启动的端口一致
    const res = await fetch('http://localhost:5000/api/upload_dataset', {
      method: 'POST',
      body: formData
    })

    const data = await res.json()

    if (data.success) {
      alert(`Success! Imported ${data.dataset.row_count} intelligence records.`)

      // 1. 重置弹窗状态
      isUploadModalOpen.value = false
      newDatasetName.value = ''
      selectedFile.value = null

      // 2. 重新拉取列表，并让 Store 自动选中刚上传的这个数据集
      await store.fetchDatasets()
      store.setActiveDataset(data.dataset.dataset_id)

    } else {
      alert("Error: " + data.error)
    }
  } catch (err) {
    alert("Upload failed: " + err.message)
  } finally {
    isUploading.value = false
  }
}

// 🌟 新增：追踪当前选中的 Tab（默认为阶段 1，如果想默认看总报告可以设为 'final_report'）
const activeTab = ref('final_report')

// 🌟 新增：根据 activeTab 动态计算当前要渲染的阶段数据
const currentPhaseData = computed(() => {
  if (!store.analysisResults?.phase_data_list) return null
  return store.analysisResults.phase_data_list.find(p => p.phase_index === activeTab.value)
})
import { nextTick } from 'vue' // 确保引入 nextTick
import html2pdf from 'html2pdf.js' // 🌟 引入 PDF 导出库
// ... (保留你原有的其他 import)

// ... (保留你原有的状态定义)

// 🌟 新增：导出 PDF 相关的状态
const isExporting = ref(false)

// 🌟 新增：核心导出函数
const exportToPDF = async () => {
  // 1. 防御性检查：确保报告已经生成
  if (!store.analysisResults?.report) {
    alert("Please generate a report first before exporting.")
    return
  }

  isExporting.value = true

  // 2. 智能视图切换：如果用户没在 Final Report 视图，自动切过去
  if (activeTab.value !== 'final_report') {
    activeTab.value = 'final_report'
    // 等待 Vue 将 DOM 切换过来
    await nextTick()
    // 故意等待 800ms，让 ECharts 的入场动画彻底播完，防止截到半截的图表
    await new Promise(resolve => setTimeout(resolve, 800))
  }

  // 3. 获取我们要导出的目标 DOM 节点
  const element = document.querySelector('.split-text-pane')
  if (!element) {
    alert("Report container not found.")
    isExporting.value = false
    return
  }

  // 4. 动态获取报告标题作为文件名
  const fileName = `${store.analysisResults.report.report_title || 'Marine_News_Analysis'}.pdf`

  // 5. 配置 html2pdf 参数 (A4 纸张，高清渲染)
  const opt = {
    margin:       [15, 15, 15, 15], // 上, 左, 下, 右 的边距 (mm)
    filename:     fileName,
    image:        { type: 'jpeg', quality: 0.98 },
    html2canvas:  { scale: 2, useCORS: true }, // scale: 2 保证 ECharts 截图清晰不模糊
    jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' }
  }

  // 6. 执行导出
  try {
    await html2pdf().set(opt).from(element).save()
  } catch (error) {
    console.error("PDF Export Failed:", error)
    alert("Export failed, please check console.")
  } finally {
    isExporting.value = false
  }
}
</script>

<template>
  <div class="app-wrapper">

    <header class="global-top-bar">
      <div class="brand-area" style="display: flex; align-items: center; gap: 40px;">
        <div class="brand">
          <span class="sys-name">EventHub</span>
          <span class="sys-subname">Multi-Agent Visual Analytics</span>
        </div>
        <div class="dataset-manager">
          <span class="icon">📂 Corpus:</span>
          <select v-model="store.activeDatasetId" class="dataset-select">
            <option disabled value="">Select Analysis Corpus...</option>
            <option v-for="ds in store.availableDatasets" :key="ds.dataset_id" :value="ds.dataset_id">
              {{ ds.dataset_name }} ({{ ds.row_count }} records)
            </option>
          </select>
          <button class="upload-trigger-btn" @click="isUploadModalOpen = true">➕ Import</button>
        </div>
      </div>

      <div class="global-actions" v-if="store.analysisResults?.report">
        <button class="export-pdf-btn" @click="exportToPDF" :disabled="isExporting">
           <span class="icon">{{ isExporting ? '⏳' : '📥' }}</span>
           {{ isExporting ? 'Generating PDF...' : 'Export Report' }}
        </button>
        <div class="phase-stepper-container">
      <div class="phase-stepper">
        <template v-for="(phase, index) in store.analysisResults.phase_data_list" :key="'step-'+phase.phase_index">
          <div class="step-item" :class="{ active: activeTab === phase.phase_index }" @click="activeTab = phase.phase_index">
            <div class="step-marker">{{ phase.phase_index }}</div>
            <div class="step-info">
              <template v-if="activeTab === phase.phase_index">
                <span class="step-label">Phase {{ phase.phase_index }}</span>
                <span class="step-title" :title="phase.phase_name">{{ phase.phase_name }}</span>
              </template>
              <template v-else>
                <span class="step-title">PHASE {{ phase.phase_index }}</span>
              </template>
            </div>
          </div>
          <div class="step-connector">➔</div>
        </template>

        <div class="step-item final-step" :class="{ active: activeTab === 'final_report' }" @click="activeTab = 'final_report'">
          <div class="step-marker">📑</div>
          <div class="step-info">
             <template v-if="activeTab === 'final_report'">
                <span class="step-label">Synthesis</span>
                <span class="step-title">Final Report</span>
              </template>
              <template v-else>
                <span class="step-title">Final Report</span>
              </template>
          </div>
        </div>
      </div>
    </div>

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
                          <span class="node-content">{{ subTask?.agent_name ? subTask.agent_name.slice(0, -6) : '' }}</span>
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
    <transition name="fade">
      <div v-if="isUploadModalOpen" class="upload-modal-overlay">
        <div class="upload-modal">
          <div class="modal-header">
            <h3>Import Intelligence Corpus</h3>
            <button class="close-btn" @click="isUploadModalOpen = false">✖</button>
          </div>

          <div class="modal-body">
            <label>Dataset Name (Optional)</label>
            <input type="text" v-model="newDatasetName" placeholder="e.g., 2025 Q4 South China Sea..." class="dataset-name-input" />

            <label>Upload File (.csv, .xlsx, .json)</label>
            <div class="file-drop-zone">
              <input type="file" accept=".csv,.xlsx,.json" @change="handleFileSelect" />
              <p class="file-hint" v-if="!selectedFile">Click to browse or drop file here</p>
              <p class="file-hint selected" v-else>📁 {{ selectedFile.name }}</p>
            </div>
          </div>

          <div class="modal-actions">
             <button @click="isUploadModalOpen = false" class="cancel-btn">Cancel</button>
             <button @click="uploadFile" class="upload-btn" :disabled="isUploading">
               {{ isUploading ? 'Extracting & Standardizing...' : 'Confirm Upload' }}
             </button>
          </div>
        </div>
      </div>
    </transition>
    <div class="dashboard-container">

      <aside class="left-panel">
  <!--      <header class="panel-header">-->
  <!--        <h2>Marine News</h2>-->
  <!--        <span class="subtitle">Multi-Agent Visual Analytics</span>-->
  <!--      </header>-->
        <main class="chat-stream">
          <ChatBox />

        </main>
        <footer class="input-area">
          <ChatInput />
        </footer>
      </aside>

      <section class="right-panel">

        <template v-if="store.analysisResults?.report">

          <div class="split-layout" v-if="activeTab !== 'final_report' && currentPhaseData">

            <div class="split-text-pane">
              <div class="phase-summary-card">
                <h2 class="section-subtitle">{{ currentPhaseData.phase_name }}</h2>
                <div class="time-badge" style="margin-bottom: 12px; font-size: 13px; color: #409EFF;">
                  🕒 {{ currentPhaseData.phase_time_range || 'Global' }}
                </div>
                <p style="font-size: 14px; line-height: 1.6; color: #333; margin-bottom: 16px;">
                  {{ currentPhaseData.phase_summary }}
                </p>
              </div>

              <article v-for="task in currentPhaseData.subtasks" :key="task.task_id" class="text-section" style="margin-top: 24px;">
                <h3 class="section-subtitle" style="font-size: 15px; border-left: 3px solid #67C23A; padding-left: 8px;">
                  🤖 {{ task.agent_name.replace('_Agent', '') }} Analysis
                </h3>

                <div class="section-content">
                  <div v-if="task.factual_grounding?.length > 0" style="margin-bottom: 12px;">
                    <h4 style="font-size: 13px; color: #909399; margin-bottom: 8px;">📌 Objective Facts</h4>
                    <TraceableText :claims="task.factual_grounding" />
                  </div>

                  <div v-if="task.strategic_insights && Object.keys(task.strategic_insights).length > 0" class="analysis-section insight-section">
                    <h4 class="section-title">💡 Strategic Insights</h4>
                    <ul class="insight-list">
                      <li v-for="(val, key) in task.strategic_insights" :key="'ins-'+key">
                        <strong>{{ key.replace('_', ' ').toUpperCase() }}:</strong>

                        {{ val.statement || val }}

                        <sup
                          v-if="val.source_ids && val.source_ids.length > 0"
                          class="citation-badge insight-citation"
                          @click.stop="store.openEvidence(val.source_ids, key, val.statement)"
                          title="View inspiring evidence"
                        >
                          [{{ val.source_ids.length }} src]
                        </sup>
                      </li>
                    </ul>
                  </div>
                </div>
              </article>
            </div>

            <div class="split-chart-pane">
              <template v-for="task in currentPhaseData.subtasks" :key="'chart-'+task.task_id">
                <ChartRenderer :task="task" v-if="task.visualization_data" />
              </template>

              <div v-if="currentPhaseData.subtasks.filter(t => t.visualization_data).length === 0" style="text-align: center; color: #909399; margin-top: 40px;">
                 <span style="font-size: 32px;">📊</span>
                 <p>No visualization data generated for this phase.</p>
              </div>
            </div>

          </div>

          <div class="narrative-layout" v-else-if="activeTab === 'final_report'">
            <div v-if="viewMode === 'split'" class="split-layout">

            <div class="split-text-pane">
                <strong style="font-size: 30px;padding-bottom: 10px">{{ store.analysisResults.report.report_title }}</strong>
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

            <div class="split-chart-pane">

                <div class="chart-box" style="grid-column: 1 / -1; " v-if="unifiedMapData">

                  <div class="chart-header map-tabs-header">
                    <span class="map-title">🌍 Spatiotemporal View</span>

                    <div class="map-tabs">
                      <button
                        class="map-tab-btn"
                        :class="{ active: activeMapTab === 'global' }"
                        @click="activeMapTab = 'global'"
                      >
                        Macro Context
                      </button>

                      <button
                        v-for="(dd, index) in unifiedMapData.deepDiveTasks"
                        :key="dd.task_id"
                        class="map-tab-btn"
                        :class="{ active: activeMapTab === dd.task_id }"
                        @click="activeMapTab = dd.task_id"
                        :title="dd.title"
                      >
                        🎯 Focus {{ index + 1 }}
                      </button>
                    </div>
                  </div>

                  <div class="chart-content" style="height: 400px;">
                    <EchartsCanvas
                      :key="'map-' + activeMapTab"
                      :chartType="currentUnifiedChartType"
                      :chartData="currentUnifiedChartData"
                    />
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
          </div>

        </template>

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

.text-section { margin-bottom: 10px; }
.section-subtitle { font-size: 1.3rem; color: #334155; border-bottom: 1px solid #e2e8f0; padding-bottom: 10px; margin-bottom: 15px; font-weight: 600; }
.section-content p { font-size: 20px; line-height: 0.5; color: #475569; margin-bottom: 1em; text-align: justify; }
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
/* --- 数据集管理器样式 --- */
.dataset-manager {
  display: flex;
  align-items: center;
  gap: 12px;
  background: rgba(255, 255, 255, 0.1);
  padding: 4px 12px;
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.2);
}
.dataset-manager .icon {
  font-size: 0.9rem;
  color: #fff;
  font-weight: bold;
}
.dataset-select {
  background: transparent;
  color: #fff;
  border: none;
  outline: none;
  font-size: 0.9rem;
  font-weight: 500;
  cursor: pointer;
  max-width: 250px;
  text-overflow: ellipsis;
}
.dataset-select option {
  color: #333; /* 下拉列表展开后的文字颜色 */
}
.upload-trigger-btn {
  background: #409EFF;
  color: white;
  border: none;
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 0.8rem;
  cursor: pointer;
  font-weight: bold;
  transition: all 0.2s;
}
.upload-trigger-btn:hover {
  background: #66b1ff;
  transform: translateY(-1px);
}

/* --- 上传弹窗样式 --- */
.upload-modal-overlay {
  position: fixed;
  top: 0; left: 0; width: 100vw; height: 100vh;
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(4px);
  display: flex; justify-content: center; align-items: center;
  z-index: 9999;
}
.upload-modal {
  background: #fff;
  width: 450px;
  border-radius: 12px;
  box-shadow: 0 12px 32px rgba(0,0,0,0.2);
  display: flex; flex-direction: column;
  overflow: hidden;
}
.modal-header {
  padding: 16px 20px;
  background: #f5f7fa;
  border-bottom: 1px solid #ebeef5;
  display: flex; justify-content: space-between; align-items: center;
}
.modal-header h3 { margin: 0; font-size: 1.1rem; color: #303133; }
.modal-header .close-btn { background: none; border: none; font-size: 1.2rem; cursor: pointer; color: #909399; }
.modal-body {
  padding: 20px;
  display: flex; flex-direction: column; gap: 10px;
}
.modal-body label { font-size: 0.85rem; font-weight: bold; color: #606266; }
.dataset-name-input {
  padding: 10px; border: 1px solid #dcdfe6; border-radius: 6px; font-size: 0.95rem; outline: none;
}
.dataset-name-input:focus { border-color: #409EFF; }
.file-drop-zone {
  position: relative;
  height: 120px;
  border: 2px dashed #dcdfe6;
  border-radius: 8px;
  display: flex; justify-content: center; align-items: center;
  background: #fafafa;
  transition: all 0.3s ease;
}
.file-drop-zone:hover { border-color: #409EFF; background: #ecf5ff; }
.file-drop-zone input[type="file"] {
  position: absolute; width: 100%; height: 100%; opacity: 0; cursor: pointer;
}
.file-hint { color: #909399; font-size: 0.9rem; pointer-events: none; }
.file-hint.selected { color: #67C23A; font-weight: bold; }
.modal-actions {
  padding: 16px 20px; background: #fafafa; border-top: 1px solid #ebeef5;
  display: flex; justify-content: flex-end; gap: 12px;
}
.cancel-btn { padding: 8px 16px; background: #fff; border: 1px solid #dcdfe6; border-radius: 6px; cursor: pointer; color: #606266;}
.upload-btn { padding: 8px 16px; background: #409EFF; border: none; border-radius: 6px; cursor: pointer; color: white; font-weight: bold;}
.upload-btn:disabled { background: #a0cfff; cursor: not-allowed; }

/* 弹窗渐变动画 */
.fade-enter-active, .fade-leave-active { transition: opacity 0.3s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

/* ================= Stepper 导航条 ================= */
/* 移除原先作为块级元素的背景和 sticky，以适应顶栏 */
.phase-stepper-container {
  display: flex;
  align-items: center;
}

.phase-stepper {
  display: flex;
  align-items: center;
  gap: 4px; /* 稍微缩小 gap 让顶栏不至于过挤 */
}

.step-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 3px 6px; /* 调整 padding 以适配顶栏高度 */
  border-radius: 8px;
  cursor: pointer;
  background: transparent; /* 未激活时背景透明 */
  border: 1px solid transparent;
  transition: all 0.3s ease;
}

.step-item:hover {
  background: #f0f4f8;
}

.step-item.active {
  background: #ecf5ff;
  border-color: #b3d8ff;
  /* 如果你的顶栏已经有阴影，建议移除这里的 box-shadow 保持清爽 */
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.1);
}

.step-marker {
  width: 24px;
  height: 24px;
  background: #e4e7ed;
  color: #909399;
  border-radius: 50%;
  display: flex;
  justify-content: center;
  align-items: center;
  font-weight: bold;
  font-size: 12px;
}

.step-item.active .step-marker {
  background: #409EFF;
  color: #fff;
}

.step-info {
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.step-label {
  font-size: 8px;
  color: #909399;
  font-weight: 600;
  text-transform: uppercase;
  margin-bottom: 2px;
}

.step-title {
  font-size: 8px;
  color: #606266;
  font-weight: bold;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 150px;
}

.step-item.active .step-title {
  color: #409EFF;
}

.step-connector {
  color: #dcdfe6;
  font-size: 14px;
}
/* ================= 左右分屏布局 ================= */
.phase-view-layout {
  display: grid;
  grid-template-columns: 45% 55%; /* 左文右图的黄金比例 */
  gap: 24px;
  padding: 24px;
  height: calc(100vh - 160px); /* 减去头部高度，实现内部滚动 */
}

.phase-left-pane, .phase-right-pane {
  overflow-y: auto;
  padding-right: 8px;
}

/* 隐藏滚动条让视觉更干净 */
.phase-left-pane::-webkit-scrollbar, .phase-right-pane::-webkit-scrollbar { width: 6px; }
.phase-left-pane::-webkit-scrollbar-thumb, .phase-right-pane::-webkit-scrollbar-thumb { background: #dcdfe6; border-radius: 4px; }

/* ================= 左侧：文本分析卡片 ================= */
.phase-summary-card {
  background: #fff;
  border-left: 4px solid #409EFF;
  padding: 20px;
  border-radius: 4px 8px 8px 4px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.05);
  margin-bottom: 24px;
}
.summary-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.summary-header h3 { margin: 0; font-size: 18px; color: #303133; }
.time-badge { font-size: 12px; background: #f0f2f5; padding: 4px 8px; border-radius: 4px; color: #606266; font-weight: bold; }
.summary-text { font-size: 15px; line-height: 1.6; color: #444; margin-bottom: 16px; text-align: justify; }
.summary-meta { font-size: 13px; color: #606266; background: #fafafa; padding: 12px; border-radius: 6px; }
.meta-item { margin-bottom: 6px; }
.meta-item:last-child { margin-bottom: 0; }

.agent-analysis-card {
  background: #fff;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
  position: relative;
}
.agent-badge {
  display: inline-block;
  background: #fdf6ec; color: #e6a23c;
  font-size: 12px; font-weight: bold;
  padding: 4px 10px; border-radius: 12px;
  margin-bottom: 12px;
}
.section-title { font-size: 13px; color: #909399; text-transform: uppercase; margin: 0 0 10px 0; border-bottom: 1px solid #f0f2f5; padding-bottom: 4px; }
.fact-list, .insight-list { margin: 0; padding-left: 20px; font-size: 14px; line-height: 1.6; color: #333; }
.fact-list li, .insight-list li { margin-bottom: 8px; text-align: justify; }

.insight-section { margin-top: 16px; background: #f4f9ff; padding: 12px; border-radius: 6px; border-left: 2px solid #b3d8ff;}
.insight-list { padding-left: 16px; }
.insight-list li { color: #2a5caa; }

.citation-badge { color: rgb(209, 218, 220); cursor: pointer; font-size: 11px; margin-top: 5px;font-weight: bold; }

/* ================= 右侧：图表占位符 ================= */
.empty-chart-placeholder {
  height: 100%; min-height: 300px;
  display: flex; flex-direction: column; justify-content: center; align-items: center;
  background: #fafafa; border: 2px dashed #e4e7ed; border-radius: 8px; color: #909399;
}
.empty-chart-placeholder .icon { font-size: 48px; margin-bottom: 16px; opacity: 0.5; }

/* 最终报告视图容器限制宽度以适合阅读 */
.final-report-layout { padding: 40px; display: flex; justify-content: center; overflow-y: auto; height: calc(100vh - 160px); }
.report-container { max-width: 900px; width: 100%; }
.insight-citation {
  color: #e6a23c;
  background-color: #fdf6ec;
  border: 1px solid #f3d19e;
}
.insight-citation:hover {
  background-color: #e6a23c;
  color: #ffffff;
  box-shadow: 0 2px 4px rgba(230, 162, 60, 0.3);
}
/* --- 地图专属选项卡 Header --- */
.map-tabs-header {
  display: flex;
  align-items: center;
  padding-bottom: 12px;
  /* 去掉 justify-content，交由子元素分配空间 */
}

.map-title {
  flex: 1; /* 🌟 核心 1：让标题占据左侧所有剩余空间 */
  font-weight: 600;
  color: #303133;
}

.map-tabs {
  display: flex;
  gap: 6px;
  background: #f0f2f5;
  padding: 4px;
  border-radius: 6px;
  /* Tabs 自身保持内容宽度，不设置 flex: 1 */
}

/* 🌟 核心 2：在右侧生成一个隐形的占位块，分配与标题相等的空间 */
.map-tabs-header::after {
  content: '';
  flex: 1;
}

.map-tab-btn {
  background: transparent;
  border: none;
  padding: 6px 14px;
  border-radius: 4px;
  font-size: 0.85rem;
  color: #606266;
  cursor: pointer;
  transition: all 0.2s;
  font-weight: 500;
}

.map-tab-btn:hover {
  color: #409EFF;
}

.map-tab-btn.active {
  background: #ffffff;
  color: #409EFF;
  box-shadow: 0 2px 6px rgba(0,0,0,0.06);
  font-weight: bold;
}
.export-pdf-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
  color: #303133;
  border: 1px solid #dcdfe6;
  padding: 6px 14px;
  border-radius: 20px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
  margin-right: 12px; /* 和右侧的历史记录拉开一点距离 */
}

.export-pdf-btn:hover:not(:disabled) {
  background: #fff;
  border-color: #409EFF;
  color: #409EFF;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(64, 158, 255, 0.2);
}

.export-pdf-btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
  background: #f5f7fa;
  color: #909399;
}
</style>