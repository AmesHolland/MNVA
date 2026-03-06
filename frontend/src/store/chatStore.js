// src/store/chatStore.js
import { defineStore } from 'pinia'
import {computed, ref} from 'vue'
import { streamChat, streamFeedback } from '../api/chat'

export const useChatStore = defineStore('chat', () => {
  // 模拟多用户环境：实际业务中 user_id 可以从登录态获取
  const userId = ref('ocean_researcher_01')
  const sessionId = ref('session_' + Date.now())

  const messages = ref([
    { id: 1, role: 'ai', content: 'Hello! The system is ready. Please enter the event or entity you want to explore.' }
  ])

  const isGenerating = ref(false)
  const hitlState = ref({
    isWaiting: false,
    plan: null
  })

  // 新增：用于存放最终传给可视化组件（右侧面板）的数据
  const analysisResults = ref({
    insight: {
      title: "Waiting for data...",
      summary: "Please initiate a query on the left side...",
      keywords: []
    },
    visualizations: []
  })

  // 溯源抽屉状态管理
  const evidenceState = ref({
    isOpen: false,
    activeSourceIds: []
  })

  // 打开溯源抽屉
  const openEvidence = (sourceIds) => {
    if (!sourceIds || sourceIds.length === 0) return
    evidenceState.value.activeSourceIds = sourceIds
    evidenceState.value.isOpen = true
  }

  // 关闭溯源抽屉
  const closeEvidence = () => {
    evidenceState.value.isOpen = false
    setTimeout(() => {
      evidenceState.value.activeSourceIds = []
    }, 300) // 等待抽屉收回动画结束后清空数据
  }

  // 【新增】：全局软刷选状态 (Soft Brushing State)
  const brushState = ref({
    timeRange: null, // 格式: [startTimestamp, endTimestamp] (毫秒级时间戳)
    spatialLabels: [] // 当前激活的空间标签，例如: ['南海', '菲律宾']
  })

  // 【新增】：更新刷选状态的 Action
  const updateBrushState = (timeRange, spatialLabels = null) => {
    brushState.value.timeRange = timeRange
    if (spatialLabels !== null) {
      brushState.value.spatialLabels = spatialLabels
    }
  }

  // 【新增】：清除刷选状态 (恢复全局高亮)
  const clearBrushState = () => {
    brushState.value.timeRange = null
    brushState.value.spatialLabels = []
  }
  // src/store/chatStore.js
  // const analysisResults = ref(null) // 存放后端发来的 integrated_payload
  const activeSectionIndex = ref(0) // 记录当前高亮的报告章节索引

  // 计算属性：根据当前激活的章节，自动筛选出需要渲染的 task 数据
  const activeTasks = computed(() => {
    if (!analysisResults.value || !analysisResults.value.report) return []

    const currentSection = analysisResults.value.report.sections[activeSectionIndex.value]
    if (!currentSection || !currentSection.ref_task_ids) return []

    // 根据 ref_task_ids 从 tasks 字典中取出数据
    return currentSection.ref_task_ids.map(id => analysisResults.value.tasks[id])
  })

  // 统一处理 SSE 返回的事件流
  const handleSSEMessage = (ev) => {
    const eventName = ev.event
    const data = JSON.parse(ev.data)
    console.log(data)
    if (eventName === 'node_progress') {
      // 当 Agent 正在不同节点间流转时，追加状态到最后一条 AI 消息中
      const lastMsg = messages.value[messages.value.length - 1]
      if (lastMsg && lastMsg.role === 'ai') {
        lastMsg.content += `\n> ${data.node}...`
      }
    }
    else if (eventName === 'interrupt') {
      // 触发 HITL 审批卡片
      hitlState.value = {
        isWaiting: true,
        plan: data.plan
      }
      isGenerating.value = false
    }
    else if (eventName === 'completed') {
      // 流程全部跑完
      isGenerating.value = false
      // 可以暂时塞进 store 里用于测试 analysisResults.value = data.results // 将结果存入 store 供右侧图表监听
      // ====== 【修改部分开始】 ======
      const isNewVisualResult = data.is_new_visual_result;

      if (isNewVisualResult) {
        // Slow branch: A complete in-depth analysis was run in this round, update the charts on the right
        analysisResults.value = data.results
        messages.value.push({
          id: Date.now(),
          role: 'ai',
          content: 'Analysis completed! Please check the visualization panel and insight results on the right side.'
        })
      } else {
        // Fast branch: Only casual Q&A in this round
        // 【Key point】: Do not touch analysisResults.value at all here, so the original visualization charts on the right will be perfectly preserved!
        messages.value.push({
          id: Date.now(),
          role: 'ai',
          content: data.direct_answer || 'Answer completed.'
        })
      }
    }
    else if (eventName === 'error') {
      isGenerating.value = false
      messages.value.push({
        id: Date.now(),
        role: 'ai',
        content: `【System Error】: ${data.error}`
      })
    }
  }

  // 发送初始查询
  const sendMessage = async (text) => {
    if (!text.trim() || isGenerating.value) return

    messages.value.push({ id: Date.now(), role: 'user', content: text })
    isGenerating.value = true

    // 预置一条空 AI 消息，用于承载后续的 node_progress 状态更新
    messages.value.push({ id: Date.now() + 1, role: 'ai', content: 'Request received, the system is orchestrating...\n' })
    try {
      await streamChat({
        user_id: userId.value,
        session_id: sessionId.value,
        query: text
      }, {
        onMessage: handleSSEMessage,
        onError: (err) => {
          console.error('SSE Error:', err)
          isGenerating.value = false
        }
      })
    } catch (error) {
      console.error('Failed to initiate request', error)
      isGenerating.value = false
    }
  }

  // Send user's approval decision
  const sendFeedback = async (feedbackText) => {
    hitlState.value.isWaiting = false

    const isApprove = feedbackText === 'approve'
    messages.value.push({
      id: Date.now(),
      role: 'user',
      content: isApprove ? '【System Prompt】Confirm execution of the above plan.' : `【Modification Suggestions】${feedbackText}`
    })

    isGenerating.value = true
    messages.value.push({ id: Date.now() + 1, role: 'ai', content: 'Feedback received, continuing analysis...\n' })

    try {
      await streamFeedback({
        user_id: userId.value,
        session_id: sessionId.value,
        feedback: feedbackText
      }, {
        onMessage: handleSSEMessage,
        onError: (err) => {
          console.error('SSE Error:', err)
          isGenerating.value = false
        }
      })
    } catch (error) {
      console.error('Failed to send feedback', error)
      isGenerating.value = false
    }
  }

  return { sessionId, messages, isGenerating, hitlState, analysisResults, sendMessage, sendFeedback, evidenceState, openEvidence, closeEvidence,
  brushState, updateBrushState, clearBrushState}
})
