// src/store/chatStore.js
import { defineStore } from 'pinia'
import {computed, ref} from 'vue'
import { streamChat, streamFeedback } from '../api/chat'

export const useChatStore = defineStore('chat', () => {
  // 模拟多用户环境：实际业务中 user_id 可以从登录态获取
  const userId = ref('ocean_researcher_01')
  const sessionId = ref('session_' + Date.now())

  const messages = ref([
    { id: 1, role: 'ai', content: '您好！后端连接已就绪。请输入您想探索的事件或实体。' }
  ])

  const isGenerating = ref(false)
  const hitlState = ref({
    isWaiting: false,
    plan: null
  })

  // 新增：用于存放最终传给可视化组件（右侧面板）的数据
  const analysisResults = ref({
    insight: {
      title: "等待数据...",
      summary: "请在左侧发起查询...",
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
        lastMsg.content += `\n> 正在执行阶段: ${data.node}...`
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
      analysisResults.value = data.results
      messages.value.push({
        id: Date.now(),
        role: 'ai',
        content: '分析已完成！请查看右侧的可视化面板与洞察结果。'
      })
    }
    else if (eventName === 'error') {
      isGenerating.value = false
      messages.value.push({
        id: Date.now(),
        role: 'ai',
        content: `【系统错误】: ${data.error}`
      })
    }
  }

  // 发送初始查询
  const sendMessage = async (text) => {
    if (!text.trim() || isGenerating.value) return

    messages.value.push({ id: Date.now(), role: 'user', content: text })
    isGenerating.value = true

    // 预置一条空 AI 消息，用于承载后续的 node_progress 状态更新
    messages.value.push({ id: Date.now() + 1, role: 'ai', content: '收到请求，系统正在编排...\n' })

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
      console.error('请求发起失败', error)
      isGenerating.value = false
    }
  }

  // 发送用户的审批决定
  const sendFeedback = async (feedbackText) => {
    hitlState.value.isWaiting = false

    const isApprove = feedbackText === 'approve'
    messages.value.push({
      id: Date.now(),
      role: 'user',
      content: isApprove ? '【系统提示】确认执行上述计划。' : `【修改意见】${feedbackText}`
    })

    isGenerating.value = true
    messages.value.push({ id: Date.now() + 1, role: 'ai', content: '收到反馈，正在继续执行分析...\n' })

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
      console.error('反馈发送失败', error)
      isGenerating.value = false
    }
  }

  return { sessionId, messages, isGenerating, hitlState, analysisResults, sendMessage, sendFeedback, evidenceState, openEvidence, closeEvidence }
})

