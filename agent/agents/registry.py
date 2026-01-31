from __future__ import annotations
from typing import Dict

from agent.agents.sub_agent.search_agent import SearchAgent
# 你后续补：TimeEvolutionAgent / TopicAnalysis1 / EntityTrackingAgent / TopicAnalysis2

def build_agent_registry() -> Dict[str, object]:
    return {
        "SearchAgent": SearchAgent(),
        # "Time-Evolution-Agent": TimeEvolutionAgent(),
        # "Topic-Analysis-1": TopicAnalysis1(),
        # "Entity-Tracking-Agent": EntityTrackingAgent(),
        # "Topic-Analysis-2": TopicAnalysis2(),
    }
