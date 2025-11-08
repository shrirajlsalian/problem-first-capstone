import logging
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from shared.schemas import PolicyState
from shared.tools import Tools
from iter3.graph_iter3 import node_apply_labels_and_report
from iter4.agents_iter4 import (
    PlannerAgent,
    RetrievalAgent,
    RemediationAgent,
    HitlSupervisorAgent,
)

logger = logging.getLogger(__name__)


def _plan_router(state: Dict[str, Any]) -> str:
    plan = state.get("plan", {})
    if plan.get("strategy") == "remediation-only":
        return "remediate"
    return "analyze"


def _wrap(agent):
    def inner(state: Dict[str, Any]) -> Dict[str, Any]:
        return agent(state)
    return inner


def build_graph(
    tools: Tools,
    *,
    use_interrupt: bool = True,
    unattended: bool = False,
    mcp_endpoint: str | None = None,
) -> StateGraph:
    """
    Happy path: compile the Iteration 4 graph with planner, retrieval, remediation, and HITL agents.
    """
    planner = PlannerAgent(tools)
    retriever = RetrievalAgent(tools)
    remediator = RemediationAgent(tools)
    supervisor = HitlSupervisorAgent(
        tools=tools,
        use_interrupt=use_interrupt,
        unattended=unattended,
        mcp_endpoint=mcp_endpoint,
    )

    graph = StateGraph(PolicyState)
    graph.add_node("plan", _wrap(planner))
    graph.add_node("analyze", _wrap(retriever))
    graph.add_node("remediate", _wrap(remediator))
    graph.add_node("hitl", _wrap(supervisor))
    graph.add_node("report", node_apply_labels_and_report)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", _plan_router, {"analyze": "analyze", "remediate": "remediate"})
    graph.add_edge("analyze", "remediate")
    graph.add_edge("remediate", "hitl")
    graph.add_edge("hitl", "report")
    graph.add_edge("report", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


