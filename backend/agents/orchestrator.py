"""
Master Orchestrator — LangGraph StateGraph
===========================================
Wires Agents 1, 2, and 3 into an explicit state machine using LangGraph.

Graph:
  START → agent1_node → agent2_node → agent3_node → END

Each node is a thin wrapper that:
  - Checks whether an upstream error occurred.
  - Calls the corresponding agent function.
  - Returns the updated state.

The `db` session is injected via a closure so that Agent 3 can write to the DB.
"""
import logging
from sqlalchemy.orm import Session

from langgraph.graph import StateGraph, END

from backend.agents.state import DiagnosticState
from backend.agents.agent1_medical_image import run_agent1

logger = logging.getLogger(__name__)


def build_graph(db: Session) -> StateGraph:
    """
    Construct the LangGraph StateGraph for the diagnostic pipeline.

    The `db` session is captured via closure and passed to `agent3_node`,
    which is the only agent that writes to the database.

    Args:
        db: SQLAlchemy database session (provided per-request by FastAPI).

    Returns:
        A compiled LangGraph application (callable with initial state dict).
    """

    def agent1_node(state: DiagnosticState) -> DiagnosticState:
        """LangGraph node wrapping Agent 1 — Medical Image Agent."""
        logger.info("[Orchestrator] Entering Agent 1 node.")
        return run_agent1(state)

    def agent3_node(state: DiagnosticState) -> DiagnosticState:
        """LangGraph node wrapping Agent 3 — Report Agent."""
        logger.info("[Orchestrator] Entering Agent 3 node.")
        from backend.agents.agent3_report import run_agent3
        return run_agent3(state, db)

    # Build graph
    workflow = StateGraph(DiagnosticState)

    workflow.add_node("agent1", agent1_node)
    workflow.add_node("agent3", agent3_node)
    workflow.set_entry_point("agent1")
    workflow.add_edge("agent1", "agent3")
    workflow.add_edge("agent3", END)
    
    return workflow.compile()


def run_diagnostic_pipeline(initial_state: DiagnosticState, db: Session) -> DiagnosticState:
    """
    Execute the full diagnostic pipeline by invoking the LangGraph app.

    Args:
        initial_state: Populated DiagnosticState with at least:
                       session_id, image_bytes, image_path, patient_id,
                       technician_id, physician_id, modality, patient_name,
                       patient_dob, patient_gender.
        db:            SQLAlchemy session for Agent 3's DB write.

    Returns:
        Final DiagnosticState with all Agent outputs populated.
    """
    app = build_graph(db)

    logger.info("[Orchestrator] Pipeline starting for session %s",
                initial_state.get("session_id"))

    final_state: DiagnosticState = app.invoke(initial_state)

    logger.info("[Orchestrator] Pipeline finished. Status: %s",
                final_state.get("status"))
    return final_state
