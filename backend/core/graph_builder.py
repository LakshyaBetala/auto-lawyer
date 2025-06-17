import logging
import json
import time
from typing import Dict, List, Optional, Any, Union, Callable, TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.graph.message import add_messages
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.prebuilt import ToolNode
except ImportError:
    # Fallback for when LangGraph is not available
    class StateGraph:
        def __init__(self, state_schema): pass
        def add_node(self, name, func): pass
        def add_edge(self, from_node, to_node): pass
        def add_conditional_edges(self, from_node, condition, mapping): pass
        def compile(self, **kwargs): pass
    
    END = "END"
    START = "START"
    SqliteSaver = None
    ToolNode = None
    add_messages = lambda x, y: x + y

from backend.config import get_config
from backend.agents.researcher import ResearchResult
from backend.agents.drafter import DraftResult
from backend.agents.summarizer import SummaryResult

# Configure logging
logger = logging.getLogger(__name__)

class NodeType(Enum):
    """Types of nodes in the workflow graph"""
    AGENT = "agent"               # Execute agent action
    DECISION = "decision"         # Decision/branching logic
    AGGREGATOR = "aggregator"     # Combine multiple results
    VALIDATOR = "validator"       # Validate results/conditions
    TRANSFORMER = "transformer"   # Transform data between agents
    HUMAN_INPUT = "human_input"   # Require human intervention
    CONDITIONAL = "conditional"   # Conditional execution
    PARALLEL = "parallel"         # Parallel execution coordinator

class WorkflowState(TypedDict):
    """State that flows through the workflow graph"""
    # Core workflow data
    workflow_id: str
    current_step: str
    step_count: int
    
    # Input/output data
    original_query: str
    research_results: List[ResearchResult]
    draft_results: List[DraftResult]
    summary_results: List[SummaryResult]
    
    # Execution metadata
    started_at: str
    intermediate_results: Dict[str, Any]
    errors: List[Dict[str, Any]]
    warnings: List[str]
    
    # Decision tracking
    decisions_made: Dict[str, Any]
    confidence_scores: Dict[str, float]
    
    # Human feedback
    human_feedback: List[Dict[str, Any]]
    
    # Final outputs
    final_recommendation: Optional[str]
    next_steps: List[str]

@dataclass
class GraphNode:
    """Definition of a node in the workflow graph"""
    node_id: str
    node_type: NodeType
    name: str
    description: str
    
    # Execution configuration
    agent_id: Optional[str] = None
    action: Optional[str] = None
    function: Optional[Callable] = None
    
    # Input/output specification
    input_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    required_inputs: List[str] = field(default_factory=list)
    
    # Conditional logic
    condition_function: Optional[Callable] = None
    success_paths: List[str] = field(default_factory=list)
    failure_paths: List[str] = field(default_factory=list)
    
    # Error handling
    retry_count: int = 0
    max_retries: int = 2
    timeout_seconds: int = 300
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "description": self.description,
            "agent_id": self.agent_id,
            "action": self.action,
            "input_keys": self.input_keys,
            "output_keys": self.output_keys,
            "required_inputs": self.required_inputs,
            "success_paths": self.success_paths,
            "failure_paths": self.failure_paths,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds
        }

@dataclass
class GraphEdge:
    """Definition of an edge in the workflow graph"""
    edge_id: str
    from_node: str
    to_node: str
    condition: Optional[Callable] = None
    condition_description: str = ""
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "edge_id": self.edge_id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "condition_description": self.condition_description,
            "weight": self.weight
        }

class WorkflowGraphBuilder:
    """
    LangGraph-based workflow builder for complex multi-agent orchestration
    Creates sophisticated DAGs with decision points, parallel execution, and human-in-the-loop
    """
    
    def __init__(self):
        """Initialize the Graph Builder"""
        self.config = get_config()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.compiled_graphs: Dict[str, Any] = {}
        
        # State management
        self.checkpointer = None
        if SqliteSaver:
            try:
                self.checkpointer = SqliteSaver.from_conn_string("workflow_checkpoints.db")
            except Exception as e:
                logger.warning(f"Could not initialize checkpointer: {e}")
        
        logger.info("Workflow Graph Builder initialized")
    
    def create_graph(self, graph_id: str, description: str = "") -> StateGraph:
        """
        Create a new workflow graph
        
        Args:
            graph_id: Unique identifier for the graph
            description: Graph description
            
        Returns:
            StateGraph: LangGraph instance
        """
        
        # Create state schema
        graph = StateGraph(WorkflowState)
        
        logger.info(f"Created new workflow graph: {graph_id}")
        return graph
    
    def add_agent_node(self,
                      graph: StateGraph,
                      node_id: str,
                      agent_id: str,
                      action: str,
                      name: str = None,
                      description: str = "",
                      **kwargs) -> GraphNode:
        """
        Add an agent execution node to the graph
        
        Args:
            graph: StateGraph to add node to
            node_id: Unique node identifier
            agent_id: ID of agent to execute
            action: Action method to call
            name: Human-readable name
            description: Node description
            **kwargs: Additional node configuration
            
        Returns:
            GraphNode: Created node definition
        """
        
        name = name or f"{agent_id}_{action}"
        
        node = GraphNode(
            node_id=node_id,
            node_type=NodeType.AGENT,
            name=name,
            description=description,
            agent_id=agent_id,
            action=action,
            **kwargs
        )
        
        # Create node function
        def agent_node_function(state: WorkflowState) -> WorkflowState:
            return self._execute_agent_node(state, node)
        
        # Add to graph
        graph.add_node(node_id, agent_node_function)
        self.nodes[node_id] = node
        
        logger.info(f"Added agent node: {node_id} ({agent_id}.{action})")
        return node
    
    def add_decision_node(self,
                         graph: StateGraph,
                         node_id: str,
                         condition_function: Callable,
                         name: str = None,
                         description: str = "",
                         **kwargs) -> GraphNode:
        """
        Add a decision/branching node to the graph
        
        Args:
            graph: StateGraph to add node to
            node_id: Unique node identifier
            condition_function: Function that returns next node ID
            name: Human-readable name
            description: Node description
            **kwargs: Additional node configuration
            
        Returns:
            GraphNode: Created node definition
        """
        
        name = name or f"decision_{node_id}"
        
        node = GraphNode(
            node_id=node_id,
            node_type=NodeType.DECISION,
            name=name,
            description=description,
            condition_function=condition_function,
            **kwargs
        )
        
        # Create node function
        def decision_node_function(state: WorkflowState) -> WorkflowState:
            return self._execute_decision_node(state, node)
        
        # Add to graph
        graph.add_node(node_id, decision_node_function)
        self.nodes[node_id] = node
        
        logger.info(f"Added decision node: {node_id}")
        return node
    
    def add_aggregator_node(self,
                           graph: StateGraph,
                           node_id: str,
                           aggregation_function: Callable,
                           name: str = None,
                           description: str = "",
                           **kwargs) -> GraphNode:
        """
        Add an aggregator node that combines multiple results
        
        Args:
            graph: StateGraph to add node to
            node_id: Unique node identifier
            aggregation_function: Function to combine results
            name: Human-readable name
            description: Node description
            **kwargs: Additional node configuration
            
        Returns:
            GraphNode: Created node definition
        """
        
        name = name or f"aggregator_{node_id}"
        
        node = GraphNode(
            node_id=node_id,
            node_type=NodeType.AGGREGATOR,
            name=name,
            description=description,
            function=aggregation_function,
            **kwargs
        )
        
        # Create node function
        def aggregator_node_function(state: WorkflowState) -> WorkflowState:
            return self._execute_aggregator_node(state, node)
        
        # Add to graph
        graph.add_node(node_id, aggregator_node_function)
        self.nodes[node_id] = node
        
        logger.info(f"Added aggregator node: {node_id}")
        return node
    
    def add_human_input_node(self,
                            graph: StateGraph,
                            node_id: str,
                            prompt: str,
                            input_schema: Dict[str, Any] = None,
                            name: str = None,
                            description: str = "",
                            **kwargs) -> GraphNode:
        """
        Add a human input node for human-in-the-loop workflows
        
        Args:
            graph: StateGraph to add node to
            node_id: Unique node identifier
            prompt: Prompt to show to human
            input_schema: Expected input format
            name: Human-readable name
            description: Node description
            **kwargs: Additional node configuration
            
        Returns:
            GraphNode: Created node definition
        """
        
        name = name or f"human_input_{node_id}"
        
        node = GraphNode(
            node_id=node_id,
            node_type=NodeType.HUMAN_INPUT,
            name=name,
            description=description,
            **kwargs
        )
        
        # Create node function
        def human_input_node_function(state: WorkflowState) -> WorkflowState:
            return self._execute_human_input_node(state, node, prompt, input_schema)
        
        # Add to graph
        graph.add_node(node_id, human_input_node_function)
        self.nodes[node_id] = node
        
        logger.info(f"Added human input node: {node_id}")
        return node
    
    def add_edge(self,
                graph: StateGraph,
                from_node: str,
                to_node: str,
                condition: Optional[Callable] = None,
                condition_description: str = "") -> GraphEdge:
        """
        Add an edge between two nodes
        
        Args:
            graph: StateGraph to add edge to
            from_node: Source node ID
            to_node: Target node ID
            condition: Optional condition function
            condition_description: Description of condition
            
        Returns:
            GraphEdge: Created edge definition
        """
        
        edge_id = f"{from_node}_to_{to_node}"
        
        edge = GraphEdge(
            edge_id=edge_id,
            from_node=from_node,
            to_node=to_node,
            condition=condition,
            condition_description=condition_description
        )
        
        # Add to graph
        if condition:
            # Conditional edge
            condition_mapping = {True: to_node, False: END}
            graph.add_conditional_edges(from_node, condition, condition_mapping)
        else:
            # Direct edge
            graph.add_edge(from_node, to_node)
        
        self.edges.append(edge)
        
        logger.info(f"Added edge: {from_node} -> {to_node}")
        return edge
    
    def add_conditional_edges(self,
                             graph: StateGraph,
                             from_node: str,
                             condition_function: Callable,
                             path_mapping: Dict[str, str]) -> List[GraphEdge]:
        """
        Add conditional edges with multiple paths
        
        Args:
            graph: StateGraph to add edges to
            from_node: Source node ID
            condition_function: Function that returns path key
            path_mapping: Mapping of condition results to node IDs
            
        Returns:
            List[GraphEdge]: Created edge definitions
        """
        
        edges = []
        
        # Add conditional edges to graph
        graph.add_conditional_edges(from_node, condition_function, path_mapping)
        
        # Create edge definitions for tracking
        for condition_result, to_node in path_mapping.items():
            edge_id = f"{from_node}_to_{to_node}_if_{condition_result}"
            
            edge = GraphEdge(
                edge_id=edge_id,
                from_node=from_node,
                to_node=to_node,
                condition=condition_function,
                condition_description=f"If condition returns '{condition_result}'"
            )
            
            edges.append(edge)
            self.edges.append(edge)
        
        logger.info(f"Added conditional edges from {from_node} to {len(path_mapping)} paths")
        return edges
    
    def _execute_agent_node(self, state: WorkflowState, node: GraphNode) -> WorkflowState:
        """Execute an agent node"""
        
        logger.info(f"Executing agent node: {node.node_id}")
        
        try:
            # Import here to avoid circular imports
            from backend.core.agent_executor import get_executor
            
            executor = get_executor()
            
            # Prepare inputs from state
            inputs = self._extract_inputs_from_state(state, node)
            
            # Execute agent
            result = executor.execute_sync(
                agent_id=node.agent_id,
                action=node.action,
                inputs=inputs,
                timeout=node.timeout_seconds
            )
            
            # Update state with results
            state = self._update_state_with_result(state, node, result)
            state["step_count"] += 1
            state["current_step"] = node.node_id
            
            logger.info(f"Agent node {node.node_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Agent node {node.node_id} failed: {e}")
            
            # Add error to state
            error_info = {
                "node_id": node.node_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            state["errors"].append(error_info)
            
            # Handle retries
            if node.retry_count < node.max_retries:
                node.retry_count += 1
                logger.info(f"Retrying node {node.node_id} ({node.retry_count}/{node.max_retries})")
                return self._execute_agent_node(state, node)
        
        return state
    
    def _execute_decision_node(self, state: WorkflowState, node: GraphNode) -> WorkflowState:
        """Execute a decision node"""
        
        logger.info(f"Executing decision node: {node.node_id}")
        
        try:
            # Execute condition function
            decision_result = node.condition_function(state)
            
            # Record decision
            state["decisions_made"][node.node_id] = {
                "result": decision_result,
                "timestamp": datetime.now().isoformat()
            }
            
            state["step_count"] += 1
            state["current_step"] = node.node_id
            
            logger.info(f"Decision node {node.node_id} decided: {decision_result}")
            
        except Exception as e:
            logger.error(f"Decision node {node.node_id} failed: {e}")
            
            error_info = {
                "node_id": node.node_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            state["errors"].append(error_info)
        
        return state
    
    def _execute_aggregator_node(self, state: WorkflowState, node: GraphNode) -> WorkflowState:
        """Execute an aggregator node"""
        
        logger.info(f"Executing aggregator node: {node.node_id}")
        
        try:
            # Execute aggregation function
            aggregated_result = node.function(state)
            
            # Update state
            state["intermediate_results"][node.node_id] = aggregated_result
            state["step_count"] += 1
            state["current_step"] = node.node_id
            
            logger.info(f"Aggregator node {node.node_id} completed")
            
        except Exception as e:
            logger.error(f"Aggregator node {node.node_id} failed: {e}")
            
            error_info = {
                "node_id": node.node_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            state["errors"].append(error_info)
        
        return state
    
    def _execute_human_input_node(self,
                                 state: WorkflowState,
                                 node: GraphNode,
                                 prompt: str,
                                 input_schema: Dict[str, Any]) -> WorkflowState:
        """Execute a human input node"""
        
        logger.info(f"Executing human input node: {node.node_id}")
        
        # In a real implementation, this would pause and wait for human input
        # For now, we'll simulate or log the requirement
        
        human_input_request = {
            "node_id": node.node_id,
            "prompt": prompt,
            "input_schema": input_schema,
            "timestamp": datetime.now().isoformat(),
            "status": "waiting_for_input"
        }
        
        state["human_feedback"].append(human_input_request)
        state["step_count"] += 1
        state["current_step"] = node.node_id
        
        logger.info(f"Human input requested for node {node.node_id}")
        
        return state
    
    def _extract_inputs_from_state(self, state: WorkflowState, node: GraphNode) -> Dict[str, Any]:
        """Extract relevant inputs for a node from the workflow state"""
        
        inputs = {}
        
        # Add required inputs
        for key in node.input_keys:
            if key in state:
                inputs[key] = state[key]
            elif key in state["intermediate_results"]:
                inputs[key] = state["intermediate_results"][key]
        
        # Add agent-specific inputs
        if node.agent_id == "researcher":
            if "original_query" in state:
                inputs["query"] = state["original_query"]
        
        elif node.agent_id == "drafter":
            if state["research_results"]:
                inputs["research_results"] = state["research_results"]
        
        elif node.agent_id == "summarizer":
            if state["research_results"] or state["draft_results"]:
                inputs["research_results"] = state["research_results"]
                inputs["draft_results"] = state["draft_results"]
        
        return inputs
    
    def _update_state_with_result(self,
                                 state: WorkflowState,
                                 node: GraphNode,
                                 result: Any) -> WorkflowState:
        """Update workflow state with node execution result"""
        
        # Store in intermediate results
        state["intermediate_results"][node.node_id] = result
        
        # Update specific result collections based on agent type
        if node.agent_id == "researcher" and hasattr(result, 'result'):
            if isinstance(result.result, list):
                state["research_results"].extend(result.result)
            else:
                state["research_results"].append(result.result)
        
        elif node.agent_id == "drafter" and hasattr(result, 'result'):
            if isinstance(result.result, list):
                state["draft_results"].extend(result.result)
            else:
                state["draft_results"].append(result.result)
        
        elif node.agent_id == "summarizer" and hasattr(result, 'result'):
            if isinstance(result.result, list):
                state["summary_results"].extend(result.result)
            else:
                state["summary_results"].append(result.result)
        
        # Update confidence scores
        if hasattr(result, 'result') and hasattr(result.result, 'confidence_score'):
            state["confidence_scores"][node.node_id] = result.result.confidence_score
        
        return state
    
    def compile_graph(self, graph: StateGraph, graph_id: str) -> Any:
        """
        Compile the graph for execution
        
        Args:
            graph: StateGraph to compile
            graph_id: Unique identifier for the compiled graph
            
        Returns:
            Compiled graph ready for execution
        """
        
        try:
            # Compile with checkpointer if available
            if self.checkpointer:
                compiled = graph.compile(checkpointer=self.checkpointer)
            else:
                compiled = graph.compile()
            
            self.compiled_graphs[graph_id] = compiled
            
            logger.info(f"Graph {graph_id} compiled successfully")
            return compiled
            
        except Exception as e:
            logger.error(f"Failed to compile graph {graph_id}: {e}")
            raise
    
    def create_standard_legal_workflow(self) -> str:
        """
        Create a standard legal analysis workflow
        
        Returns:
            str: Graph ID of the created workflow
        """
        
        graph_id = f"legal_workflow_{int(time.time())}"
        graph = self.create_graph(graph_id, "Standard Legal Analysis Workflow")
        
        # Add nodes
        research_node = self.add_agent_node(
            graph, "research", "researcher", "research",
            name="Legal Research",
            description="Conduct comprehensive legal research"
        )
        
        draft_node = self.add_agent_node(
            graph, "draft", "drafter", "draft_document",
            name="Document Drafting",
            description="Draft legal document based on research"
        )
        
        summarize_node = self.add_agent_node(
            graph, "summarize", "summarizer", "summarize",
            name="Summary Creation",
            description="Create executive summary"
        )
        
        # Add quality check decision node
        def quality_check(state: WorkflowState) -> str:
            """Check if research quality is sufficient"""
            if state["research_results"]:
                avg_confidence = sum(
                    r.confidence_score for r in state["research_results"] 
                    if hasattr(r, 'confidence_score')
                ) / len(state["research_results"])
                
                return "proceed" if avg_confidence > 0.7 else "additional_research"
            return "additional_research"
        
        quality_node = self.add_decision_node(
            graph, "quality_check", quality_check,
            name="Quality Assessment",
            description="Assess research quality and decide next steps"
        )
        
        # Add edges
        graph.add_edge(START, "research")
        
        self.add_conditional_edges(
            graph, "research", quality_check,
            {"proceed": "draft", "additional_research": "research"}
        )
        
        graph.add_edge("draft", "summarize")
        graph.add_edge("summarize", END)
        
        # Compile graph
        compiled = self.compile_graph(graph, graph_id)
        
        logger.info(f"Created standard legal workflow: {graph_id}")
        return graph_id
    
    def execute_graph(self,
                     graph_id: str,
                     initial_state: WorkflowState,
                     config: Dict[str, Any] = None) -> WorkflowState:
        """
        Execute a compiled graph
        
        Args:
            graph_id: ID of graph to execute
            initial_state: Initial workflow state
            config: Execution configuration
            
        Returns:
            WorkflowState: Final state after execution
        """
        
        if graph_id not in self.compiled_graphs:
            raise ValueError(f"Graph {graph_id} not found or not compiled")
        
        compiled_graph = self.compiled_graphs[graph_id]
        config = config or {}
        
        logger.info(f"Executing graph: {graph_id}")
        
        try:
            # Execute the graph
            result = compiled_graph.invoke(initial_state, config)
            
            logger.info(f"Graph {graph_id} executed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Graph {graph_id} execution failed: {e}")
            raise
    
    def get_graph_status(self, graph_id: str) -> Optional[Dict[str, Any]]:
        """Get status information about a graph"""
        
        if graph_id not in self.compiled_graphs:
            return None
        
        node_count = len([n for n in self.nodes.values() if n.node_id.startswith(graph_id)])
        edge_count = len([e for e in self.edges if e.edge_id.startswith(graph_id)])
        
        return {
            "graph_id": graph_id,
            "compiled": True,
            "node_count": node_count,
            "edge_count": edge_count,
            "nodes": [n.to_dict() for n in self.nodes.values() if n.node_id.startswith(graph_id)],
            "edges": [e.to_dict() for e in self.edges if e.edge_id.startswith(graph_id)]
        }

# Global graph builder instance
_graph_builder: Optional[WorkflowGraphBuilder] = None

def get_graph_builder() -> WorkflowGraphBuilder:
    """Get or create the global graph builder instance"""
    global _graph_builder
    if _graph_builder is None:
        _graph_builder = WorkflowGraphBuilder()
    return _graph_builder

# Convenience functions for common workflow patterns
def create_legal_research_workflow() -> str:
    """Create a comprehensive legal research workflow"""
    builder = get_graph_builder()
    return builder.create_standard_legal_workflow()

def create_human_in_loop_workflow() -> str:
    """Create a workflow with human intervention points"""
    builder = get_graph_builder()
    
    graph_id = f"human_loop_workflow_{int(time.time())}"
    graph = builder.create_graph(graph_id, "Human-in-the-Loop Legal Workflow")
    
    # Research phase
    builder.add_agent_node(graph, "initial_research", "researcher", "research")
    
    # Human review
    builder.add_human_input_node(
        graph, "human_review", 
        "Please review the research findings and provide additional direction",
        {"feedback": "string", "additional_queries": "list"}
    )
    
    # Additional research based on human input
    builder.add_agent_node(graph, "additional_research", "researcher", "research")
    
    # Final drafting
    builder.add_agent_node(graph, "final_draft", "drafter", "draft_document")
    
    # Add edges
    graph.add_edge(START, "initial_research")
    graph.add_edge("initial_research", "human_review")
    graph.add_edge("human_review", "additional_research")
    graph.add_edge("additional_research", "final_draft")
    graph.add_edge("final_draft", END)
    
    builder.compile_graph(graph, graph_id)
    return graph_id