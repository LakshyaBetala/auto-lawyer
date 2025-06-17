import logging
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, Future, as_completed, TimeoutError as FutureTimeoutError
import json
import traceback
import queue
from contextlib import contextmanager
import signal

from backend.config import get_config
from backend.local_llm.llm_runner import get_llm_runner

# Configure logging
logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """Agent execution modes"""
    SEQUENTIAL = "sequential"      # Execute one at a time
    PARALLEL = "parallel"          # Execute multiple agents simultaneously
    PIPELINE = "pipeline"          # Execute in pipeline fashion
    ADAPTIVE = "adaptive"          # Automatically choose best mode

class ExecutionPriority(Enum):
    """Execution priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class ExecutionStatus(Enum):
    """Individual execution status"""
    QUEUED = "queued"
    PREPARING = "preparing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"

@dataclass
class ExecutionContext:
    """Context information for agent execution"""
    execution_id: str
    agent_id: str
    action: str
    inputs: Dict[str, Any]
    
    # Execution parameters
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    timeout_seconds: int = 300
    max_retries: int = 2
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    
    # Environment
    environment_vars: Dict[str, Any] = field(default_factory=dict)
    execution_hints: Dict[str, Any] = field(default_factory=dict)
    
    # Callbacks
    on_start: Optional[Callable] = None
    on_progress: Optional[Callable] = None
    on_complete: Optional[Callable] = None
    on_error: Optional[Callable] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            "execution_id": self.execution_id,
            "agent_id": self.agent_id,
            "action": self.action,
            "priority": self.priority.value,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }

@dataclass
class ExecutionResult:
    """Result of agent execution with comprehensive metadata"""
    execution_id: str
    agent_id: str
    action: str
    status: ExecutionStatus
    
    # Results
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Performance metrics
    execution_time_seconds: float = 0.0
    queue_time_seconds: float = 0.0
    preparation_time_seconds: float = 0.0
    actual_execution_time_seconds: float = 0.0
    
    # Retry information
    retry_count: int = 0
    retry_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    llm_tokens_used: int = 0
    
    # Timestamps
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "execution_id": self.execution_id,
            "agent_id": self.agent_id,
            "action": self.action,
            "status": self.status.value,
            "result": self.result if self.status == ExecutionStatus.COMPLETED else None,
            "error": self.error,
            "error_type": self.error_type,
            "execution_time_seconds": self.execution_time_seconds,
            "queue_time_seconds": self.queue_time_seconds,
            "retry_count": self.retry_count,
            "memory_usage_mb": self.memory_usage_mb,
            "llm_tokens_used": self.llm_tokens_used,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }

class ResourceMonitor:
    """Monitor system resources during execution"""
    
    def __init__(self):
        self.start_time = None
        self.peak_memory = 0.0
        self.cpu_samples = []
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self.start_time = time.time()
        self.peak_memory = 0.0
        self.cpu_samples = []
    
    def sample_resources(self):
        """Sample current resource usage"""
        try:
            import psutil
            process = psutil.Process()
            
            # Memory usage
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.peak_memory = max(self.peak_memory, memory_mb)
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            self.cpu_samples.append(cpu_percent)
            
        except ImportError:
            # psutil not available, use basic monitoring
            pass
        except Exception as e:
            logger.warning(f"Resource monitoring error: {e}")
    
    def get_metrics(self) -> Dict[str, float]:
        """Get resource usage metrics"""
        return {
            "peak_memory_mb": self.peak_memory,
            "average_cpu_percent": sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0,
            "execution_time": time.time() - self.start_time if self.start_time else 0.0
        }

class ExecutionQueue:
    """Priority queue for agent executions"""
    
    def __init__(self, max_size: int = 1000):
        self.queue = queue.PriorityQueue(max_size)
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.completed_executions: Dict[str, ExecutionResult] = {}
        self._lock = threading.Lock()
    
    def enqueue(self, context: ExecutionContext) -> bool:
        """Add execution to queue"""
        try:
            # Priority queue uses (priority, item) tuples
            # Lower number = higher priority
            priority_value = 6 - context.priority.value  # Invert for correct ordering
            
            self.queue.put((priority_value, time.time(), context), block=False)
            
            with self._lock:
                self.active_executions[context.execution_id] = context
            
            logger.info(f"Execution {context.execution_id} queued with priority {context.priority.value}")
            return True
            
        except queue.Full:
            logger.error(f"Execution queue full, cannot enqueue {context.execution_id}")
            return False
    
    def dequeue(self, timeout: Optional[float] = None) -> Optional[ExecutionContext]:
        """Get next execution from queue"""
        try:
            priority, timestamp, context = self.queue.get(timeout=timeout)
            context.started_at = datetime.now()
            return context
        except queue.Empty:
            return None
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()
    
    def get_active_executions(self) -> List[ExecutionContext]:
        """Get list of active executions"""
        with self._lock:
            return list(self.active_executions.values())
    
    def complete_execution(self, execution_id: str, result: ExecutionResult):
        """Mark execution as completed"""
        with self._lock:
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            self.completed_executions[execution_id] = result

class AgentExecutor:
    """
    Advanced agent execution engine with sophisticated orchestration capabilities
    Handles parallel execution, resource management, error recovery, and performance optimization
    """
    
    def __init__(self, max_workers: int = None):
        """Initialize the Agent Executor"""
        self.config = get_config()
        self.llm = get_llm_runner()
        
        # Execution configuration
        self.max_workers = max_workers or self.config.llm.max_concurrent_requests
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Execution management
        self.execution_queue = ExecutionQueue()
        self.active_futures: Dict[str, Future] = {}
        self.execution_history: List[ExecutionResult] = []
        
        # Performance tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_execution_time = 0.0
        
        # Resource management
        self.resource_monitor = ResourceMonitor()
        
        # Execution control
        self._shutdown_event = threading.Event()
        self._worker_thread = None
        
        # Start background worker
        self._start_background_worker()
        
        logger.info(f"Agent Executor initialized with {self.max_workers} max workers")
    
    def _start_background_worker(self):
        """Start background worker thread for queue processing"""
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
    
    def _process_queue(self):
        """Background worker to process execution queue"""
        while not self._shutdown_event.is_set():
            try:
                # Get next execution from queue
                context = self.execution_queue.dequeue(timeout=1.0)
                if context is None:
                    continue
                
                # Submit to thread pool
                future = self.executor.submit(self._execute_with_monitoring, context)
                self.active_futures[context.execution_id] = future
                
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                time.sleep(1.0)
    
    def _execute_with_monitoring(self, context: ExecutionContext) -> ExecutionResult:
        """Execute agent action with comprehensive monitoring"""
        
        result = ExecutionResult(
            execution_id=context.execution_id,
            agent_id=context.agent_id,
            action=context.action,
            status=ExecutionStatus.PREPARING,
            queued_at=context.created_at,
            started_at=context.started_at
        )
        
        # Calculate queue time
        if context.started_at and context.created_at:
            result.queue_time_seconds = (context.started_at - context.created_at).total_seconds()
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        try:
            # Call start callback
            if context.on_start:
                context.on_start(context)
            
            # Preparation phase
            prep_start = time.time()
            agent_instance = self._get_agent_instance(context.agent_id)
            action_method = self._get_agent_method(agent_instance, context.action)
            prep_time = time.time() - prep_start
            result.preparation_time_seconds = prep_time
            
            # Execution phase
            result.status = ExecutionStatus.EXECUTING
            exec_start = time.time()
            
            # Set up timeout handling
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Execution timeout after {context.timeout_seconds} seconds")
            
            # Execute with timeout (if supported by system)
            try:
                if hasattr(signal, 'SIGALRM'):  # Unix systems
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(context.timeout_seconds)
                
                # Execute the agent action
                execution_result = action_method(**context.inputs)
                
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel timeout
                
            except TimeoutError:
                result.status = ExecutionStatus.TIMEOUT
                result.error = f"Execution timed out after {context.timeout_seconds} seconds"
                result.error_type = "TimeoutError"
                raise
            
            exec_time = time.time() - exec_start
            result.actual_execution_time_seconds = exec_time
            
            # Success
            result.result = execution_result
            result.status = ExecutionStatus.COMPLETED
            result.completed_at = datetime.now()
            
            # Call completion callback
            if context.on_complete:
                context.on_complete(context, result)
            
            logger.info(f"Execution {context.execution_id} completed successfully in {exec_time:.2f}s")
            
        except Exception as e:
            # Error handling
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            result.error_type = type(e).__name__
            result.error_traceback = traceback.format_exc()
            result.completed_at = datetime.now()
            
            # Call error callback
            if context.on_error:
                context.on_error(context, result, e)
            
            # Retry logic
            if result.retry_count < context.max_retries:
                logger.warning(f"Execution {context.execution_id} failed, retrying ({result.retry_count + 1}/{context.max_retries})")
                
                # Add to retry history
                result.retry_history.append({
                    "attempt": result.retry_count + 1,
                    "error": result.error,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Wait before retry with exponential backoff
                retry_delay = context.retry_delay * (context.retry_backoff ** result.retry_count)
                time.sleep(retry_delay)
                
                # Reset for retry
                result.retry_count += 1
                result.status = ExecutionStatus.RETRYING
                
                # Recursive retry
                return self._execute_with_monitoring(context)
            
            logger.error(f"Execution {context.execution_id} failed permanently: {e}")
        
        finally:
            # Resource monitoring
            monitor.sample_resources()
            metrics = monitor.get_metrics()
            result.memory_usage_mb = metrics["peak_memory_mb"]
            result.cpu_usage_percent = metrics["average_cpu_percent"]
            result.execution_time_seconds = metrics["execution_time"]
            
            # Update statistics
            self.total_executions += 1
            if result.status == ExecutionStatus.COMPLETED:
                self.successful_executions += 1
            else:
                self.failed_executions += 1
            
            self.total_execution_time += result.execution_time_seconds
            
            # Store result
            self.execution_history.append(result)
            self.execution_queue.complete_execution(context.execution_id, result)
            
            # Clean up future reference
            if context.execution_id in self.active_futures:
                del self.active_futures[context.execution_id]
        
        return result
    
    def _get_agent_instance(self, agent_id: str) -> Any:
        """Get agent instance from registry"""
        # Import here to avoid circular imports
        from backend.agents.registry import get_registry
        
        registry = get_registry()
        agent_info = registry.get_agent(agent_id)
        
        if not agent_info:
            raise ValueError(f"Agent {agent_id} not found in registry")
        
        if agent_info.status.value != "available":
            raise RuntimeError(f"Agent {agent_id} is not available (status: {agent_info.status.value})")
        
        return agent_info.instance
    
    def _get_agent_method(self, agent_instance: Any, action: str) -> Callable:
        """Get agent method for execution"""
        if not hasattr(agent_instance, action):
            raise AttributeError(f"Agent does not have action: {action}")
        
        method = getattr(agent_instance, action)
        if not callable(method):
            raise AttributeError(f"Action {action} is not callable")
        
        return method
    
    def execute_async(self, 
                     agent_id: str,
                     action: str,
                     inputs: Dict[str, Any],
                     **kwargs) -> str:
        """
        Execute agent action asynchronously
        
        Args:
            agent_id: ID of agent to execute
            action: Action method name
            inputs: Input parameters
            **kwargs: Additional execution parameters
            
        Returns:
            str: Execution ID for tracking
        """
        
        # Generate execution ID
        execution_id = f"exec_{agent_id}_{action}_{int(time.time() * 1000)}"
        
        # Create execution context
        context = ExecutionContext(
            execution_id=execution_id,
            agent_id=agent_id,
            action=action,
            inputs=inputs,
            priority=ExecutionPriority(kwargs.get('priority', ExecutionPriority.NORMAL.value)),
            timeout_seconds=kwargs.get('timeout_seconds', 300),
            max_retries=kwargs.get('max_retries', 2),
            retry_delay=kwargs.get('retry_delay', 1.0),
            retry_backoff=kwargs.get('retry_backoff', 2.0),
            environment_vars=kwargs.get('environment_vars', {}),
            execution_hints=kwargs.get('execution_hints', {}),
            on_start=kwargs.get('on_start'),
            on_progress=kwargs.get('on_progress'),
            on_complete=kwargs.get('on_complete'),
            on_error=kwargs.get('on_error')
        )
        
        # Queue for execution
        if self.execution_queue.enqueue(context):
            logger.info(f"Execution {execution_id} queued successfully")
            return execution_id
        else:
            raise RuntimeError(f"Failed to queue execution {execution_id}")
    
    def execute_sync(self,
                    agent_id: str,
                    action: str,
                    inputs: Dict[str, Any],
                    timeout: Optional[float] = None,
                    **kwargs) -> ExecutionResult:
        """
        Execute agent action synchronously
        
        Args:
            agent_id: ID of agent to execute
            action: Action method name
            inputs: Input parameters
            timeout: Max wait time for completion
            **kwargs: Additional execution parameters
            
        Returns:
            ExecutionResult: Complete execution result
        """
        
        # Start async execution
        execution_id = self.execute_async(agent_id, action, inputs, **kwargs)
        
        # Wait for completion
        return self.wait_for_completion(execution_id, timeout)
    
    def execute_batch(self,
                     executions: List[Dict[str, Any]],
                     mode: ExecutionMode = ExecutionMode.PARALLEL,
                     max_concurrent: Optional[int] = None) -> List[ExecutionResult]:
        """
        Execute multiple agent actions in batch
        
        Args:
            executions: List of execution specifications
            mode: Execution mode (sequential, parallel, etc.)
            max_concurrent: Max concurrent executions
            
        Returns:
            List[ExecutionResult]: Results for all executions
        """
        
        if mode == ExecutionMode.SEQUENTIAL:
            return self._execute_sequential(executions)
        elif mode == ExecutionMode.PARALLEL:
            return self._execute_parallel(executions, max_concurrent)
        elif mode == ExecutionMode.PIPELINE:
            return self._execute_pipeline(executions)
        elif mode == ExecutionMode.ADAPTIVE:
            return self._execute_adaptive(executions)
        else:
            raise ValueError(f"Unsupported execution mode: {mode}")
    
    def _execute_sequential(self, executions: List[Dict[str, Any]]) -> List[ExecutionResult]:
        """Execute batch sequentially"""
        results = []
        
        for exec_spec in executions:
            result = self.execute_sync(**exec_spec)
            results.append(result)
            
            # Stop on failure if configured
            if result.status == ExecutionStatus.FAILED and exec_spec.get('stop_on_failure', False):
                break
        
        return results
    
    def _execute_parallel(self, 
                         executions: List[Dict[str, Any]], 
                         max_concurrent: Optional[int] = None) -> List[ExecutionResult]:
        """Execute batch in parallel"""
        max_concurrent = max_concurrent or self.max_workers
        
        # Start all executions
        execution_ids = []
        for exec_spec in executions:
            exec_id = self.execute_async(**exec_spec)
            execution_ids.append(exec_id)
        
        # Wait for all to complete
        results = []
        for exec_id in execution_ids:
            result = self.wait_for_completion(exec_id)
            results.append(result)
        
        return results
    
    def _execute_pipeline(self, executions: List[Dict[str, Any]]) -> List[ExecutionResult]:
        """Execute batch in pipeline fashion (output of one feeds into next)"""
        results = []
        previous_result = None
        
        for exec_spec in executions:
            # Add previous result to inputs if available
            if previous_result and previous_result.status == ExecutionStatus.COMPLETED:
                exec_spec['inputs']['previous_result'] = previous_result.result
            
            result = self.execute_sync(**exec_spec)
            results.append(result)
            previous_result = result
            
            # Stop pipeline on failure
            if result.status == ExecutionStatus.FAILED:
                break
        
        return results
    
    def _execute_adaptive(self, executions: List[Dict[str, Any]]) -> List[ExecutionResult]:
        """Intelligently choose execution mode based on dependencies and resources"""
        # Analyze dependencies
        has_dependencies = any(exec_spec.get('dependencies') for exec_spec in executions)
        
        # Choose mode
        if has_dependencies:
            return self._execute_pipeline(executions)
        elif len(executions) <= 2:
            return self._execute_sequential(executions)
        else:
            return self._execute_parallel(executions)
    
    def wait_for_completion(self, 
                          execution_id: str, 
                          timeout: Optional[float] = None) -> ExecutionResult:
        """Wait for execution to complete"""
        
        start_time = time.time()
        check_interval = 0.1
        
        while True:
            # Check if execution is complete
            if execution_id in self.execution_queue.completed_executions:
                return self.execution_queue.completed_executions[execution_id]
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                # Try to cancel execution
                self.cancel_execution(execution_id)
                raise TimeoutError(f"Execution {execution_id} timed out after {timeout} seconds")
            
            time.sleep(check_interval)
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        
        # Cancel future if active
        if execution_id in self.active_futures:
            future = self.active_futures[execution_id]
            cancelled = future.cancel()
            
            if cancelled:
                logger.info(f"Execution {execution_id} cancelled successfully")
                return True
        
        logger.warning(f"Could not cancel execution {execution_id}")
        return False
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of execution"""
        
        # Check completed executions
        if execution_id in self.execution_queue.completed_executions:
            result = self.execution_queue.completed_executions[execution_id]
            return result.to_dict()
        
        # Check active executions
        active_executions = self.execution_queue.get_active_executions()
        for context in active_executions:
            if context.execution_id == execution_id:
                return {
                    "execution_id": execution_id,
                    "status": "running",
                    "agent_id": context.agent_id,
                    "action": context.action,
                    "started_at": context.started_at.isoformat() if context.started_at else None
                }
        
        return None
    
    def get_executor_status(self) -> Dict[str, Any]:
        """Get executor status and statistics"""
        return {
            "max_workers": self.max_workers,
            "active_executions": len(self.active_futures),
            "queue_size": self.execution_queue.get_queue_size(),
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": (self.successful_executions / max(self.total_executions, 1)) * 100,
            "average_execution_time": self.total_execution_time / max(self.total_executions, 1),
            "uptime_seconds": (datetime.now() - datetime.now()).total_seconds(),  # Will be replaced with actual uptime
            "recent_executions": [result.to_dict() for result in self.execution_history[-10:]]
        }
    
    def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """Shutdown the executor gracefully"""
        logger.info("Shutting down Agent Executor...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel all active futures
        for execution_id, future in list(self.active_futures.items()):
            future.cancel()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=wait, timeout=timeout)
        
        # Wait for worker thread
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        
        logger.info("Agent Executor shutdown complete")

# Global executor instance
_executor: Optional[AgentExecutor] = None

def get_executor() -> AgentExecutor:
    """Get or create the global agent executor instance"""
    global _executor
    if _executor is None:
        _executor = AgentExecutor()
    return _executor