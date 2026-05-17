# =============================================================================
# tests/test_agent.py
# Tests for: TaskPlanner, StepExecutor, AgentChain
# Uses pytest + MagicMock + patch — no real LLM calls are made
# =============================================================================

import pytest
from unittest.mock import MagicMock, patch
from agent.planner import TaskPlanner, Plan, PlanStep
from agent.executor import StepExecutor, ToolSpec, ToolDecision, ExecutorOutput, StepExecutionResult
from agent.chain import AgentChain
from agent.memory import MemoryManager


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_plan():
    """Return a simple one-step Plan for reuse across tests."""
    return Plan(steps=[
        PlanStep(
            id=1,
            task="Add type hints",
            target_function="hello",
            new_logic="def hello() -> None: pass",
            expected_output="File updated",
            acceptance_criteria="File updated"
        )
    ])


def make_fake_tool():
    """Return a fake ToolSpec that simulates a file edit."""
    def fake_fn(inputs):
        return {
            "file_changes": [
                {
                    "filename": "utils/helpers.py",
                    "updated_content": "def hello() -> None:\n    pass",
                    "reason": "Added type hint"
                }
            ],
            "notes": "Type hint added"
        }
    return ToolSpec(
        name="fake_file_tool",
        description="A fake tool that edits files",
        fn=fake_fn
    )


# =============================================================================
# SECTION 1 — TaskPlanner
# =============================================================================

class TestTaskPlanner:

    # --- Normal tests ---

    def test_planner_creates_plan(self):
        """TaskPlanner.plan() returns a Plan with steps."""
        planner = TaskPlanner(llm=MagicMock())
        mock_plan = make_plan()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_plan

        with patch.object(planner, 'build_chain', return_value=mock_chain):
            result = planner.plan(
                instruction="Add type hints",
                context_messages=[]
            )

        assert isinstance(result, Plan), "plan() must return a Plan object"
        assert len(result.steps) == 1, "Plan should have 1 step"

    def test_planner_step_has_correct_task(self):
        """The step inside the Plan should have the correct task text."""
        planner = TaskPlanner(llm=MagicMock())
        mock_plan = make_plan()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_plan

        with patch.object(planner, 'build_chain', return_value=mock_chain):
            result = planner.plan(
                instruction="Add type hints",
                context_messages=[]
            )

        assert result.steps[0].task == "Add type hints"

    def test_planner_accepts_context_messages(self):
        """plan() should work fine when context_messages has prior messages."""
        planner = TaskPlanner(llm=MagicMock())
        mock_plan = make_plan()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_plan

        fake_context = [
            MagicMock(type="human", content="previous instruction"),
            MagicMock(type="ai", content="previous response"),
        ]

        with patch.object(planner, 'build_chain', return_value=mock_chain):
            result = planner.plan(
                instruction="Now add docstrings too",
                context_messages=fake_context
            )

        assert isinstance(result, Plan)

    def test_planner_plan_has_multiple_steps(self):
        """plan() can return a Plan with more than one step."""
        planner = TaskPlanner(llm=MagicMock())

        multi_step_plan = Plan(steps=[
            PlanStep(id=1, task="Step one", target_function="f1",
                     new_logic="pass", expected_output="done",
                     acceptance_criteria="done"),
            PlanStep(id=2, task="Step two", target_function="f2",
                     new_logic="pass", expected_output="done",
                     acceptance_criteria="done"),
        ])

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = multi_step_plan

        with patch.object(planner, 'build_chain', return_value=mock_chain):
            result = planner.plan(
                instruction="Do two things",
                context_messages=[]
            )

        assert len(result.steps) == 2

    def test_planner_step_fields_are_populated(self):
        """Each PlanStep should have all required fields filled."""
        planner = TaskPlanner(llm=MagicMock())
        mock_plan = make_plan()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_plan

        with patch.object(planner, 'build_chain', return_value=mock_chain):
            result = planner.plan(
                instruction="Add type hints",
                context_messages=[]
            )

        step = result.steps[0]
        assert step.id is not None
        assert step.task is not None
        assert step.target_function is not None
        assert step.new_logic is not None
        assert step.expected_output is not None
        assert step.acceptance_criteria is not None

    # --- Error tests ---

    def test_planner_raises_when_chain_fails(self):
        """If the LLM chain raises, plan() should propagate the error."""
        planner = TaskPlanner(llm=MagicMock())
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("LLM connection failed")

        with patch.object(planner, 'build_chain', return_value=mock_chain):
            with pytest.raises((RuntimeError, Exception)):
                planner.plan(
                    instruction="Add type hints",
                    context_messages=[]
                )

    def test_planner_can_be_instantiated_with_mock_llm(self):
        """TaskPlanner should be creatable with a mocked LLM."""
        planner = TaskPlanner(llm=MagicMock())
        assert planner is not None


# =============================================================================
# SECTION 2 — StepExecutor
# =============================================================================

class TestStepExecutor:

    # --- Normal tests ---

    def test_executor_runs_tool_and_returns_output(self):
        """execute() should return an ExecutorOutput object."""
        fake_tool = make_fake_tool()
        executor = StepExecutor(llm=MagicMock(), tools=[fake_tool])

        dummy_plan = make_plan()
        mock_decision = ToolDecision(
            tool_name="fake_file_tool",
            tool_input={"filename": "utils/helpers.py"}
        )

        with patch.object(executor, '_decide_tool', return_value=mock_decision):
            result = executor.execute(dummy_plan)

        assert isinstance(result, ExecutorOutput), \
            "execute() must return an ExecutorOutput"

    def test_executor_output_has_results(self):
        """ExecutorOutput.results should contain one result per step."""
        fake_tool = make_fake_tool()
        executor = StepExecutor(llm=MagicMock(), tools=[fake_tool])

        dummy_plan = make_plan()
        mock_decision = ToolDecision(
            tool_name="fake_file_tool",
            tool_input={"filename": "utils/helpers.py"}
        )

        with patch.object(executor, '_decide_tool', return_value=mock_decision):
            result = executor.execute(dummy_plan)

        assert len(result.results) == 1
        assert result.results[0].tool_name == "fake_file_tool"

    def test_executor_output_has_file_changes(self):
        """ExecutorOutput.all_file_changes should list the changed files."""
        fake_tool = make_fake_tool()
        executor = StepExecutor(llm=MagicMock(), tools=[fake_tool])

        dummy_plan = make_plan()
        mock_decision = ToolDecision(
            tool_name="fake_file_tool",
            tool_input={"filename": "utils/helpers.py"}
        )

        with patch.object(executor, '_decide_tool', return_value=mock_decision):
            result = executor.execute(dummy_plan)

        assert len(result.all_file_changes) == 1
        assert result.all_file_changes[0].filename == "utils/helpers.py"

    def test_executor_file_change_has_updated_content(self):
        """Each file change should have updated_content populated."""
        fake_tool = make_fake_tool()
        executor = StepExecutor(llm=MagicMock(), tools=[fake_tool])

        dummy_plan = make_plan()
        mock_decision = ToolDecision(
            tool_name="fake_file_tool",
            tool_input={"filename": "utils/helpers.py"}
        )

        with patch.object(executor, '_decide_tool', return_value=mock_decision):
            result = executor.execute(dummy_plan)

        assert result.all_file_changes[0].updated_content is not None

    def test_executor_handles_multi_step_plan(self):
        """execute() should process every step in a multi-step plan."""
        def fake_fn(inputs):
            return {
                "file_changes": [
                    {"filename": "a.py", "updated_content": "x=1", "reason": "edit"}
                ],
                "notes": "done"
            }

        fake_tool = ToolSpec(name="multi_tool", description="multi", fn=fake_fn)
        executor = StepExecutor(llm=MagicMock(), tools=[fake_tool])

        multi_plan = Plan(steps=[
            PlanStep(id=1, task="Step 1", target_function="f1",
                     new_logic="pass", expected_output="ok",
                     acceptance_criteria="ok"),
            PlanStep(id=2, task="Step 2", target_function="f2",
                     new_logic="pass", expected_output="ok",
                     acceptance_criteria="ok"),
        ])

        mock_decision = ToolDecision(
            tool_name="multi_tool",
            tool_input={"filename": "a.py"}
        )

        with patch.object(executor, '_decide_tool', return_value=mock_decision):
            result = executor.execute(multi_plan)

        assert len(result.results) == 2

    # --- Error tests ---

    def test_executor_raises_when_tool_not_found(self):
        """If _decide_tool picks a tool that doesn't exist, execute() should raise."""
        fake_tool = make_fake_tool()
        executor = StepExecutor(llm=MagicMock(), tools=[fake_tool])

        dummy_plan = make_plan()
        bad_decision = ToolDecision(
            tool_name="nonexistent_tool",
            tool_input={}
        )

        with patch.object(executor, '_decide_tool', return_value=bad_decision):
            with pytest.raises((KeyError, ValueError, Exception)):
                executor.execute(dummy_plan)

    def test_executor_can_be_instantiated(self):
        """StepExecutor should be creatable with a mock LLM and empty tools list."""
        executor = StepExecutor(llm=MagicMock(), tools=[])
        assert executor is not None


# =============================================================================
# SECTION 3 — AgentChain
# =============================================================================

class TestAgentChain:

    # --- Normal tests ---

    @patch("agent.chain.ChatGroq")
    def test_chain_processes_request(self, MockChatGroq):
        """AgentChain.run() should return an object with session_id, plan, execution."""
        MockChatGroq.return_value = MagicMock()

        chain = AgentChain(llm=MagicMock(), tools=[])

        dummy_plan = make_plan()
        dummy_execution = ExecutorOutput(
            results=[
                StepExecutionResult(
                    step_id=1,
                    step_task="Add type hints",
                    tool_name="noop",
                    tool_input={}
                )
            ],
            all_file_changes=[]
        )

        with patch.object(chain.planner, 'plan', return_value=dummy_plan), \
             patch.object(chain.executor, 'execute', return_value=dummy_execution):
            result = chain.run(
                session_id="session_abc",
                instruction="Add type hints"
            )

        assert result.session_id == "session_abc"
        assert result.plan == dummy_plan
        assert result.execution == dummy_execution

    @patch("agent.chain.ChatGroq")
    def test_chain_stores_memory(self, MockChatGroq):
        """After run(), memory should contain human and ai messages."""
        MockChatGroq.return_value = MagicMock()

        chain = AgentChain(llm=MagicMock(), tools=[])

        dummy_plan = make_plan()
        dummy_execution = ExecutorOutput(
            results=[
                StepExecutionResult(
                    step_id=1,
                    step_task="Dummy step",
                    tool_name="noop",
                    tool_input={}
                )
            ],
            all_file_changes=[]
        )

        with patch.object(chain.planner, 'plan', return_value=dummy_plan), \
             patch.object(chain.executor, 'execute', return_value=dummy_execution):
            chain.run(session_id="session_123", instruction="Do something")

        context = chain.memory.get_context_messages("session_123")
        assert len(context) == 2
        assert context[0].type == "human"
        assert context[1].type == "ai"

    @patch("agent.chain.ChatGroq")
    def test_chain_memory_human_message_matches_instruction(self, MockChatGroq):
        """The human message stored in memory should match the instruction."""
        MockChatGroq.return_value = MagicMock()

        chain = AgentChain(llm=MagicMock(), tools=[])

        dummy_plan = make_plan()
        dummy_execution = ExecutorOutput(results=[], all_file_changes=[])

        with patch.object(chain.planner, 'plan', return_value=dummy_plan), \
             patch.object(chain.executor, 'execute', return_value=dummy_execution):
            chain.run(
                session_id="session_xyz",
                instruction="Refactor database calls"
            )

        context = chain.memory.get_context_messages("session_xyz")
        assert context[0].type == "human"
        assert "Refactor database calls" in context[0].content

    @patch("agent.chain.ChatGroq")
    def test_chain_has_planner_and_executor(self, MockChatGroq):
        """AgentChain should expose a planner and executor after init."""
        MockChatGroq.return_value = MagicMock()

        chain = AgentChain(llm=MagicMock(), tools=[])

        assert hasattr(chain, "planner"),  "AgentChain must have a 'planner' attribute"
        assert hasattr(chain, "executor"), "AgentChain must have an 'executor' attribute"

    @patch("agent.chain.ChatGroq")
    def test_chain_has_memory(self, MockChatGroq):
        """AgentChain should expose a memory object after init."""
        MockChatGroq.return_value = MagicMock()

        chain = AgentChain(llm=MagicMock(), tools=[])

        assert hasattr(chain, "memory"), "AgentChain must have a 'memory' attribute"

    # --- Error tests ---

    @patch("agent.chain.ChatGroq")
    def test_chain_raises_when_planner_fails(self, MockChatGroq):
        """If planner raises, chain.run() should propagate the error."""
        MockChatGroq.return_value = MagicMock()

        chain = AgentChain(llm=MagicMock(), tools=[])

        with patch.object(chain.planner, 'plan', side_effect=RuntimeError("planner failed")):
            with pytest.raises((RuntimeError, Exception)):
                chain.run(session_id="s1", instruction="Do something")

    @patch("agent.chain.ChatGroq")
    def test_chain_raises_when_executor_fails(self, MockChatGroq):
        """If executor raises, chain.run() should propagate the error."""
        MockChatGroq.return_value = MagicMock()

        chain = AgentChain(llm=MagicMock(), tools=[])
        dummy_plan = make_plan()

        with patch.object(chain.planner, 'plan', return_value=dummy_plan), \
             patch.object(chain.executor, 'execute', side_effect=RuntimeError("executor failed")):
            with pytest.raises((RuntimeError, Exception)):
                chain.run(session_id="s2", instruction="Do something")
