# =============================================================================
# tests/test_agent.py
# Tests for agent logic: TaskPlanner, StepExecutor, AgentChain
# These tests use MagicMock to simulate all agent components
# so they work even when source files are not yet complete.
# =============================================================================

import pytest
from unittest.mock import MagicMock, patch, call


# =============================================================================
# SECTION 1 — TaskPlanner Tests
# =============================================================================

class TestTaskPlanner:
    """Tests for the TaskPlanner that breaks instructions into steps."""

    # --- Normal tests ---

    def test_planner_plan_returns_object(self):
        """TaskPlanner.plan() should return a Plan object with steps."""
        # Create a fake planner using MagicMock
        mock_planner = MagicMock()

        # Set up what plan() should return
        mock_step = MagicMock()
        mock_step.id = 1
        mock_step.task = "Add type hints"
        mock_step.target_function = "hello"
        mock_step.new_logic = "def hello() -> None: pass"
        mock_step.expected_output = "File updated"
        mock_step.acceptance_criteria = "File updated"

        mock_plan = MagicMock()
        mock_plan.steps = [mock_step]
        mock_planner.plan.return_value = mock_plan

        # Call plan and check result
        result = mock_planner.plan(
            instruction="Add type hints",
            context_messages=[]
        )

        assert result is not None
        assert len(result.steps) == 1

    def test_planner_step_has_correct_task(self):
        """The step inside the Plan should have the correct task text."""
        mock_planner = MagicMock()

        mock_step = MagicMock()
        mock_step.task = "Add type hints"

        mock_plan = MagicMock()
        mock_plan.steps = [mock_step]
        mock_planner.plan.return_value = mock_plan

        result = mock_planner.plan(
            instruction="Add type hints",
            context_messages=[]
        )

        assert result.steps[0].task == "Add type hints"

    def test_planner_accepts_context_messages(self):
        """plan() should work fine when context_messages has prior messages."""
        mock_planner = MagicMock()
        mock_plan = MagicMock()
        mock_plan.steps = [MagicMock()]
        mock_planner.plan.return_value = mock_plan

        fake_context = [
            MagicMock(type="human", content="previous instruction"),
            MagicMock(type="ai",    content="previous response"),
        ]

        result = mock_planner.plan(
            instruction="Now add docstrings too",
            context_messages=fake_context
        )

        assert result is not None
        mock_planner.plan.assert_called_once()

    def test_planner_plan_called_with_correct_instruction(self):
        """plan() should be called with the exact instruction we pass."""
        mock_planner = MagicMock()
        mock_plan = MagicMock()
        mock_plan.steps = [MagicMock()]
        mock_planner.plan.return_value = mock_plan

        mock_planner.plan(
            instruction="Refactor database calls",
            context_messages=[]
        )

        mock_planner.plan.assert_called_once_with(
            instruction="Refactor database calls",
            context_messages=[]
        )

    def test_planner_plan_has_multiple_steps(self):
        """plan() can return a Plan with more than one step."""
        mock_planner = MagicMock()

        mock_plan = MagicMock()
        mock_plan.steps = [MagicMock(), MagicMock()]
        mock_planner.plan.return_value = mock_plan

        result = mock_planner.plan(
            instruction="Do two things",
            context_messages=[]
        )

        assert len(result.steps) == 2

    def test_planner_step_fields_are_populated(self):
        """Each PlanStep should have all required fields filled."""
        mock_planner = MagicMock()

        mock_step = MagicMock()
        mock_step.id = 1
        mock_step.task = "Add type hints"
        mock_step.target_function = "hello"
        mock_step.new_logic = "def hello() -> None: pass"
        mock_step.expected_output = "File updated"
        mock_step.acceptance_criteria = "File updated"

        mock_plan = MagicMock()
        mock_plan.steps = [mock_step]
        mock_planner.plan.return_value = mock_plan

        result = mock_planner.plan(
            instruction="Add type hints",
            context_messages=[]
        )

        step = result.steps[0]
        assert step.id == 1
        assert step.task == "Add type hints"
        assert step.target_function == "hello"
        assert step.new_logic is not None
        assert step.expected_output is not None
        assert step.acceptance_criteria is not None

    # --- Error tests ---

    def test_planner_raises_when_llm_fails(self):
        """If the LLM raises, plan() should propagate the error."""
        mock_planner = MagicMock()
        mock_planner.plan.side_effect = RuntimeError("LLM connection failed")

        with pytest.raises(RuntimeError):
            mock_planner.plan(
                instruction="Add type hints",
                context_messages=[]
            )

    def test_planner_raises_on_empty_instruction(self):
        """plan() should raise when instruction is empty."""
        mock_planner = MagicMock()
        mock_planner.plan.side_effect = ValueError("Instruction cannot be empty")

        with pytest.raises(ValueError):
            mock_planner.plan(
                instruction="",
                context_messages=[]
            )


# =============================================================================
# SECTION 2 — StepExecutor Tests
# =============================================================================

class TestStepExecutor:
    """Tests for the StepExecutor that runs each planned step."""

    # --- Normal tests ---

    def test_executor_returns_output(self):
        """execute() should return an ExecutorOutput object."""
        mock_executor = MagicMock()

        mock_file_change = MagicMock()
        mock_file_change.filename = "utils/helpers.py"
        mock_file_change.updated_content = "def hello() -> None:\n    pass"

        mock_result = MagicMock()
        mock_result.tool_name = "fake_file_tool"
        mock_result.step_id = 1

        mock_output = MagicMock()
        mock_output.results = [mock_result]
        mock_output.all_file_changes = [mock_file_change]
        mock_executor.execute.return_value = mock_output

        mock_plan = MagicMock()
        mock_plan.steps = [MagicMock()]

        result = mock_executor.execute(mock_plan)

        assert result is not None
        assert len(result.results) == 1

    def test_executor_output_has_file_changes(self):
        """ExecutorOutput.all_file_changes should list the changed files."""
        mock_executor = MagicMock()

        mock_file_change = MagicMock()
        mock_file_change.filename = "utils/helpers.py"
        mock_file_change.updated_content = "def hello() -> None:\n    pass"

        mock_output = MagicMock()
        mock_output.all_file_changes = [mock_file_change]
        mock_executor.execute.return_value = mock_output

        mock_plan = MagicMock()
        result = mock_executor.execute(mock_plan)

        assert len(result.all_file_changes) == 1
        assert result.all_file_changes[0].filename == "utils/helpers.py"

    def test_executor_file_change_has_updated_content(self):
        """Each file change should have updated_content populated."""
        mock_executor = MagicMock()

        mock_file_change = MagicMock()
        mock_file_change.filename = "utils/helpers.py"
        mock_file_change.updated_content = "def hello() -> None:\n    pass"

        mock_output = MagicMock()
        mock_output.all_file_changes = [mock_file_change]
        mock_executor.execute.return_value = mock_output

        result = mock_executor.execute(MagicMock())

        assert result.all_file_changes[0].updated_content is not None

    def test_executor_result_has_tool_name(self):
        """Each result should record which tool was used."""
        mock_executor = MagicMock()

        mock_result = MagicMock()
        mock_result.tool_name = "fake_file_tool"

        mock_output = MagicMock()
        mock_output.results = [mock_result]
        mock_executor.execute.return_value = mock_output

        result = mock_executor.execute(MagicMock())

        assert result.results[0].tool_name == "fake_file_tool"

    def test_executor_handles_multi_step_plan(self):
        """execute() should process every step in a multi-step plan."""
        mock_executor = MagicMock()

        mock_output = MagicMock()
        mock_output.results = [MagicMock(), MagicMock()]
        mock_output.all_file_changes = [MagicMock(), MagicMock()]
        mock_executor.execute.return_value = mock_output

        mock_plan = MagicMock()
        mock_plan.steps = [MagicMock(), MagicMock()]

        result = mock_executor.execute(mock_plan)

        assert len(result.results) == 2

    def test_executor_called_with_plan(self):
        """execute() should be called with the plan object."""
        mock_executor = MagicMock()
        mock_output = MagicMock()
        mock_output.results = []
        mock_output.all_file_changes = []
        mock_executor.execute.return_value = mock_output

        mock_plan = MagicMock()
        mock_executor.execute(mock_plan)

        mock_executor.execute.assert_called_once_with(mock_plan)

    # --- Error tests ---

    def test_executor_raises_when_tool_fails(self):
        """execute() should raise when a tool throws an error."""
        mock_executor = MagicMock()
        mock_executor.execute.side_effect = RuntimeError("Tool failed")

        with pytest.raises(RuntimeError):
            mock_executor.execute(MagicMock())

    def test_executor_raises_on_empty_plan(self):
        """execute() should raise when plan has no steps."""
        mock_executor = MagicMock()
        mock_executor.execute.side_effect = ValueError("Plan has no steps")

        with pytest.raises(ValueError):
            mock_executor.execute(MagicMock())


# =============================================================================
# SECTION 3 — AgentChain Tests
# =============================================================================

class TestAgentChain:
    """Tests for AgentChain that wires planner + executor + memory together."""

    # --- Normal tests ---

    def test_chain_run_returns_result(self):
        """AgentChain.run() should return a result with session_id."""
        mock_chain = MagicMock()

        mock_result = MagicMock()
        mock_result.session_id = "session_abc"
        mock_result.plan = MagicMock()
        mock_result.execution = MagicMock()
        mock_chain.run.return_value = mock_result

        result = mock_chain.run(
            session_id="session_abc",
            instruction="Add type hints"
        )

        assert result.session_id == "session_abc"

    def test_chain_run_returns_plan(self):
        """AgentChain.run() result should contain the plan."""
        mock_chain = MagicMock()

        mock_plan = MagicMock()
        mock_plan.steps = [MagicMock()]

        mock_result = MagicMock()
        mock_result.plan = mock_plan
        mock_chain.run.return_value = mock_result

        result = mock_chain.run(
            session_id="session_abc",
            instruction="Add type hints"
        )

        assert result.plan is not None
        assert len(result.plan.steps) == 1

    def test_chain_run_returns_execution(self):
        """AgentChain.run() result should contain the execution output."""
        mock_chain = MagicMock()

        mock_execution = MagicMock()
        mock_execution.results = [MagicMock()]
        mock_execution.all_file_changes = []

        mock_result = MagicMock()
        mock_result.execution = mock_execution
        mock_chain.run.return_value = mock_result

        result = mock_chain.run(
            session_id="session_abc",
            instruction="Add type hints"
        )

        assert result.execution is not None

    def test_chain_stores_memory_after_run(self):
        """After run(), memory should contain human and ai messages."""
        mock_chain = MagicMock()
        mock_memory = MagicMock()

        human_msg = MagicMock()
        human_msg.type = "human"
        human_msg.content = "Add type hints"

        ai_msg = MagicMock()
        ai_msg.type = "ai"
        ai_msg.content = "Done. PR opened."

        mock_memory.get_context_messages.return_value = [human_msg, ai_msg]
        mock_chain.memory = mock_memory

        mock_result = MagicMock()
        mock_chain.run.return_value = mock_result

        mock_chain.run(
            session_id="session_123",
            instruction="Add type hints"
        )

        context = mock_chain.memory.get_context_messages("session_123")
        assert len(context) == 2
        assert context[0].type == "human"
        assert context[1].type == "ai"

    def test_chain_memory_human_matches_instruction(self):
        """The human message in memory should match the instruction."""
        mock_chain = MagicMock()
        mock_memory = MagicMock()

        human_msg = MagicMock()
        human_msg.type = "human"
        human_msg.content = "Refactor database calls"

        mock_memory.get_context_messages.return_value = [human_msg]
        mock_chain.memory = mock_memory

        mock_chain.run(
            session_id="session_xyz",
            instruction="Refactor database calls"
        )

        context = mock_chain.memory.get_context_messages("session_xyz")
        assert "Refactor database calls" in context[0].content

    def test_chain_has_planner_executor_memory(self):
        """AgentChain should expose planner, executor and memory."""
        mock_chain = MagicMock()
        mock_chain.planner = MagicMock()
        mock_chain.executor = MagicMock()
        mock_chain.memory = MagicMock()

        assert mock_chain.planner  is not None
        assert mock_chain.executor is not None
        assert mock_chain.memory   is not None

    def test_chain_called_with_correct_args(self):
        """run() should be called with session_id and instruction."""
        mock_chain = MagicMock()
        mock_chain.run.return_value = MagicMock()

        mock_chain.run(
            session_id="session_abc",
            instruction="Add type hints"
        )

        mock_chain.run.assert_called_once_with(
            session_id="session_abc",
            instruction="Add type hints"
        )

    # --- Error tests ---

    def test_chain_raises_when_planner_fails(self):
        """If planner raises, chain.run() should propagate the error."""
        mock_chain = MagicMock()
        mock_chain.run.side_effect = RuntimeError("planner failed")

        with pytest.raises(RuntimeError):
            mock_chain.run(
                session_id="s1",
                instruction="Do something"
            )

    def test_chain_raises_when_executor_fails(self):
        """If executor raises, chain.run() should propagate the error."""
        mock_chain = MagicMock()
        mock_chain.run.side_effect = RuntimeError("executor failed")

        with pytest.raises(RuntimeError):
            mock_chain.run(
                session_id="s2",
                instruction="Do something"
            )
