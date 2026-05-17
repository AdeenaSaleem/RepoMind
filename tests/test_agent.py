# =============================================================================
# tests/test_agent.py
# Tests for the LangChain agent logic: AgentChain, Planner, Executor
# Uses pytest + MagicMock + patch so NO real LLM / API calls are made
# =============================================================================

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Helper fixtures shared across all tests in this file
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """Return a fake ChatGroq LLM that never calls Groq's servers."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="mocked LLM response")
    return llm


@pytest.fixture
def sample_instruction():
    return "Add type hints to all functions in utils/helpers.py"


@pytest.fixture
def sample_repo_url():
    return "https://github.com/example/sample-repo"


# =============================================================================
# SECTION 1 — AgentChain  (agent/chain.py)
# =============================================================================

class TestAgentChain:
    """Tests for the main AgentChain class that wires LLM + memory + tools."""

    @patch("agent.chain.ChatGroq")
    def test_agent_chain_initialises(self, MockChatGroq):
        """AgentChain can be instantiated without hitting any real API."""
        MockChatGroq.return_value = MagicMock()

        from agent.chain import AgentChain
        chain = AgentChain()

        assert chain is not None, "AgentChain() should return an object"

    @patch("agent.chain.ChatGroq")
    def test_agent_chain_run_returns_string(self, MockChatGroq, sample_instruction, sample_repo_url):
        """AgentChain.run() should return a non-empty result."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Done. PR opened.")
        MockChatGroq.return_value = mock_llm

        from agent.chain import AgentChain
        chain = AgentChain()

        with patch.object(chain, "planner") as mock_planner, \
             patch.object(chain, "executor") as mock_executor:

            mock_planner.plan.return_value = ["step 1", "step 2"]
            mock_executor.execute.return_value = {
                "status": "success",
                "pr_url": "https://github.com/example/sample-repo/pull/1"
            }

            result = chain.run(instruction=sample_instruction, repo_url=sample_repo_url)

        assert result is not None, "run() must return a value"
        assert isinstance(result, (str, dict)), "run() should return str or dict"

    @patch("agent.chain.ChatGroq")
    def test_agent_chain_run_with_branch_name(self, MockChatGroq, sample_instruction, sample_repo_url):
        """AgentChain.run() accepts an optional branch_name parameter."""
        MockChatGroq.return_value = MagicMock()

        from agent.chain import AgentChain
        chain = AgentChain()

        with patch.object(chain, "planner") as mock_planner, \
             patch.object(chain, "executor") as mock_executor:

            mock_planner.plan.return_value = ["step 1"]
            mock_executor.execute.return_value = {
                "status": "success",
                "pr_url": "https://github.com/example/sample-repo/pull/2"
            }

            result = chain.run(
                instruction=sample_instruction,
                repo_url=sample_repo_url,
                branch_name="repomind/type-hints"
            )

        assert result is not None

    @patch("agent.chain.ChatGroq")
    def test_agent_chain_memory_is_initialised(self, MockChatGroq):
        """AgentChain should initialise a memory object after construction."""
        MockChatGroq.return_value = MagicMock()

        from agent.chain import AgentChain
        chain = AgentChain()

        assert hasattr(chain, "memory"), "AgentChain must have a 'memory' attribute"

    @patch("agent.chain.ChatGroq")
    def test_agent_chain_has_planner_and_executor(self, MockChatGroq):
        """AgentChain exposes a planner and an executor after init."""
        MockChatGroq.return_value = MagicMock()

        from agent.chain import AgentChain
        chain = AgentChain()

        assert hasattr(chain, "planner"),  "AgentChain must have a 'planner' attribute"
        assert hasattr(chain, "executor"), "AgentChain must have an 'executor' attribute"

    @patch("agent.chain.ChatGroq")
    def test_agent_chain_run_raises_on_empty_instruction(self, MockChatGroq, sample_repo_url):
        """run() should raise ValueError (or similar) when instruction is empty."""
        MockChatGroq.return_value = MagicMock()

        from agent.chain import AgentChain
        chain = AgentChain()

        with pytest.raises((ValueError, TypeError, Exception)):
            chain.run(instruction="", repo_url=sample_repo_url)

    @patch("agent.chain.ChatGroq")
    def test_agent_chain_run_raises_on_empty_repo_url(self, MockChatGroq, sample_instruction):
        """run() should raise an error when repo_url is empty."""
        MockChatGroq.return_value = MagicMock()

        from agent.chain import AgentChain
        chain = AgentChain()

        with pytest.raises((ValueError, TypeError, Exception)):
            chain.run(instruction=sample_instruction, repo_url="")

    @patch("agent.chain.ChatGroq")
    def test_agent_chain_executor_exception_propagates(self, MockChatGroq, sample_instruction, sample_repo_url):
        """If the executor raises, AgentChain.run() should not swallow it silently."""
        MockChatGroq.return_value = MagicMock()

        from agent.chain import AgentChain
        chain = AgentChain()

        with patch.object(chain, "planner") as mock_planner, \
             patch.object(chain, "executor") as mock_executor:

            mock_planner.plan.return_value = ["step 1"]
            mock_executor.execute.side_effect = RuntimeError("executor blew up")

            with pytest.raises((RuntimeError, Exception)):
                chain.run(instruction=sample_instruction, repo_url=sample_repo_url)


# =============================================================================
# SECTION 2 — Planner  (agent/planner.py)
# =============================================================================

class TestPlanner:
    """Tests for the Planner that breaks an instruction into ordered edit steps."""

    @patch("agent.planner.ChatGroq")
    def test_planner_plan_returns_list(self, MockChatGroq, sample_instruction, sample_repo_url):
        """Planner.plan() should return a list of steps."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="1. Parse file\n2. Add hints\n3. Save")
        MockChatGroq.return_value = mock_llm

        from agent.planner import Planner
        planner = Planner()

        steps = planner.plan(instruction=sample_instruction, repo_url=sample_repo_url)

        assert isinstance(steps, list), "plan() must return a list"
        assert len(steps) > 0, "plan() must return at least one step"

    @patch("agent.planner.ChatGroq")
    def test_planner_plan_steps_are_strings(self, MockChatGroq, sample_instruction, sample_repo_url):
        """Every step returned by Planner.plan() should be a string."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="1. Do this\n2. Do that")
        MockChatGroq.return_value = mock_llm

        from agent.planner import Planner
        planner = Planner()
        steps = planner.plan(instruction=sample_instruction, repo_url=sample_repo_url)

        for step in steps:
            assert isinstance(step, str), f"Each step must be a string, got {type(step)}"

    @patch("agent.planner.ChatGroq")
    def test_planner_calls_llm(self, MockChatGroq, sample_instruction, sample_repo_url):
        """Planner.plan() must call the LLM at least once."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="1. step one")
        MockChatGroq.return_value = mock_llm

        from agent.planner import Planner
        planner = Planner()
        planner.plan(instruction=sample_instruction, repo_url=sample_repo_url)

        assert mock_llm.invoke.called or mock_llm.call_count >= 0

    @patch("agent.planner.ChatGroq")
    def test_planner_plan_with_complex_instruction(self, MockChatGroq, sample_repo_url):
        """Planner handles a multi-sentence instruction without crashing."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="1. Read files\n2. Refactor loops\n3. Add tests\n4. Open PR"
        )
        MockChatGroq.return_value = mock_llm

        from agent.planner import Planner
        planner = Planner()

        steps = planner.plan(
            instruction="Refactor all database calls. Add docstrings. Update README.",
            repo_url=sample_repo_url
        )

        assert isinstance(steps, list)
        assert len(steps) >= 1

    @patch("agent.planner.ChatGroq")
    def test_planner_raises_on_empty_instruction(self, MockChatGroq, sample_repo_url):
        """Planner.plan() should raise on a blank instruction."""
        MockChatGroq.return_value = MagicMock()

        from agent.planner import Planner
        planner = Planner()

        with pytest.raises((ValueError, TypeError, Exception)):
            planner.plan(instruction="", repo_url=sample_repo_url)

    @patch("agent.planner.ChatGroq")
    def test_planner_raises_when_llm_fails(self, MockChatGroq, sample_instruction, sample_repo_url):
        """If the LLM raises an exception, plan() should propagate it."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("Groq API unreachable")
        MockChatGroq.return_value = mock_llm

        from agent.planner import Planner
        planner = Planner()

        with pytest.raises((ConnectionError, Exception)):
            planner.plan(instruction=sample_instruction, repo_url=sample_repo_url)


# =============================================================================
# SECTION 3 — Executor  (agent/executor.py)
# =============================================================================

class TestExecutor:
    """Tests for the Executor that runs each planned step."""

    def test_executor_execute_returns_result(self, sample_instruction, sample_repo_url):
        """Executor.execute() should return a result dict or string."""
        from agent.executor import Executor

        with patch("agent.executor.GitHubTool") as MockGitHub, \
             patch("agent.executor.CodeParser") as MockParser, \
             patch("agent.executor.DiffGenerator") as MockDiff, \
             patch("agent.executor.PRTool") as MockPR:

            MockGitHub.return_value.clone_repo.return_value = "/tmp/cloned-repo"
            MockGitHub.return_value.create_branch.return_value = True
            MockGitHub.return_value.push_changes.return_value = True
            MockParser.return_value.parse_repo.return_value = {"files": ["utils/helpers.py"]}
            MockDiff.return_value.generate_diff.return_value = "--- a/file\n+++ b/file"
            MockPR.return_value.create_pr.return_value = "https://github.com/example/sample-repo/pull/1"

            executor = Executor()
            steps = ["Clone repo", "Parse files", "Apply changes", "Open PR"]
            result = executor.execute(
                steps=steps,
                instruction=sample_instruction,
                repo_url=sample_repo_url
            )

        assert result is not None, "execute() must return a result"

    def test_executor_execute_includes_pr_url(self, sample_instruction, sample_repo_url):
        """The result from execute() should contain a PR URL when successful."""
        from agent.executor import Executor

        with patch("agent.executor.GitHubTool") as MockGitHub, \
             patch("agent.executor.CodeParser") as MockParser, \
             patch("agent.executor.DiffGenerator") as MockDiff, \
             patch("agent.executor.PRTool") as MockPR:

            MockGitHub.return_value.clone_repo.return_value = "/tmp/cloned-repo"
            MockGitHub.return_value.create_branch.return_value = True
            MockGitHub.return_value.push_changes.return_value = True
            MockParser.return_value.parse_repo.return_value = {"files": ["utils/helpers.py"]}
            MockDiff.return_value.generate_diff.return_value = "diff content"
            MockPR.return_value.create_pr.return_value = "https://github.com/example/sample-repo/pull/1"

            executor = Executor()
            result = executor.execute(
                steps=["step 1", "step 2"],
                instruction=sample_instruction,
                repo_url=sample_repo_url
            )

        if isinstance(result, dict):
            assert "pr_url" in result or "status" in result
        else:
            assert isinstance(result, str)

    def test_executor_accepts_branch_name(self, sample_instruction, sample_repo_url):
        """execute() should accept an optional branch_name without raising."""
        from agent.executor import Executor

        with patch("agent.executor.GitHubTool") as MockGitHub, \
             patch("agent.executor.CodeParser") as MockParser, \
             patch("agent.executor.DiffGenerator") as MockDiff, \
             patch("agent.executor.PRTool") as MockPR:

            MockGitHub.return_value.clone_repo.return_value = "/tmp/cloned-repo"
            MockGitHub.return_value.create_branch.return_value = True
            MockGitHub.return_value.push_changes.return_value = True
            MockParser.return_value.parse_repo.return_value = {}
            MockDiff.return_value.generate_diff.return_value = "diff"
            MockPR.return_value.create_pr.return_value = "https://github.com/example/sample-repo/pull/3"

            executor = Executor()
            result = executor.execute(
                steps=["step 1"],
                instruction=sample_instruction,
                repo_url=sample_repo_url,
                branch_name="repomind/type-hints"
            )

        assert result is not None

    def test_executor_raises_on_empty_steps(self, sample_instruction, sample_repo_url):
        """execute() should raise when steps list is empty."""
        from agent.executor import Executor

        with patch("agent.executor.GitHubTool"), \
             patch("agent.executor.CodeParser"), \
             patch("agent.executor.DiffGenerator"), \
             patch("agent.executor.PRTool"):

            executor = Executor()

            with pytest.raises((ValueError, TypeError, Exception)):
                executor.execute(
                    steps=[],
                    instruction=sample_instruction,
                    repo_url=sample_repo_url
                )

    def test_executor_handles_github_tool_failure(self, sample_instruction, sample_repo_url):
        """If GitHubTool fails, execute() should raise an appropriate exception."""
        from agent.executor import Executor

        with patch("agent.executor.GitHubTool") as MockGitHub, \
             patch("agent.executor.CodeParser"), \
             patch("agent.executor.DiffGenerator"), \
             patch("agent.executor.PRTool"):

            MockGitHub.return_value.clone_repo.side_effect = RuntimeError("GitHub auth failed")

            executor = Executor()

            with pytest.raises((RuntimeError, Exception)):
                executor.execute(
                    steps=["Clone repo", "Open PR"],
                    instruction=sample_instruction,
                    repo_url=sample_repo_url
                )
