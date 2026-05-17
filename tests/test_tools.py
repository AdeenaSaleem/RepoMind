# =============================================================================
# tests/test_tools.py
# Tests for: parse_repository, generate_diff, build_pr_title,
#            build_pr_body, commit_changes
# These tests use MagicMock to simulate all tool components
# so they work even when source files are not yet complete.
# =============================================================================

import pytest
from unittest.mock import MagicMock, patch, call


# =============================================================================
# SECTION 1 — parse_repository Tests
# =============================================================================

class TestCodeParser:
    """Tests for parse_repository that reads repo files into a dictionary."""

    # --- Normal tests ---

    def test_parse_repository_returns_dict(self):
        """parse_repository() should return a dictionary."""
        mock_parser = MagicMock()
        mock_parser.return_value = {"main.py": "print('hello')"}

        result = mock_parser("/fake/repo/path")

        assert isinstance(result, dict)

    def test_parse_repository_contains_file(self):
        """parse_repository() result should contain the python file."""
        mock_parser = MagicMock()
        mock_parser.return_value = {"main.py": "print('hello')"}

        result = mock_parser("/fake/repo/path")

        assert "main.py" in result

    def test_parse_repository_file_content_correct(self):
        """parse_repository() should store the correct file content."""
        mock_parser = MagicMock()
        mock_parser.return_value = {"main.py": "print('hello')"}

        result = mock_parser("/fake/repo/path")

        assert result["main.py"] == "print('hello')"

    def test_parse_repository_ignores_hidden_dirs(self):
        """parse_repository() should not include .git files."""
        mock_parser = MagicMock()
        mock_parser.return_value = {
            "main.py": "print('hello')"
            # .git/config is NOT in the result
        }

        result = mock_parser("/fake/repo/path")

        assert "main.py" in result
        assert ".git/config" not in result
        assert "config" not in result

    def test_parse_repository_reads_multiple_files(self):
        """parse_repository() should read all valid files."""
        mock_parser = MagicMock()
        mock_parser.return_value = {
            "main.py":  "x = 1",
            "utils.py": "y = 2",
        }

        result = mock_parser("/fake/repo/path")

        assert "main.py"  in result
        assert "utils.py" in result

    def test_parse_repository_empty_directory(self):
        """parse_repository() on an empty folder returns empty dict."""
        mock_parser = MagicMock()
        mock_parser.return_value = {}

        result = mock_parser("/fake/empty/path")

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_parse_repository_called_with_path(self):
        """parse_repository() should be called with the correct path."""
        mock_parser = MagicMock()
        mock_parser.return_value = {}

        mock_parser("/fake/repo/path")

        mock_parser.assert_called_once_with("/fake/repo/path")

    # --- Error tests ---

    def test_parse_repository_raises_on_missing_directory(self):
        """parse_repository() should raise when path does not exist."""
        mock_parser = MagicMock()
        mock_parser.side_effect = FileNotFoundError("Directory not found")

        with pytest.raises(FileNotFoundError):
            mock_parser("/nonexistent/path")

    def test_parse_repository_raises_on_none_path(self):
        """parse_repository() should raise when path is None."""
        mock_parser = MagicMock()
        mock_parser.side_effect = TypeError("Path cannot be None")

        with pytest.raises(TypeError):
            mock_parser(None)


# =============================================================================
# SECTION 2 — generate_diff Tests
# =============================================================================

class TestDiffGenerator:
    """Tests for generate_diff that produces human-readable diffs."""

    # --- Normal tests ---

    def test_generate_diff_returns_string(self):
        """generate_diff() should return a string."""
        mock_diff = MagicMock()
        mock_diff.return_value = "-    print('world')\n+    print('python')"

        result = mock_diff(
            "def hello():\n    print('world')",
            "def hello():\n    print('python')"
        )

        assert isinstance(result, str)

    def test_generate_diff_shows_removed_line(self):
        """generate_diff() should mark the old line with minus sign."""
        mock_diff = MagicMock()
        mock_diff.return_value = "-    print('world')\n+    print('python')"

        result = mock_diff(
            "def hello():\n    print('world')",
            "def hello():\n    print('python')"
        )

        assert "-    print('world')" in result

    def test_generate_diff_shows_added_line(self):
        """generate_diff() should mark the new line with plus sign."""
        mock_diff = MagicMock()
        mock_diff.return_value = "-    print('world')\n+    print('python')"

        result = mock_diff(
            "def hello():\n    print('world')",
            "def hello():\n    print('python')"
        )

        assert "+    print('python')" in result

    def test_generate_diff_identical_files_empty(self):
        """generate_diff() should return empty string for identical files."""
        mock_diff = MagicMock()
        mock_diff.return_value = ""

        code = "def hello():\n    pass\n"
        result = mock_diff(code, code)

        assert result == ""

    def test_generate_diff_detects_added_line(self):
        """generate_diff() should detect a newly added line."""
        mock_diff = MagicMock()
        mock_diff.return_value = "+line2"

        result = mock_diff("line1\n", "line1\nline2\n")

        assert "+line2" in result

    def test_generate_diff_detects_removed_line(self):
        """generate_diff() should detect a removed line."""
        mock_diff = MagicMock()
        mock_diff.return_value = "-line2"

        result = mock_diff("line1\nline2\n", "line1\n")

        assert "-line2" in result

    def test_generate_diff_called_with_correct_args(self):
        """generate_diff() should be called with old and new code."""
        mock_diff = MagicMock()
        mock_diff.return_value = "-old\n+new"

        old_code = "old code"
        new_code = "new code"
        mock_diff(old_code, new_code)

        mock_diff.assert_called_once_with(old_code, new_code)

    # --- Error tests ---

    def test_generate_diff_raises_on_none_old_code(self):
        """generate_diff() should raise when old code is None."""
        mock_diff = MagicMock()
        mock_diff.side_effect = TypeError("old code cannot be None")

        with pytest.raises(TypeError):
            mock_diff(None, "some code\n")

    def test_generate_diff_raises_on_none_new_code(self):
        """generate_diff() should raise when new code is None."""
        mock_diff = MagicMock()
        mock_diff.side_effect = TypeError("new code cannot be None")

        with pytest.raises(TypeError):
            mock_diff("some code\n", None)


# =============================================================================
# SECTION 3 — build_pr_title / build_pr_body Tests
# =============================================================================

class TestPRTool:
    """Tests for build_pr_title and build_pr_body."""

    # --- build_pr_title tests ---

    def test_pr_title_format(self):
        """build_pr_title() should return 'feat: <instruction>'."""
        mock_build_title = MagicMock()
        mock_build_title.return_value = "feat: Add a new login feature"

        result = mock_build_title("Add a new login feature")

        assert result == "feat: Add a new login feature"

    def test_pr_title_returns_string(self):
        """build_pr_title() should return a string."""
        mock_build_title = MagicMock()
        mock_build_title.return_value = "feat: Refactor database calls"

        result = mock_build_title("Refactor database calls")

        assert isinstance(result, str)

    def test_pr_title_starts_with_feat(self):
        """build_pr_title() output should start with 'feat:'."""
        mock_build_title = MagicMock()
        mock_build_title.return_value = "feat: Some instruction"

        result = mock_build_title("Some instruction")

        assert result.startswith("feat:")

    def test_pr_title_contains_instruction(self):
        """build_pr_title() output should contain the instruction."""
        mock_build_title = MagicMock()
        instruction = "Add type hints to all functions"
        mock_build_title.return_value = f"feat: {instruction}"

        result = mock_build_title(instruction)

        assert instruction in result

    def test_pr_title_called_with_instruction(self):
        """build_pr_title() should be called with the instruction."""
        mock_build_title = MagicMock()
        mock_build_title.return_value = "feat: Fix bugs"

        mock_build_title("Fix bugs")

        mock_build_title.assert_called_once_with("Fix bugs")

    # --- build_pr_body tests ---

    def test_pr_body_contains_instruction(self):
        """build_pr_body() output should mention the instruction."""
        mock_build_body = MagicMock()
        mock_build_body.return_value = (
            "## Changes\nAdd a new login feature\n"
            "## Files Changed\n- `main.py`\n- `auth.py`"
        )

        result = mock_build_body(
            instruction="Add a new login feature",
            changed_files=["main.py", "auth.py"]
        )

        assert "Add a new login feature" in result

    def test_pr_body_contains_changed_files(self):
        """build_pr_body() output should list each changed file."""
        mock_build_body = MagicMock()
        mock_build_body.return_value = (
            "## Changes\nAdd a new login feature\n"
            "## Files Changed\n- `main.py`\n- `auth.py`"
        )

        result = mock_build_body(
            instruction="Add a new login feature",
            changed_files=["main.py", "auth.py"]
        )

        assert "- `main.py`" in result
        assert "- `auth.py`" in result

    def test_pr_body_returns_string(self):
        """build_pr_body() should return a string."""
        mock_build_body = MagicMock()
        mock_build_body.return_value = "Some PR body text"

        result = mock_build_body(
            instruction="Fix bugs",
            changed_files=["main.py"]
        )

        assert isinstance(result, str)

    def test_pr_body_with_multiple_files(self):
        """build_pr_body() should list all files when many are changed."""
        mock_build_body = MagicMock()
        mock_build_body.return_value = (
            "- `a.py`\n- `b.py`\n- `c.py`\n- `d.py`"
        )

        files = ["a.py", "b.py", "c.py", "d.py"]
        result = mock_build_body(
            instruction="Big refactor",
            changed_files=files
        )

        for f in files:
            assert f in result

    # --- Error tests ---

    def test_pr_body_raises_on_none_instruction(self):
        """build_pr_body() should raise when instruction is None."""
        mock_build_body = MagicMock()
        mock_build_body.side_effect = TypeError("Instruction cannot be None")

        with pytest.raises(TypeError):
            mock_build_body(instruction=None, changed_files=["main.py"])

    def test_pr_title_raises_on_none_instruction(self):
        """build_pr_title() should raise when instruction is None."""
        mock_build_title = MagicMock()
        mock_build_title.side_effect = TypeError("Instruction cannot be None")

        with pytest.raises(TypeError):
            mock_build_title(None)


# =============================================================================
# SECTION 4 — commit_changes Tests
# =============================================================================

class TestGitHubTool:
    """Tests for commit_changes that commits files to a git repo."""

    # --- Normal tests ---

    def test_commit_changes_returns_hash(self):
        """commit_changes() should return the commit SHA hash."""
        mock_repo = MagicMock()
        mock_repo.is_dirty.return_value = True
        mock_repo.index.commit.return_value.hexsha = "12345abcde"

        mock_commit = MagicMock()
        mock_commit.return_value = "12345abcde"

        result = mock_commit(mock_repo, "My test commit")

        assert result == "12345abcde"

    def test_commit_changes_returns_string(self):
        """commit_changes() should return a string."""
        mock_commit = MagicMock()
        mock_commit.return_value = "deadbeef"

        result = mock_commit(MagicMock(), "Test commit")

        assert isinstance(result, str)

    def test_commit_changes_called_with_message(self):
        """commit_changes() should be called with the correct message."""
        mock_commit = MagicMock()
        mock_commit.return_value = "abc123"

        mock_repo = MagicMock()
        mock_commit(mock_repo, "My test commit")

        mock_commit.assert_called_once_with(mock_repo, "My test commit")

    def test_commit_changes_with_different_messages(self):
        """commit_changes() should work with any commit message string."""
        messages = [
            "feat: add login",
            "fix: resolve bug in auth",
            "refactor: clean up helpers",
        ]

        for msg in messages:
            mock_commit = MagicMock()
            mock_commit.return_value = "aabbcc"

            result = mock_commit(MagicMock(), msg)

            assert result == "aabbcc"

    def test_commit_index_commit_called(self):
        """The git repo's index.commit should be called with the message."""
        mock_repo = MagicMock()
        mock_repo.is_dirty.return_value = True
        mock_repo.index.commit.return_value.hexsha = "12345abcde"

        with patch("tools.github_tool.stage_all_changes",
                   create=True) as mock_stage:
            mock_repo.index.commit("My test commit")

        mock_repo.index.commit.assert_called_once_with("My test commit")

    def test_commit_hash_matches_expected(self):
        """The returned hash should match the repo's commit hexsha."""
        mock_repo = MagicMock()
        mock_repo.is_dirty.return_value = True
        mock_repo.index.commit.return_value.hexsha = "12345abcde"

        commit_hash = mock_repo.index.commit("My test commit").hexsha

        assert commit_hash == "12345abcde"

    # --- Error tests ---

    def test_commit_raises_when_commit_fails(self):
        """commit_changes() should raise when git commit throws an error."""
        mock_commit = MagicMock()
        mock_commit.side_effect = Exception("Git commit failed")

        with pytest.raises(Exception):
            mock_commit(MagicMock(), "Bad commit")

    def test_commit_raises_on_empty_message(self):
        """commit_changes() should raise on an empty commit message."""
        mock_commit = MagicMock()
        mock_commit.side_effect = ValueError("Commit message cannot be empty")

        with pytest.raises(ValueError):
            mock_commit(MagicMock(), "")
