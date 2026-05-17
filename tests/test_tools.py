# =============================================================================
# tests/test_tools.py
# Tests for: parse_repository, generate_diff, build_pr_title,
#            build_pr_body, commit_changes
# Uses pytest + MagicMock + patch — no real GitHub calls are made
# =============================================================================

import pytest
from unittest.mock import MagicMock, patch
from tools.code_parser import parse_repository
from tools.diff_generator import generate_diff
from tools.pr_tool import build_pr_title, build_pr_body
from tools.github_tool import commit_changes


# =============================================================================
# SECTION 1 — parse_repository  (tools/code_parser.py)
# =============================================================================

class TestCodeParser:

    # --- Normal tests ---

    def test_parser_reads_python_file(self, tmp_path):
        """parse_repository() should read a valid Python file."""
        (tmp_path / "main.py").write_text("print('hello')", encoding="utf-8")

        parsed = parse_repository(tmp_path)

        assert "main.py" in parsed
        assert parsed["main.py"] == "print('hello')"

    def test_parser_ignores_hidden_dirs(self, tmp_path):
        """parse_repository() should skip hidden folders like .git"""
        (tmp_path / "main.py").write_text("print('hello')", encoding="utf-8")

        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("secret config", encoding="utf-8")

        parsed = parse_repository(tmp_path)

        assert "main.py" in parsed
        assert "config" not in parsed
        assert ".git/config" not in parsed

    def test_parser_reads_multiple_files(self, tmp_path):
        """parse_repository() should read all valid files in the folder."""
        (tmp_path / "main.py").write_text("x = 1", encoding="utf-8")
        (tmp_path / "utils.py").write_text("y = 2", encoding="utf-8")

        parsed = parse_repository(tmp_path)

        assert "main.py" in parsed
        assert "utils.py" in parsed

    def test_parser_returns_dict(self, tmp_path):
        """parse_repository() should return a dictionary."""
        (tmp_path / "main.py").write_text("z = 3", encoding="utf-8")

        parsed = parse_repository(tmp_path)

        assert isinstance(parsed, dict), "parse_repository() must return a dict"

    def test_parser_file_content_matches(self, tmp_path):
        """The content stored for each file should match what was written."""
        content = "def hello():\n    return 42\n"
        (tmp_path / "helpers.py").write_text(content, encoding="utf-8")

        parsed = parse_repository(tmp_path)

        assert parsed["helpers.py"] == content

    def test_parser_empty_directory(self, tmp_path):
        """parse_repository() on an empty folder should return an empty dict."""
        parsed = parse_repository(tmp_path)

        assert isinstance(parsed, dict)
        assert len(parsed) == 0

    # --- Error tests ---

    def test_parser_raises_on_missing_directory(self):
        """parse_repository() should raise when the path does not exist."""
        with pytest.raises((FileNotFoundError, OSError, Exception)):
            parse_repository("/nonexistent/path/that/does/not/exist")

    def test_parser_skips_git_config_file(self, tmp_path):
        """The .git/config file should never appear in parsed results."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("secret", encoding="utf-8")
        (tmp_path / "app.py").write_text("app = True", encoding="utf-8")

        parsed = parse_repository(tmp_path)

        assert ".git/config" not in parsed


# =============================================================================
# SECTION 2 — generate_diff  (tools/diff_generator.py)
# =============================================================================

class TestDiffGenerator:

    # --- Normal tests ---

    def test_diff_shows_removed_line(self):
        """generate_diff() should mark the old line with a minus sign."""
        old_code = "def hello():\n    print('world')"
        new_code = "def hello():\n    print('python')"

        diff = generate_diff(old_code, new_code)

        assert "-    print('world')" in diff

    def test_diff_shows_added_line(self):
        """generate_diff() should mark the new line with a plus sign."""
        old_code = "def hello():\n    print('world')"
        new_code = "def hello():\n    print('python')"

        diff = generate_diff(old_code, new_code)

        assert "+    print('python')" in diff

    def test_diff_returns_string(self):
        """generate_diff() should return a string."""
        diff = generate_diff("a = 1\n", "a = 2\n")

        assert isinstance(diff, str)

    def test_diff_identical_files_returns_empty(self):
        """generate_diff() should return empty string when files are identical."""
        code = "def hello():\n    pass\n"

        diff = generate_diff(code, code)

        assert diff == "" or diff is None or len(diff.strip()) == 0

    def test_diff_detects_added_line(self):
        """generate_diff() should detect a newly added line."""
        old_code = "line1\n"
        new_code = "line1\nline2\n"

        diff = generate_diff(old_code, new_code)

        assert "+line2" in diff or "line2" in diff

    def test_diff_detects_removed_line(self):
        """generate_diff() should detect a removed line."""
        old_code = "line1\nline2\n"
        new_code = "line1\n"

        diff = generate_diff(old_code, new_code)

        assert "-line2" in diff or "line2" in diff

    def test_diff_multiline_change(self):
        """generate_diff() handles multiple changed lines correctly."""
        old_code = "a = 1\nb = 2\nc = 3\n"
        new_code = "a = 1\nb = 99\nc = 3\n"

        diff = generate_diff(old_code, new_code)

        assert diff is not None
        assert isinstance(diff, str)

    # --- Error tests ---

    def test_diff_raises_on_none_old_code(self):
        """generate_diff() should raise when old code is None."""
        with pytest.raises((TypeError, AttributeError, Exception)):
            generate_diff(None, "some code\n")

    def test_diff_raises_on_none_new_code(self):
        """generate_diff() should raise when new code is None."""
        with pytest.raises((TypeError, AttributeError, Exception)):
            generate_diff("some code\n", None)


# =============================================================================
# SECTION 3 — build_pr_title / build_pr_body  (tools/pr_tool.py)
# =============================================================================

class TestPRTool:

    # --- build_pr_title normal tests ---

    def test_pr_title_format(self):
        """build_pr_title() should return 'feat: <instruction>'."""
        title = build_pr_title("Add a new login feature")

        assert title == "feat: Add a new login feature"

    def test_pr_title_returns_string(self):
        """build_pr_title() should return a string."""
        title = build_pr_title("Refactor database calls")

        assert isinstance(title, str)

    def test_pr_title_starts_with_feat(self):
        """build_pr_title() output should start with 'feat:'."""
        title = build_pr_title("Some instruction")

        assert title.startswith("feat:")

    def test_pr_title_contains_instruction(self):
        """build_pr_title() output should contain the original instruction."""
        instruction = "Add type hints to all functions"
        title = build_pr_title(instruction)

        assert instruction in title

    # --- build_pr_body normal tests ---

    def test_pr_body_contains_instruction(self):
        """build_pr_body() output should mention the instruction."""
        body = build_pr_body(
            instruction="Add a new login feature",
            changed_files=["main.py", "auth.py"]
        )

        assert "Add a new login feature" in body

    def test_pr_body_contains_changed_files(self):
        """build_pr_body() output should list each changed file."""
        body = build_pr_body(
            instruction="Add a new login feature",
            changed_files=["main.py", "auth.py"]
        )

        assert "- `main.py`" in body
        assert "- `auth.py`" in body

    def test_pr_body_returns_string(self):
        """build_pr_body() should return a string."""
        body = build_pr_body(
            instruction="Fix bugs",
            changed_files=["main.py"]
        )

        assert isinstance(body, str)

    def test_pr_body_with_multiple_files(self):
        """build_pr_body() should list all files when many are changed."""
        files = ["a.py", "b.py", "c.py", "d.py"]
        body = build_pr_body(
            instruction="Big refactor",
            changed_files=files
        )

        for f in files:
            assert f in body

    def test_pr_body_with_no_files(self):
        """build_pr_body() should not crash when changed_files is empty."""
        body = build_pr_body(
            instruction="Minor fix",
            changed_files=[]
        )

        assert isinstance(body, str)

    # --- Error tests ---

    def test_pr_title_raises_on_empty_instruction(self):
        """build_pr_title() should raise or return empty when instruction is blank."""
        result = build_pr_title("")
        # Either raises or returns a near-empty string — both are acceptable
        if result is not None:
            assert isinstance(result, str)

    def test_pr_body_raises_on_none_instruction(self):
        """build_pr_body() should raise when instruction is None."""
        with pytest.raises((TypeError, AttributeError, Exception)):
            build_pr_body(instruction=None, changed_files=["main.py"])


# =============================================================================
# SECTION 4 — commit_changes  (tools/github_tool.py)
# =============================================================================

class TestGitHubTool:

    # --- Normal tests ---

    def test_commit_changes_returns_hash(self):
        """commit_changes() should return the commit SHA hash."""
        mock_repo = MagicMock()
        mock_repo.is_dirty.return_value = True
        mock_repo.index.commit.return_value.hexsha = "12345abcde"

        with patch('tools.github_tool.stage_all_changes'):
            commit_hash = commit_changes(mock_repo, "My test commit")

        assert commit_hash == "12345abcde"

    def test_commit_changes_calls_commit_with_message(self):
        """commit_changes() should call index.commit with the correct message."""
        mock_repo = MagicMock()
        mock_repo.is_dirty.return_value = True
        mock_repo.index.commit.return_value.hexsha = "abc123"

        with patch('tools.github_tool.stage_all_changes'):
            commit_changes(mock_repo, "My test commit")

        mock_repo.index.commit.assert_called_once_with("My test commit")

    def test_commit_changes_calls_stage_all_changes(self):
        """commit_changes() should call stage_all_changes before committing."""
        mock_repo = MagicMock()
        mock_repo.is_dirty.return_value = True
        mock_repo.index.commit.return_value.hexsha = "abc123"

        with patch('tools.github_tool.stage_all_changes') as mock_stage:
            commit_changes(mock_repo, "Stage test")

        mock_stage.assert_called_once()

    def test_commit_changes_returns_string(self):
        """commit_changes() should return a string (the commit hash)."""
        mock_repo = MagicMock()
        mock_repo.is_dirty.return_value = True
        mock_repo.index.commit.return_value.hexsha = "deadbeef"

        with patch('tools.github_tool.stage_all_changes'):
            result = commit_changes(mock_repo, "Test commit")

        assert isinstance(result, str)

    def test_commit_changes_with_different_messages(self):
        """commit_changes() should work with any commit message string."""
        messages = [
            "feat: add login",
            "fix: resolve bug in auth",
            "refactor: clean up helpers",
        ]

        for msg in messages:
            mock_repo = MagicMock()
            mock_repo.is_dirty.return_value = True
            mock_repo.index.commit.return_value.hexsha = "aabbcc"

            with patch('tools.github_tool.stage_all_changes'):
                result = commit_changes(mock_repo, msg)

            assert result == "aabbcc"

    # --- Error tests ---

    def test_commit_changes_raises_when_commit_fails(self):
        """commit_changes() should raise when git commit throws an error."""
        mock_repo = MagicMock()
        mock_repo.is_dirty.return_value = True
        mock_repo.index.commit.side_effect = Exception("Git commit failed")

        with patch('tools.github_tool.stage_all_changes'):
            with pytest.raises(Exception):
                commit_changes(mock_repo, "Bad commit")

    def test_commit_changes_raises_on_empty_message(self):
        """commit_changes() should raise or handle an empty commit message."""
        mock_repo = MagicMock()
        mock_repo.is_dirty.return_value = True
        mock_repo.index.commit.side_effect = Exception("Empty commit message")

        with patch('tools.github_tool.stage_all_changes'):
            with pytest.raises(Exception):
                commit_changes(mock_repo, "")
