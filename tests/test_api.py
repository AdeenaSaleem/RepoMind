# =============================================================================
# tests/test_api.py
# Integration tests for: POST /run, GET /status, POST /refine
# Also tests: Pydantic models (RunRequest, JobStatus) and job_manager
# Uses FastAPI TestClient + MagicMock + patch — no real LLM/GitHub calls
# =============================================================================

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from pydantic import ValidationError
from api.main import app
from api.schemas import RunRequest, JobStatus
from utils.job_manager import job_manager


client = TestClient(app)


# =============================================================================
# SECTION 1 — Pydantic Model Tests
# =============================================================================

class TestPydanticModels:

    # --- Normal tests ---

    def test_run_request_valid(self):
        """RunRequest accepts repo_url and instruction."""
        req = RunRequest(
            repo_url="https://github.com/QuantumLogicsLabs/RepoMind",
            instruction="Fix bugs"
        )

        assert req.repo_url == "https://github.com/QuantumLogicsLabs/RepoMind"
        assert req.instruction == "Fix bugs"

    def test_run_request_default_branch_name(self):
        """RunRequest should set a default branch_name automatically."""
        req = RunRequest(
            repo_url="https://github.com/QuantumLogicsLabs/RepoMind",
            instruction="Fix bugs"
        )

        assert req.branch_name == "repomind/auto-fix"

    def test_run_request_custom_branch_name(self):
        """RunRequest should accept a custom branch_name."""
        req = RunRequest(
            repo_url="https://github.com/QuantumLogicsLabs/RepoMind",
            instruction="Fix bugs",
            branch_name="repomind/custom-branch"
        )

        assert req.branch_name == "repomind/custom-branch"

    def test_run_request_returns_correct_instruction(self):
        """RunRequest should store the exact instruction provided."""
        req = RunRequest(
            repo_url="https://github.com/QuantumLogicsLabs/RepoMind",
            instruction="Add type hints to all functions"
        )

        assert req.instruction == "Add type hints to all functions"

    # --- Error tests ---

    def test_run_request_missing_instruction_raises(self):
        """RunRequest should raise ValidationError when instruction is missing."""
        with pytest.raises(ValidationError):
            RunRequest(repo_url="https://github.com/test")

    def test_run_request_missing_repo_url_raises(self):
        """RunRequest should raise ValidationError when repo_url is missing."""
        with pytest.raises(ValidationError):
            RunRequest(instruction="Fix bugs")

    def test_run_request_empty_body_raises(self):
        """RunRequest should raise ValidationError when both fields are missing."""
        with pytest.raises(ValidationError):
            RunRequest()


# =============================================================================
# SECTION 2 — JobManager Tests
# =============================================================================

class TestJobManager:

    # --- Normal tests ---

    def test_create_job_returns_string_id(self):
        """create_job() should return a non-empty string job ID."""
        job_id = job_manager.create_job(
            "https://github.com/test/repo",
            "test instruction"
        )

        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_create_job_initial_status_is_queued(self):
        """A newly created job should have status queued."""
        job_id = job_manager.create_job(
            "https://github.com/test/repo",
            "test instruction"
        )
        job = job_manager.get(job_id)

        assert job.status == JobStatus.queued

    def test_get_job_returns_correct_repo_url(self):
        """get() should return the job with the correct repo_url."""
        job_id = job_manager.create_job(
            "https://github.com/test/repo",
            "test instruction"
        )
        job = job_manager.get(job_id)

        assert job.repo_url == "https://github.com/test/repo"

    def test_update_job_status_to_running(self):
        """update() should change job status to running."""
        job_id = job_manager.create_job(
            "https://github.com/test/repo",
            "test instruction"
        )
        job_manager.update(job_id, status=JobStatus.running)
        updated_job = job_manager.get(job_id)

        assert updated_job.status == JobStatus.running

    def test_update_job_with_pr_url(self):
        """update() should store the pr_url on the job."""
        job_id = job_manager.create_job(
            "https://github.com/test/repo",
            "test instruction"
        )
        job_manager.update(
            job_id,
            status=JobStatus.running,
            pr_url="https://github.com/fake/pull/1"
        )
        updated_job = job_manager.get(job_id)

        assert updated_job.pr_url == "https://github.com/fake/pull/1"

    def test_update_job_status_to_completed(self):
        """update() should allow setting status to completed."""
        job_id = job_manager.create_job(
            "https://github.com/test/repo",
            "test instruction"
        )
        job_manager.update(job_id, status=JobStatus.running)
        job_manager.update(
            job_id,
            status=JobStatus.completed,
            pr_url="https://github.com/fake/pull/2"
        )
        final_job = job_manager.get(job_id)

        assert final_job.status == JobStatus.completed

    def test_unique_job_ids(self):
        """Each call to create_job() should produce a different ID."""
        ids = {
            job_manager.create_job(
                "https://github.com/test/repo",
                f"instruction {i}"
            )
            for i in range(5)
        }

        assert len(ids) == 5

    # --- Error tests ---

    def test_get_nonexistent_job_raises(self):
        """get() should raise an exception for an unknown job ID."""
        with pytest.raises(Exception):
            job_manager.get("this_job_does_not_exist")


# =============================================================================
# SECTION 3 — POST /run Endpoint Tests
# =============================================================================

class TestRunEndpoint:

    # --- Normal tests ---

    def test_run_returns_200(self):
        """POST /run with valid data should return HTTP 200."""
        with patch("api.routes.job_manager") as mock_jm:
            mock_jm.create_job.return_value = "test-job-id-001"
            mock_jm.get.return_value = MagicMock(
                status=JobStatus.queued,
                repo_url="https://github.com/QuantumLogicsLabs/RepoMind"
            )

            response = client.post("/run", json={
                "repo_url": "https://github.com/QuantumLogicsLabs/RepoMind",
                "instruction": "Add type hints"
            })

        assert response.status_code == 200

    def test_run_returns_job_id(self):
        """POST /run response should contain a job_id."""
        with patch("api.routes.job_manager") as mock_jm:
            mock_jm.create_job.return_value = "test-job-id-002"
            mock_jm.get.return_value = MagicMock(
                status=JobStatus.queued,
                repo_url="https://github.com/QuantumLogicsLabs/RepoMind"
            )

            response = client.post("/run", json={
                "repo_url": "https://github.com/QuantumLogicsLabs/RepoMind",
                "instruction": "Refactor code"
            })

        assert "job_id" in response.json()

    def test_run_initial_status_is_queued(self):
        """POST /run response status should be queued."""
        with patch("api.routes.job_manager") as mock_jm:
            mock_jm.create_job.return_value = "test-job-id-003"
            mock_jm.get.return_value = MagicMock(
                status=JobStatus.queued,
                repo_url="https://github.com/QuantumLogicsLabs/RepoMind"
            )

            response = client.post("/run", json={
                "repo_url": "https://github.com/QuantumLogicsLabs/RepoMind",
                "instruction": "Fix bugs"
            })

        assert response.json()["status"] == "queued"

    def test_run_with_custom_branch_name(self):
        """POST /run should accept an optional branch_name field."""
        with patch("api.routes.job_manager") as mock_jm:
            mock_jm.create_job.return_value = "test-job-id-004"
            mock_jm.get.return_value = MagicMock(
                status=JobStatus.queued,
                repo_url="https://github.com/QuantumLogicsLabs/RepoMind"
            )

            response = client.post("/run", json={
                "repo_url": "https://github.com/QuantumLogicsLabs/RepoMind",
                "instruction": "Refactor db calls",
                "branch_name": "repomind/custom-branch"
            })

        assert response.status_code == 200

    # --- Error tests ---

    def test_run_returns_422_missing_instruction(self):
        """POST /run without instruction should return HTTP 422."""
        response = client.post("/run", json={
            "repo_url": "https://github.com/QuantumLogicsLabs/RepoMind"
        })

        assert response.status_code == 422

    def test_run_returns_422_missing_repo_url(self):
        """POST /run without repo_url should return HTTP 422."""
        response = client.post("/run", json={
            "instruction": "Fix bugs"
        })

        assert response.status_code == 422

    def test_run_returns_422_empty_body(self):
        """POST /run with empty body should return HTTP 422."""
        response = client.post("/run", json={})

        assert response.status_code == 422


# =============================================================================
# SECTION 4 — GET /status Endpoint Tests
# =============================================================================

class TestStatusEndpoint:

    # --- Normal tests ---

    def test_status_returns_200_for_existing_job(self):
        """GET /status/{job_id} should return 200 for a known job."""
        with patch("api.routes.job_manager") as mock_jm:
            mock_jm.create_job.return_value = "test-job-id-005"
            mock_jm.get.return_value = MagicMock(
                status=JobStatus.queued,
                repo_url="https://github.com/QuantumLogicsLabs/RepoMind",
                pr_url=None
            )

            run_resp = client.post("/run", json={
                "repo_url": "https://github.com/QuantumLogicsLabs/RepoMind",
                "instruction": "Fix bugs"
            })
            job_id = run_resp.json()["job_id"]

            status_resp = client.get(f"/status/{job_id}")

        assert status_resp.status_code == 200

    def test_status_response_has_status_field(self):
        """GET /status/{job_id} response should contain a status field."""
        with patch("api.routes.job_manager") as mock_jm:
            mock_jm.create_job.return_value = "test-job-id-006"
            mock_jm.get.return_value = MagicMock(
                status=JobStatus.queued,
                repo_url="https://github.com/QuantumLogicsLabs/RepoMind",
                pr_url=None
            )

            run_resp = client.post("/run", json={
                "repo_url": "https://github.com/QuantumLogicsLabs/RepoMind",
                "instruction": "Fix bugs"
            })
            job_id = run_resp.json()["job_id"]

            status_resp = client.get(f"/status/{job_id}")

        assert "status" in status_resp.json()

    def test_status_valid_status_value(self):
        """GET /status/{job_id} status should be one of the valid values."""
        with patch("api.routes.job_manager") as mock_jm:
            mock_jm.create_job.return_value = "test-job-id-007"
            mock_jm.get.return_value = MagicMock(
                status=JobStatus.queued,
                repo_url="https://github.com/QuantumLogicsLabs/RepoMind",
                pr_url=None
            )

            run_resp = client.post("/run", json={
                "repo_url": "https://github.com/QuantumLogicsLabs/RepoMind",
                "instruction": "Fix bugs"
            })
            job_id = run_resp.json()["job_id"]

            status_resp = client.get(f"/status/{job_id}")
            status_value = status_resp.json()["status"]

        assert status_value in ["queued", "running", "completed", "failed"]

    # --- Error tests ---

    def test_status_returns_404_for_unknown_job(self):
        """GET /status/{job_id} should return 404 for a non-existent job."""
        with patch("api.routes.job_manager") as mock_jm:
            mock_jm.get.side_effect = Exception("Job not found")

            response = client.get("/status/fake_job_that_does_not_exist_12345")

        assert response.status_code == 404


# =============================================================================
# SECTION 5 — POST /refine Endpoint Tests
# =============================================================================

class TestRefineEndpoint:

    # --- Normal tests ---

    def test_refine_returns_200(self):
        """POST /refine should return 200 for a valid existing job."""
        with patch("api.routes.job_manager") as mock_jm:
            mock_jm.create_job.return_value = "test-job-id-008"
            mock_jm.get.return_value = MagicMock(
                status=JobStatus.queued,
                repo_url="https://github.com/QuantumLogicsLabs/RepoMind",
                pr_url=None,
                instruction="Fix bugs"
            )

            run_resp = client.post("/run", json={
                "repo_url": "https://github.com/QuantumLogicsLabs/RepoMind",
                "instruction": "Fix bugs"
            })
            job_id = run_resp.json()["job_id"]

            refine_resp = client.post("/refine", json={
                "job_id": job_id,
                "instruction": "Also add docstrings"
            })

        assert refine_resp.status_code == 200

    def test_refine_response_has_job_id(self):
        """POST /refine response should include the job_id."""
        with patch("api.routes.job_manager") as mock_jm:
            mock_jm.create_job.return_value = "test-job-id-009"
            mock_jm.get.return_value = MagicMock(
                status=JobStatus.queued,
                repo_url="https://github.com/QuantumLogicsLabs/RepoMind",
                pr_url=None,
                instruction="Fix bugs"
            )

            run_resp = client.post("/run", json={
                "repo_url": "https://github.com/QuantumLogicsLabs/RepoMind",
                "instruction": "Fix bugs"
            })
            job_id = run_resp.json()["job_id"]

            refine_resp = client.post("/refine", json={
                "job_id": job_id,
                "instruction": "Also add docstrings"
            })

        assert refine_resp.json()["job_id"] == job_id

    def test_refine_same_job_id_returned(self):
        """POST /refine should return the same job_id that was passed in."""
        with patch("api.routes.job_manager") as mock_jm:
            mock_jm.create_job.return_value = "test-job-id-010"
            mock_jm.get.return_value = MagicMock(
                status=JobStatus.queued,
                repo_url="https://github.com/QuantumLogicsLabs/RepoMind",
                pr_url=None,
                instruction="Step 1"
            )

            run_resp = client.post("/run", json={
                "repo_url": "https://github.com/QuantumLogicsLabs/RepoMind",
                "instruction": "Step 1"
            })
            job_id = run_resp.json()["job_id"]

            refine_resp = client.post("/refine", json={
                "job_id": job_id,
                "instruction": "Step 2"
            })

        assert refine_resp.json()["job_id"] == job_id

    # --- Error tests ---

    def test_refine_returns_422_missing_job_id(self):
        """POST /refine without job_id should return HTTP 422."""
        response = client.post("/refine", json={
            "instruction": "Add docstrings"
        })

        assert response.status_code == 422

    def test_refine_returns_422_missing_instruction(self):
        """POST /refine without instruction should return HTTP 422."""
        response = client.post("/refine", json={
            "job_id": "abc-123"
        })

        assert response.status_code == 422

    def test_refine_returns_422_empty_body(self):
        """POST /refine with empty body should return HTTP 422."""
        response = client.post("/refine", json={})

        assert response.status_code == 422


# =============================================================================
# SECTION 6 — API Health Tests
# =============================================================================

class TestAPIHealth:

    def test_docs_page_accessible(self):
        """FastAPI /docs page should return 200."""
        response = client.get("/docs")

        assert response.status_code == 200

    def test_openapi_json_accessible(self):
        """FastAPI /openapi.json should return 200."""
        response = client.get("/openapi.json")

        assert response.status_code == 200

    def test_openapi_contains_run_path(self):
        """OpenAPI schema should list the /run endpoint."""
        response = client.get("/openapi.json")
        paths = response.json().get("paths", {})

        assert "/run" in paths

    def test_openapi_contains_status_path(self):
        """OpenAPI schema should list the /status endpoint."""
        response = client.get("/openapi.json")
        paths = response.json().get("paths", {})
        status_paths = [p for p in paths if "status" in p]

        assert len(status_paths) > 0

    def test_openapi_contains_refine_path(self):
        """OpenAPI schema should list the /refine endpoint."""
        response = client.get("/openapi.json")
        paths = response.json().get("paths", {})

        assert "/refine" in paths
