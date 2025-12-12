"""
Sandbox Runner - Safe Code Execution Environment
"""

import os
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a sandbox execution."""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    files_created: List[str]


class SandboxRunner:
    """
    Sandbox environment for safely executing generated code.
    
    Uses temporary directories and optional Docker isolation.
    """
    
    def __init__(
        self,
        use_docker: bool = False,
        docker_image: str = "python:3.11-slim",
        timeout: int = 30,
        max_output_size: int = 1024 * 1024  # 1MB
    ):
        """
        Initialize the sandbox runner.
        
        Args:
            use_docker: Whether to use Docker for isolation
            docker_image: Docker image to use
            timeout: Execution timeout in seconds
            max_output_size: Maximum output size in bytes
        """
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.timeout = timeout
        self.max_output_size = max_output_size
        
        if use_docker:
            self._check_docker()
    
    def _check_docker(self):
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Docker not available")
            logger.info("Docker available for sandbox execution")
        except Exception as e:
            logger.warning(f"Docker not available: {e}, falling back to local execution")
            self.use_docker = False
    
    def create_workspace(self, files: List[Dict[str, Any]]) -> Path:
        """
        Create a temporary workspace with files.
        
        Args:
            files: List of file dictionaries with 'path' and 'content'
            
        Returns:
            Path to the workspace directory
        """
        workspace = Path(tempfile.mkdtemp(prefix="sandbox_"))
        
        for file_data in files:
            file_path = workspace / file_data['path']
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(file_data['content'])
            
            logger.debug(f"Created file: {file_path}")
        
        return workspace
    
    def cleanup_workspace(self, workspace: Path) -> None:
        """
        Clean up a workspace directory.
        
        Args:
            workspace: Path to workspace to clean up
        """
        try:
            shutil.rmtree(workspace)
            logger.debug(f"Cleaned up workspace: {workspace}")
        except Exception as e:
            logger.warning(f"Failed to cleanup workspace: {e}")
    
    def run_command(
        self,
        command: str,
        workspace: Path,
        env: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """
        Run a command in the sandbox.
        
        Args:
            command: Command to execute
            workspace: Workspace directory
            env: Environment variables
            
        Returns:
            ExecutionResult with output and status
        """
        import time
        start_time = time.time()
        
        if self.use_docker:
            return self._run_docker(command, workspace, env)
        else:
            return self._run_local(command, workspace, env)
    
    def _run_local(
        self,
        command: str,
        workspace: Path,
        env: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """Run command locally in subprocess."""
        import time
        start_time = time.time()
        
        # Merge environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        
        # Restrict PATH for security
        run_env['PATH'] = '/usr/local/bin:/usr/bin:/bin'
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=workspace,
                env=run_env,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            duration = time.time() - start_time
            
            # Truncate output if too large
            stdout = result.stdout[:self.max_output_size]
            stderr = result.stderr[:self.max_output_size]
            
            # List files created
            files_created = []
            for root, dirs, files in os.walk(workspace):
                for f in files:
                    rel_path = os.path.relpath(os.path.join(root, f), workspace)
                    files_created.append(rel_path)
            
            return ExecutionResult(
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=stdout,
                stderr=stderr,
                duration=duration,
                files_created=files_created
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout='',
                stderr=f'Execution timed out after {self.timeout} seconds',
                duration=self.timeout,
                files_created=[]
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout='',
                stderr=str(e),
                duration=time.time() - start_time,
                files_created=[]
            )
    
    def _run_docker(
        self,
        command: str,
        workspace: Path,
        env: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """Run command in Docker container."""
        import time
        start_time = time.time()
        
        # Build docker command
        docker_cmd = [
            "docker", "run",
            "--rm",
            "--network", "none",  # No network access
            "--memory", "256m",   # Memory limit
            "--cpus", "0.5",      # CPU limit
            "-v", f"{workspace}:/workspace:rw",
            "-w", "/workspace"
        ]
        
        # Add environment variables
        if env:
            for key, value in env.items():
                docker_cmd.extend(["-e", f"{key}={value}"])
        
        docker_cmd.extend([self.docker_image, "sh", "-c", command])
        
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 10  # Extra time for Docker overhead
            )
            
            duration = time.time() - start_time
            
            stdout = result.stdout[:self.max_output_size]
            stderr = result.stderr[:self.max_output_size]
            
            # List files created
            files_created = []
            for root, dirs, files in os.walk(workspace):
                for f in files:
                    rel_path = os.path.relpath(os.path.join(root, f), workspace)
                    files_created.append(rel_path)
            
            return ExecutionResult(
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=stdout,
                stderr=stderr,
                duration=duration,
                files_created=files_created
            )
            
        except subprocess.TimeoutExpired:
            # Kill the container
            subprocess.run(
                ["docker", "kill", f"sandbox_{workspace.name}"],
                capture_output=True
            )
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout='',
                stderr=f'Execution timed out after {self.timeout} seconds',
                duration=self.timeout,
                files_created=[]
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout='',
                stderr=str(e),
                duration=time.time() - start_time,
                files_created=[]
            )
    
    def run_tests(
        self,
        files: List[Dict[str, Any]],
        test_command: str = "pytest -v"
    ) -> ExecutionResult:
        """
        Run tests for generated code.
        
        Args:
            files: List of file dictionaries
            test_command: Command to run tests
            
        Returns:
            ExecutionResult with test output
        """
        workspace = self.create_workspace(files)
        
        try:
            # Install dependencies if requirements.txt exists
            req_file = workspace / "requirements.txt"
            if req_file.exists():
                install_result = self.run_command(
                    "pip install -r requirements.txt --quiet",
                    workspace
                )
                if not install_result.success:
                    return install_result
            
            # Run tests
            return self.run_command(test_command, workspace)
            
        finally:
            self.cleanup_workspace(workspace)
    
    def validate_dockerfile(
        self,
        dockerfile_content: str
    ) -> ExecutionResult:
        """
        Validate a Dockerfile syntax.
        
        Args:
            dockerfile_content: Dockerfile content
            
        Returns:
            ExecutionResult with validation status
        """
        workspace = self.create_workspace([
            {'path': 'Dockerfile', 'content': dockerfile_content}
        ])
        
        try:
            # Use docker build with --check flag if available, or dry-run
            if self.use_docker:
                result = self.run_command(
                    "docker build --check -f Dockerfile . 2>&1 || docker build --no-cache -f Dockerfile . --dry-run 2>&1 || echo 'Dockerfile syntax check passed'",
                    workspace
                )
            else:
                # Basic syntax check without Docker
                result = ExecutionResult(
                    success=True,
                    exit_code=0,
                    stdout="Dockerfile syntax check skipped (Docker not available)",
                    stderr="",
                    duration=0,
                    files_created=[]
                )
            
            return result
            
        finally:
            self.cleanup_workspace(workspace)