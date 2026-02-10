"""
Process Supervision Reward Modeling for Code Generation.

This module implements "Process Supervision" for code generation, solving the sparse reward
problem in RLHF by treating Git Commits as MDP states. Instead of only rewarding the final
outcome, we provide dense rewards at each commit step based on syntactic validity, diff
application, incremental test passing, and final completion.
"""

import ast
from typing import List


class GitProcessReward:
    """
    Calculates dense rewards for a coding trajectory represented as a sequence of Git commits.
    
    This class implements Process Reward Modeling (PRM) which provides intermediate feedback
    during code generation, making the reinforcement learning problem more tractable by
    reducing the sparsity of rewards.
    
    The reward structure:
    - Syntactic Check: +0.1 if code is valid Python syntax, -0.1 if invalid
    - Diff Validity: +0.2 if commit applies cleanly to previous state
    - Incremental Test Passing: +0.5 if commit passes new tests compared to previous state
    - Final Completion: +1.0 if final state passes all tests
    """
    
    def __init__(self):
        """Initialize the reward calculator."""
        self.previous_state = None
        self.previous_test_results = set()
    
    def compute_step_rewards(self, commits: List[str]) -> List[float]:
        """
        Compute rewards for each commit in the trajectory.
        
        Args:
            commits: List of code strings representing sequential Git commits.
                    Each commit is the code state after applying that commit's diff.
        
        Returns:
            List of reward values (floats) for each commit step.
        """
        rewards = []
        self.previous_state = None
        self.previous_test_results = set()
        
        for i, commit_code in enumerate(commits):
            step_reward = 0.0
            
            # 1. Syntactic Check (Reward +0.1 or -0.1)
            if not self._check_syntax(commit_code):
                rewards.append(-0.1)
                continue
            step_reward += 0.1
            
            # 2. Diff Validity (Reward +0.2)
            if self._check_diff_application(commit_code):
                step_reward += 0.2
            
            # 3. Incremental Test Passing (Reward +0.5)
            current_test_results = self._run_tests(commit_code)
            new_tests_passed = current_test_results - self.previous_test_results
            if new_tests_passed:
                step_reward += 0.5
            
            # 4. Final Completion (Reward +1.0)
            is_final = (i == len(commits) - 1)
            if is_final and self._all_tests_pass(commit_code):
                step_reward += 1.0
            
            rewards.append(step_reward)
            
            # Update state for next iteration
            self.previous_state = commit_code
            self.previous_test_results = current_test_results
        
        return rewards
    
    def _check_syntax(self, code: str) -> bool:
        """
        Check if the code is valid Python syntax using ast.parse().
        
        Args:
            code: Python code string to validate.
        
        Returns:
            True if syntax is valid, False otherwise.
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _check_diff_application(self, new_code: str) -> bool:
        """
        Mock function that simulates checking if the commit applies cleanly to previous state.
        
        In a real implementation, this would:
        1. Apply the diff to the previous code state
        2. Check for merge conflicts or application errors
        3. Verify the resulting code structure is valid
        
        Args:
            new_code: The new code state after applying the commit.
        
        Returns:
            True if diff applies cleanly, False otherwise.
        """
        # Mock implementation: assume diff applies cleanly if code is non-empty
        # In practice, this would use a diff library or git apply simulation
        if self.previous_state is None:
            return True  # First commit always applies
        
        # Simulate: diff applies cleanly if new code is different from previous
        return new_code != self.previous_state
    
    def _run_tests(self, code: str) -> set:
        """
        Mock function that simulates running a subset of tests.
        
        In a real implementation, this would:
        1. Execute the code in a sandboxed environment
        2. Run unit tests or integration tests
        3. Return set of test identifiers that passed
        
        Args:
            code: Code state to test.
        
        Returns:
            Set of test identifiers (strings) that passed.
        """
        # Mock implementation: simulate test results based on code characteristics
        # In practice, this would execute actual tests
        passed_tests = set()
        
        # Simulate: tests pass if code contains certain patterns
        if 'def ' in code:
            passed_tests.add('test_function_definition')
        if 'class ' in code:
            passed_tests.add('test_class_definition')
        if 'import ' in code:
            passed_tests.add('test_imports')
        if 'return ' in code:
            passed_tests.add('test_return_statements')
        if len(code) > 100:
            passed_tests.add('test_code_length')
        
        return passed_tests
    
    def _all_tests_pass(self, code: str) -> bool:
        """
        Check if all tests pass for the final state.
        
        Args:
            code: Final code state to evaluate.
        
        Returns:
            True if all tests pass, False otherwise.
        """
        # Mock implementation: all tests pass if code has certain completeness indicators
        # In practice, this would run the full test suite
        test_results = self._run_tests(code)
        required_tests = {'test_function_definition', 'test_imports', 'test_return_statements'}
        return required_tests.issubset(test_results)
