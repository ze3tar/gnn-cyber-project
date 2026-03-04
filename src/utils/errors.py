"""
Custom Exception Hierarchy for GNN Cyber Project
Provides structured error handling with context and recovery suggestions
"""

from typing import Any, Dict, Optional


class GNNCyberException(Exception):
    """Base exception for all GNN Cyber Project errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        remediation: Optional[str] = None,
    ):
        """
        Initialize exception with detailed context.

        Args:
            message: Error message
            details: Additional error context
            remediation: Suggested fix for the error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.remediation = remediation

    def __str__(self) -> str:
        """Format error message with details and remediation."""
        msg = self.message
        if self.details:
            msg += f"\nDetails: {self.details}"
        if self.remediation:
            msg += f"\nRemediation: {self.remediation}"
        return msg


class DataLoadError(GNNCyberException):
    """Raised when data loading fails."""

    pass


class DataValidationError(GNNCyberException):
    """Raised when data validation fails."""

    pass


class GraphConstructionError(GNNCyberException):
    """Raised when graph construction fails."""

    pass


class ModelError(GNNCyberException):
    """Raised when model operations fail."""

    pass


class TrainingError(GNNCyberException):
    """Raised when training fails."""

    pass


class InferenceError(GNNCyberException):
    """Raised when inference fails."""

    pass


class ConfigurationError(GNNCyberException):
    """Raised when configuration is invalid."""

    pass


class ResourceError(GNNCyberException):
    """Raised when resource allocation fails (GPU, memory, etc.)."""

    pass


class CheckpointError(GNNCyberException):
    """Raised when checkpoint loading/saving fails."""

    pass
