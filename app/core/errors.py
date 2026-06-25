class SignalLensError(Exception):
    """Base exception for expected application-layer failures."""


class UnsupportedDocumentTypeError(SignalLensError):
    """Raised when no parser supports the uploaded document type."""


class DependencyUnavailableError(SignalLensError):
    """Raised when an optional production dependency is not installed."""


class IngestionValidationError(SignalLensError):
    """Raised when ingestion input is structurally invalid."""


class RetrievalValidationError(SignalLensError):
    """Raised when retrieval or retrieval-evaluation input is invalid."""
