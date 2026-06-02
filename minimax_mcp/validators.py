"""Parameter validation for Minimax MCP tool functions."""
from minimax_mcp.exceptions import MinimaxValidationError


def _validate_range(name: str, value, min_val, max_val):
    if not (min_val <= value <= max_val):
        raise MinimaxValidationError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )


def _validate_enum(name: str, value, valid_values: set):
    if value not in valid_values:
        raise MinimaxValidationError(
            f"{name} must be one of {sorted(valid_values)}, got {value!r}"
        )
