# src/schema_validation.py
from typing import Dict, List
import pandas as pd

class SchemaValidationError(Exception):
    pass


def validate_schema(
    df: pd.DataFrame,
    required_columns: Dict[str, List[str]],
) -> None:
    """
    Validate presence and type of required columns.

    required_columns format:
    {
        "numeric": ["price", "quantity"],
        "categorical": ["category", "region"],
        "datetime": ["order_date"]
    }
    """

    missing_cols = []
    for cols in required_columns.values():
        for col in cols:
            if col not in df.columns:
                missing_cols.append(col)

    if missing_cols:
        raise SchemaValidationError(
            f"Missing required columns: {missing_cols}"
        )

    # Type checks (soft, not coercive)
    for col in required_columns.get("numeric", []):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise SchemaValidationError(f"{col} must be numeric")

    for col in required_columns.get("datetime", []):
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise SchemaValidationError(f"{col} must be datetime")

    # Categoricals are flexible → no strict dtype check
