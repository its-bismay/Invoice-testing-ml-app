from pathlib import Path
import sqlite3

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = Path("../data/inventory.db")


FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]


def load_invoice_data(
    database_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """Load invoice data joined with aggregated purchase information."""

    database_path = Path(database_path)

    query = """
    WITH purchase_agg AS (
        SELECT
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(
                julianday(p.ReceivingDate) -
                julianday(p.PODate)
            ) AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )
    SELECT
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        (
            julianday(vi.InvoiceDate) -
            julianday(vi.PODate)
        ) AS days_po_to_invoice,
        (
            julianday(vi.PayDate) -
            julianday(vi.InvoiceDate)
        ) AS days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa
        ON vi.PONumber = pa.PONumber
    """

    if not database_path.exists():
        raise FileNotFoundError(f"Database not found: {database_path.resolve()}")

    with sqlite3.connect(database_path) as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        raise ValueError("No invoice records were found.")

    return df


def create_invoice_risk_label(
    df: pd.DataFrame,
) -> pd.Series:
    """Create the same rule-based target as the notebook."""

    price_mismatch = (df["invoice_dollars"] - df["total_item_dollars"]).abs() > 5

    delayed_receiving = df["avg_receiving_delay"] > 10

    return (price_mismatch | delayed_receiving).fillna(False).astype(int)


def prepare_data():
    """Return the feature matrix, target, and raw dataframe."""

    df = load_invoice_data()

    df["flag_invoice"] = create_invoice_risk_label(df)

    X = df[FEATURES].copy()
    y = df["flag_invoice"].copy()

    # Convert invalid numeric values to NaN.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([float("inf"), float("-inf")], float("nan"))

    # Remove rows where every selected feature is missing.
    valid_rows = X.notna().any(axis=1)

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)
    df = df.loc[valid_rows].reset_index(drop=True)

    if y.nunique() < 2:
        raise ValueError(
            "The target contains only one class. "
            "Check the invoice-risk labeling rules."
        )

    if y.value_counts().min() < 5:
        raise ValueError(
            "Each target class needs at least 5 samples "
            "for stratified splitting and 5-fold CV."
        )

    return X, y, df
