import pandas as pd
from sklearn.model_selection import train_test_split

from fuzzy_aqi.config import INPUT_COLUMNS, METADATA_COLUMNS, TARGET_COLUMN


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = INPUT_COLUMNS + [TARGET_COLUMN] + METADATA_COLUMNS
    return df[required_columns].copy()


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify_values = df[stratify_column] if stratify_column else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_values,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
