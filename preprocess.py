import pandas as pd


INPUT_PATH = "city_day.csv"
OUTPUT_PATH = "final.csv"
KEEP_COLUMNS = ["City", "Date", "PM2.5", "PM10", "NO2", "CO", "O3", "SO2", "AQI", "AQI_Bucket"]
INPUT_COLUMNS = ["PM2.5", "PM10", "NO2", "CO", "O3", "SO2"]
IMPUTE_COLUMNS = ["PM2.5", "NO2", "CO", "O3", "SO2"]
CAPS = {
    "PM2.5": 500,
    "PM10": 600,
    "NO2": 400,
    "CO": 50,
    "O3": 400,
    "SO2": 800,
    "AQI": 500,
}


def print_summary(step_name: str, before_rows: int, after_df: pd.DataFrame) -> None:
    print(f"\n{step_name}")
    print(f"rows before: {before_rows}")
    print(f"rows after : {len(after_df)}")
    print("nulls remaining:")
    print(after_df[KEEP_COLUMNS].isna().sum().to_string())


def main() -> None:
    df = pd.read_csv(INPUT_PATH)[KEEP_COLUMNS].copy()
    print_summary("Step 0 - Keep required columns", len(df), df)

    before_rows = len(df)
    df = df[df["AQI"].notna()].copy()
    print_summary("Step 1 - Drop rows with null AQI", before_rows, df)

    before_rows = len(df)
    df = df[~df[INPUT_COLUMNS].isna().all(axis=1)].copy()
    print_summary("Step 2 - Drop rows where all 6 inputs are null", before_rows, df)

    before_rows = len(df)
    df = df[df["PM10"].notna()].copy()
    print_summary("Step 3 - Drop rows where PM10 is null", before_rows, df)

    before_rows = len(df)
    city_medians = df.groupby("City")[IMPUTE_COLUMNS].transform("median")
    df[IMPUTE_COLUMNS] = df[IMPUTE_COLUMNS].fillna(city_medians)
    print_summary("Step 4 - City-wise median imputation for non-PM10 inputs", before_rows, df)

    before_rows = len(df)
    df = df.dropna().copy()
    print_summary("Step 5 - Drop rows still containing nulls", before_rows, df)

    before_rows = len(df)
    for column, upper_bound in CAPS.items():
        df[column] = df[column].clip(upper=upper_bound)
    print_summary("Step 6 - Cap outliers using CPCB bounds", before_rows, df)

    before_rows = len(df)
    df = df.drop_duplicates().copy()
    print_summary("Step 7 - Drop duplicate rows", before_rows, df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned dataset to {OUTPUT_PATH}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()
