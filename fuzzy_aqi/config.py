INPUT_COLUMNS = ["PM2.5", "PM10", "NO2", "CO", "O3", "SO2"]
TARGET_COLUMN = "AQI"
METADATA_COLUMNS = ["City", "Date", "AQI_Bucket"]
INPUT_FUZZY_LABELS = ["Low", "Medium", "High"]
OUTPUT_FUZZY_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
AQI_UNIVERSE_MIN = 0.0
AQI_UNIVERSE_MAX = 500.0
AQI_UNIVERSE_POINTS = 1001
