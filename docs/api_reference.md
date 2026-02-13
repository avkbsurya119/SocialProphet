# SocialProphet API Reference

## Data Processing Module

### DataCollector

```python
from src.data_processing.collector import DataCollector

collector = DataCollector()

# Load CSV file
df = collector.load_csv('path/to/file.csv')

# Load Kaggle dataset
df = collector.load_kaggle_dataset('username/dataset-name')

# Collect Twitter data
df = collector.collect_twitter_data(query='#python', max_results=100)

# Save data
collector.save_data(df, 'filename.csv', data_type='processed')
```

### DataPreprocessor

```python
from src.data_processing.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()

# Clean data
df_clean = preprocessor.clean_data(df)

# Handle missing values
df = preprocessor.handle_missing_values(df, method='forward_fill')

# Remove outliers
df = preprocessor.remove_outliers(df, column='engagement', method='iqr')

# Temporal train-test split
train_df, test_df = preprocessor.temporal_train_test_split(df, test_size=0.2)

# Prepare for Prophet
prophet_df = preprocessor.prepare_prophet_data(df)
```

### FeatureEngineer

```python
from src.data_processing.features import FeatureEngineer

fe = FeatureEngineer()

# Add temporal features
df = fe.add_temporal_features(df)

# Add lag features
df = fe.add_lag_features(df, lags=[1, 7, 14, 30])

# Add rolling features
df = fe.add_rolling_features(df, windows=[7, 14, 30])

# Create all features
df = fe.create_all_features(df)
```

## Configuration

### Config

```python
from src.utils.config import Config

# Access paths
raw_path = Config.RAW_DATA_DIR
processed_path = Config.PROCESSED_DATA_DIR

# Access parameters
horizon = Config.FORECAST_HORIZON
prophet_params = Config.PROPHET_PARAMS

# Validate API keys
api_status = Config.validate_api_keys()

# Ensure directories exist
Config.ensure_directories()
```

## Utility Functions

### Helpers

```python
from src.utils.helpers import (
    save_json, load_json,
    save_pickle, load_pickle,
    get_timestamp_str,
    format_number,
    print_dataframe_info,
    timer
)

# Save/load JSON
save_json(data, 'output.json')
data = load_json('input.json')

# Timing decorator
@timer
def my_function():
    pass
```

## Coming Soon (Phase 2-4)

### Forecasting Module
- ProphetForecaster
- SARIMAForecaster
- EnsembleForecaster

### Insights Module
- InsightExtractor
- PromptBuilder

### Generation Module
- HuggingFaceClient
- ContentGenerator
- FIITValidator

### Evaluation Module
- ForecastMetrics
- Visualizer
