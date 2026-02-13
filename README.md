# SocialProphet

**Hybrid Time-Series Forecasting & Generative Content Agent**

A system that bridges the gap between social media engagement prediction and content generation by creating an integrated "Predict → Generate" pipeline.

## Overview

SocialProphet uses time-series forecasting models to analyze historical engagement data, then leverages these insights to generate context-aware content recommendations through a generative AI agent.

### Core Innovation

Unlike existing tools that either predict engagement OR generate content, SocialProphet creates an actionable bridge between analytics and content creation.

## Features

- **Time-Series Forecasting**: Ensemble approach using Prophet + SARIMA
- **Insight Extraction**: Automated pattern recognition and trend analysis
- **Content Generation**: LLM-powered content recommendations via Hugging Face
- **FIIT Framework**: Quality validation (Fluency, Interactivity, Information, Tone)

## Project Structure

```
SocialProphet/
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Cleaned data
│   └── predictions/      # Model outputs
├── notebooks/
│   ├── 01_EDA.ipynb      # Exploratory analysis
│   ├── 02_Forecasting.ipynb
│   ├── 03_Insights.ipynb
│   └── 04_Generation.ipynb
├── src/
│   ├── data_processing/  # Data collection & preprocessing
│   ├── forecasting/      # Time-series models
│   ├── insights/         # Pattern extraction
│   ├── generation/       # LLM integration
│   ├── evaluation/       # Metrics & visualization
│   └── utils/            # Configuration & helpers
├── tests/                # Unit tests
├── dashboard/            # Streamlit app
├── requirements.txt
└── setup.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/avkbsurya119/SocialProphet.git
cd SocialProphet
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

```python
from src.forecasting.prophet_model import ProphetForecaster
from src.generation.content_gen import ContentGenerator

# Initialize the pipeline
forecaster = ProphetForecaster()
generator = ContentGenerator()

# Train and predict
forecaster.train(historical_data)
predictions = forecaster.predict(days_ahead=7)

# Generate content recommendations
recommendations = generator.generate(predictions)
```

## Technology Stack

- **Forecasting**: Facebook Prophet, SARIMA (statsmodels)
- **Generation**: Hugging Face Inference API (Llama 3.1)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Dashboard**: Streamlit

## Evaluation Metrics

- **Forecasting**: RMSE, MAE, MAPE, R² Score
- **Content Quality**: FIIT Framework compliance
- **System**: End-to-end latency < 5 seconds

## Contributors

- [navadeep555](https://github.com/navadeep555)
- [avkbsurya119](https://github.com/avkbsurya119)

## License

MIT License

## Acknowledgments

- Facebook Prophet team
- Hugging Face community
- Academic references cited in the project report
