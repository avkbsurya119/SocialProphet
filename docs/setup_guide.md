# SocialProphet Setup Guide

Complete guide to set up the development environment for SocialProphet.

## Prerequisites

- Python 3.9 or higher
- Git
- pip (Python package manager)

## Step 1: Clone the Repository

```bash
git clone https://github.com/avkbsurya119/SocialProphet.git
cd SocialProphet
```

## Step 2: Create Virtual Environment

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/Mac
```bash
python -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional: Install GPU Support (for LSTM models)
```bash
pip install tensorflow  # or
pip install torch
```

## Step 4: Configure Environment Variables

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your API keys:
```
HF_TOKEN=your_huggingface_token_here
TWITTER_CONSUMER_KEY=your_twitter_key
TWITTER_CONSUMER_SECRET=your_twitter_secret
TWITTER_BEARER_TOKEN=your_bearer_token
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

## Step 5: Set Up API Keys

### Hugging Face Token
1. Create account at https://huggingface.co/
2. Go to Settings > Access Tokens
3. Create a new token with read permissions
4. Copy token to `.env` file

### Twitter/X API (Optional)
1. Apply for developer access at https://developer.twitter.com/
2. Create a project and app
3. Generate Bearer Token
4. Copy credentials to `.env` file

### Kaggle API (Optional)
1. Create account at https://www.kaggle.com/
2. Go to Account > Create New API Token
3. Move `kaggle.json` to `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)
4. Set appropriate permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Step 6: Download Dataset

### Option A: Use Sample Data
```bash
python scripts/download_data.py --source sample
```

### Option B: Download from Kaggle
```bash
python scripts/download_data.py --source kaggle
```

### Option C: Manual Download
1. Download from Kaggle:
   - https://www.kaggle.com/datasets/subashmaster0411/social-media-engagement-dataset
   - https://www.kaggle.com/datasets/kundanbedmutha/instagram-analytics-dataset
2. Place CSV files in `data/raw/`

## Step 7: Verify Installation

Run the verification script:
```python
python -c "
from src.utils.config import Config
from src.data_processing.collector import DataCollector

print('Config loaded:', Config.validate_api_keys())
print('Setup complete!')
"
```

## Step 8: Run EDA Notebook

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

## Project Structure

```
SocialProphet/
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Cleaned data
│   └── predictions/      # Model outputs
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
├── scripts/              # Utility scripts
├── tests/                # Unit tests
├── docs/                 # Documentation
└── dashboard/            # Streamlit app
```

## Troubleshooting

### Prophet Installation Issues

If Prophet fails to install:
```bash
# Windows
conda install -c conda-forge prophet

# Or use pystan
pip install pystan==2.19.1.1
pip install prophet
```

### CUDA/GPU Issues

For GPU support with TensorFlow:
```bash
pip install tensorflow-gpu
```

For GPU support with PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Missing Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Next Steps

1. Run EDA notebook: `notebooks/01_EDA.ipynb`
2. Train forecasting models: `notebooks/02_Forecasting.ipynb`
3. Set up LLM integration: `notebooks/03_Insights.ipynb`
4. Generate content: `notebooks/04_Generation.ipynb`

## Support

- GitHub Issues: https://github.com/avkbsurya119/SocialProphet/issues
- Contributors: navadeep555, avkbsurya119
