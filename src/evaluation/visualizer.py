"""
Forecast Visualization Module for SocialProphet.

Creates plots for model evaluation and comparison.
All visualizations work with original-scale engagement values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class Visualizer:
    """Visualization utilities for forecasting results."""

    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize visualizer with style settings.

        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('seaborn-whitegrid')

        self.figsize = (12, 6)
        self.colors = {
            'actual': '#1f77b4',      # Blue
            'prophet': '#ff7f0e',     # Orange
            'sarima': '#2ca02c',      # Green
            'lstm': '#d62728',        # Red
            'ensemble': '#9467bd',    # Purple
        }

    def plot_predictions(
        self,
        dates: pd.DatetimeIndex,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        title: str = "Forecast vs Actual",
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Plot actual vs predicted values.

        Args:
            dates: Date index
            y_true: Actual values (original scale)
            predictions: Dict of model_name -> predictions (original scale)
            title: Plot title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        if figsize is None:
            figsize = self.figsize

        fig, ax = plt.subplots(figsize=figsize)

        # Plot actual values
        ax.plot(
            dates[:len(y_true)], y_true,
            label='Actual',
            color=self.colors['actual'],
            linewidth=2
        )

        # Plot predictions
        for name, preds in predictions.items():
            color = self.colors.get(name, '#808080')
            ax.plot(
                dates[:len(preds)], preds,
                label=name.capitalize(),
                color=color,
                linewidth=1.5,
                alpha=0.8
            )

        ax.set_xlabel('Date')
        ax.set_ylabel('Engagement (Original Scale)')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        return fig

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create residual diagnostic plots.

        Includes:
        - Residuals over time
        - Residual histogram
        - Q-Q plot
        - Residuals vs fitted

        Args:
            y_true: Actual values (original scale)
            y_pred: Predicted values (original scale)
            model_name: Name for plot titles
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        if figsize is None:
            figsize = (12, 10)

        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Residuals over time
        ax1 = axes[0, 0]
        ax1.plot(residuals, color=self.colors['actual'])
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel('Observation')
        ax1.set_ylabel('Residual')
        ax1.set_title(f'{model_name} Residuals Over Time')
        ax1.grid(True, alpha=0.3)

        # 2. Residual histogram
        ax2 = axes[0, 1]
        ax2.hist(residuals, bins=20, color=self.colors['actual'], edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel('Residual')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residual Distribution')
        ax2.grid(True, alpha=0.3)

        # 3. Q-Q plot
        ax3 = axes[1, 0]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normality Check)')
        ax3.grid(True, alpha=0.3)

        # 4. Residuals vs Fitted
        ax4 = axes[1, 1]
        ax4.scatter(y_pred, residuals, alpha=0.5, color=self.colors['actual'])
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax4.set_xlabel('Fitted Values')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals vs Fitted')
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'{model_name} Diagnostic Plots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_model_comparison(
        self,
        metrics_df: pd.DataFrame,
        metric_names: List[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Bar chart comparing model performance.

        Args:
            metrics_df: DataFrame with model metrics
            metric_names: Metrics to plot
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        if metric_names is None:
            metric_names = ['mape', 'rmse_pct', 'r2']

        if figsize is None:
            figsize = (12, 4)

        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

        if n_metrics == 1:
            axes = [axes]

        models = metrics_df['model'].tolist()
        x = np.arange(len(models))
        width = 0.6

        for idx, metric in enumerate(metric_names):
            ax = axes[idx]
            values = metrics_df[metric].values

            # Color bars by performance
            colors = []
            for i, model in enumerate(models):
                colors.append(self.colors.get(model.lower(), '#808080'))

            bars = ax.bar(x, values, width, color=colors, alpha=0.8)

            # Add threshold line
            if metric == 'mape':
                ax.axhline(y=15, color='red', linestyle='--', label='Threshold (15%)')
                ax.set_ylabel('MAPE (%)')
            elif metric == 'rmse_pct':
                ax.axhline(y=15, color='red', linestyle='--', label='Threshold (15%)')
                ax.set_ylabel('RMSE (% of mean)')
            elif metric == 'r2':
                ax.axhline(y=0.7, color='red', linestyle='--', label='Threshold (0.7)')
                ax.set_ylabel('R² Score')

            ax.set_xlabel('Model')
            ax.set_title(metric.upper())
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper right', fontsize=8)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        return fig

    def plot_forecast_horizon(
        self,
        train_dates: pd.DatetimeIndex,
        train_values: np.ndarray,
        test_dates: pd.DatetimeIndex,
        test_values: np.ndarray,
        predictions: np.ndarray,
        confidence_lower: Optional[np.ndarray] = None,
        confidence_upper: Optional[np.ndarray] = None,
        model_name: str = "Model",
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Plot forecast with confidence intervals.

        Shows train/test split boundary.

        Args:
            train_dates: Training period dates
            train_values: Training values
            test_dates: Test period dates
            test_values: Actual test values
            predictions: Predicted values for test period
            confidence_lower: Lower CI bound
            confidence_upper: Upper CI bound
            model_name: Name for title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        if figsize is None:
            figsize = (14, 6)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot training data
        ax.plot(
            train_dates[-60:], train_values[-60:],  # Last 60 days of training
            label='Training Data',
            color=self.colors['actual'],
            linewidth=1.5
        )

        # Plot test actual
        ax.plot(
            test_dates[:len(test_values)], test_values,
            label='Actual (Test)',
            color=self.colors['actual'],
            linewidth=2,
            linestyle='--'
        )

        # Plot predictions
        ax.plot(
            test_dates[:len(predictions)], predictions,
            label=f'{model_name} Forecast',
            color=self.colors['ensemble'],
            linewidth=2
        )

        # Plot confidence interval
        if confidence_lower is not None and confidence_upper is not None:
            ax.fill_between(
                test_dates[:len(confidence_lower)],
                confidence_lower,
                confidence_upper,
                color=self.colors['ensemble'],
                alpha=0.2,
                label='95% Confidence Interval'
            )

        # Add vertical line at train/test split
        split_date = train_dates[-1]
        ax.axvline(x=split_date, color='gray', linestyle=':', linewidth=2, label='Train/Test Split')

        ax.set_xlabel('Date')
        ax.set_ylabel('Engagement (Original Scale)')
        ax.set_title(f'{model_name} Forecast with Confidence Interval')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        return fig

    def plot_training_history(
        self,
        history: Dict,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Plot LSTM training loss over epochs.

        Args:
            history: Training history dict with 'loss' and 'val_loss'
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        if figsize is None:
            figsize = (10, 5)

        fig, ax = plt.subplots(figsize=figsize)

        # Handle both Keras History object and plain dict
        if hasattr(history, 'history'):
            history = history.history

        epochs = range(1, len(history['loss']) + 1)

        ax.plot(epochs, history['loss'], label='Training Loss', color=self.colors['actual'])
        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'], label='Validation Loss', color=self.colors['ensemble'])

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('LSTM Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Mark best epoch
        if 'val_loss' in history:
            best_epoch = np.argmin(history['val_loss']) + 1
            best_loss = min(history['val_loss'])
            ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
            ax.annotate(f'Best: {best_loss:.4f}',
                       xy=(best_epoch, best_loss),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=9)

        plt.tight_layout()
        return fig

    def plot_ensemble_weights(
        self,
        weights: Dict[str, float],
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Pie chart of ensemble model weights.

        Args:
            weights: Dict of model_name -> weight
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        if figsize is None:
            figsize = (8, 8)

        fig, ax = plt.subplots(figsize=figsize)

        labels = [name.capitalize() for name in weights.keys()]
        sizes = list(weights.values())
        colors = [self.colors.get(name.lower(), '#808080') for name in weights.keys()]

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=[0.02] * len(sizes)
        )

        ax.set_title('Ensemble Model Weights', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_actual_vs_predicted_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Scatter plot of actual vs predicted values.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name for title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        if figsize is None:
            figsize = (8, 8)

        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(y_true, y_pred, alpha=0.5, color=self.colors['actual'])

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{model_name}: Actual vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add R² annotation
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax.annotate(f'R² = {r2:.4f}',
                   xy=(0.05, 0.95),
                   xycoords='axes fraction',
                   fontsize=12,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig

    def create_dashboard(
        self,
        results: Dict,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create comprehensive multi-panel dashboard.

        Args:
            results: Evaluation results dict
            train_df: Training data
            test_df: Test data
            predictions_df: Predictions DataFrame
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(16, 12))

        # Layout: 2x2 grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)

        # 1. Forecast vs Actual (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        test_dates = pd.to_datetime(test_df['ds'])
        y_actual = test_df['y_raw'].values if 'y_raw' in test_df.columns else np.expm1(test_df['y'].values)
        y_ensemble = predictions_df['ensemble_pred_original'].values

        ax1.plot(test_dates[:len(y_actual)], y_actual, label='Actual', color=self.colors['actual'], linewidth=2)
        ax1.plot(test_dates[:len(y_ensemble)], y_ensemble, label='Ensemble', color=self.colors['ensemble'], linewidth=2)
        ax1.set_title('Forecast vs Actual')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Engagement')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 2. Model Comparison (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        models = list(results.get('models', {}).keys()) + ['ensemble']
        mape_values = [results['models'][m]['mape'] for m in results.get('models', {})] + [results['ensemble']['mape']]

        colors = [self.colors.get(m, '#808080') for m in models]
        bars = ax2.bar(models, mape_values, color=colors, alpha=0.8)
        ax2.axhline(y=15, color='red', linestyle='--', label='Threshold (15%)')
        ax2.set_title('Model MAPE Comparison')
        ax2.set_ylabel('MAPE (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Residuals (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        residuals = y_actual[:len(y_ensemble)] - y_ensemble
        ax3.plot(residuals, color=self.colors['actual'])
        ax3.axhline(y=0, color='red', linestyle='--')
        ax3.set_title('Ensemble Residuals')
        ax3.set_xlabel('Observation')
        ax3.set_ylabel('Residual')
        ax3.grid(True, alpha=0.3)

        # 4. Metrics Summary (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        metrics_text = []
        metrics_text.append("EVALUATION SUMMARY")
        metrics_text.append("=" * 30)
        metrics_text.append(f"\nEnsemble Metrics:")
        for metric, value in results['ensemble'].items():
            metrics_text.append(f"  {metric.upper()}: {value:.4f}")

        metrics_text.append(f"\nThreshold Check:")
        metrics_text.append(f"  MAPE < 15%: {'PASS' if results['ensemble']['mape'] < 15 else 'FAIL'}")
        metrics_text.append(f"  R² > 0.70: {'PASS' if results['ensemble']['r2'] > 0.70 else 'FAIL'}")

        ax4.text(0.1, 0.9, '\n'.join(metrics_text), transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('SocialProphet Forecasting Dashboard', fontsize=16, fontweight='bold')

        if save_path:
            self.save_figure(fig, save_path)

        return fig

    def save_figure(
        self,
        fig: plt.Figure,
        filepath: Path,
        dpi: int = 150
    ) -> None:
        """
        Save figure to file.

        Args:
            fig: matplotlib Figure
            filepath: Output path
            dpi: Resolution
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved: {filepath}")


def run_visualization():
    """Create visualizations from saved results."""
    print("\n" + "#" * 60)
    print("# SocialProphet - Visualization")
    print("#" * 60)

    # Load data
    test_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "test_data.csv")
    train_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "train_data.csv")

    # Check for ensemble results
    results_path = Config.PROCESSED_DATA_DIR / "ensemble_results.json"
    if not results_path.exists():
        print("No ensemble results found. Run ensemble forecasting first.")
        return

    import json
    with open(results_path) as f:
        results = json.load(f)

    print(f"\nLoaded results with {len(results.get('models', {}))} models")

    viz = Visualizer()

    # Create output directory
    output_dir = Config.PREDICTIONS_DIR / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot ensemble weights
    fig = viz.plot_ensemble_weights(results['weights'])
    viz.save_figure(fig, output_dir / "ensemble_weights.png")

    print("\n" + "#" * 60)
    print("# Visualization Complete!")
    print("#" * 60)


if __name__ == "__main__":
    run_visualization()
