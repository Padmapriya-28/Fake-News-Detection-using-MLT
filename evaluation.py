"""
Evaluation and Visualization Module
Generates evaluation metrics and visualizations for fake news detection models
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from config import ASSETS_DIR


# Set style for matplotlib
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class ModelEvaluator:
    """
    Class for evaluating and visualizing model performance
    """
    
    def __init__(self, results):
        """
        Initialize evaluator with model results
        
        Args:
            results (dict): Dictionary containing results from all models
        """
        self.results = results
        
    def plot_confusion_matrix(self, model_name, save_path=None):
        """
        Plot confusion matrix for a specific model
        
        Args:
            model_name (str): Name of the model
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The confusion matrix figure
        """
        if model_name not in self.results:
            raise ValueError(f"No results found for {model_name}")
        
        cm = self.results[model_name]['confusion_matrix']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'],
                    yticklabels=['Real', 'Fake'],
                    cbar_kws={'label': 'Count'},
                    ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_all_confusion_matrices(self, save_dir=None):
        """
        Plot confusion matrices for all models
        
        Args:
            save_dir (str): Directory to save plots
            
        Returns:
            dict: Dictionary of figures
        """
        figures = {}
        
        for model_name in self.results.keys():
            if save_dir:
                save_path = os.path.join(save_dir, f'cm_{model_name.lower().replace(" ", "_")}.png')
            else:
                save_path = None
            
            fig = self.plot_confusion_matrix(model_name, save_path)
            figures[model_name] = fig
            plt.close(fig)
        
        return figures
    
    def plot_metrics_comparison(self, save_path=None):
        """
        Plot comparison of all metrics across models
        
        Args:
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The comparison figure
        """
        # Prepare data
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        data = {metric: [] for metric in metrics}
        
        for model_name in models:
            for metric in metrics:
                data[metric].append(self.results[model_name][metric])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(models))
        width = 0.2
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, data[metric], width, label=label, color=colors[i], alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Metrics comparison saved to {save_path}")
        
        return fig
    
    def create_interactive_comparison(self):
        """
        Create interactive comparison chart using Plotly
        
        Returns:
            plotly.graph_objects.Figure: Interactive figure
        """
        # Prepare data
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metric_labels,
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            values = [self.results[model][metric] for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=label,
                    marker_color=[colors[i % len(colors)] for i in range(len(models))],
                    text=[f'{v:.3f}' for v in values],
                    textposition='outside',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            fig.update_yaxes(range=[0, 1.1], row=row, col=col)
        
        fig.update_layout(
            title_text="Model Performance Metrics Comparison",
            title_font_size=20,
            height=700,
            showlegend=False
        )
        
        return fig
    
    def plot_model_comparison_radar(self, save_path=None):
        """
        Create radar chart comparing all models
        
        Args:
            save_path (str): Path to save the plot
            
        Returns:
            plotly.graph_objects.Figure: Radar chart figure
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig = go.Figure()
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for idx, model_name in enumerate(self.results.keys()):
            values = [
                self.results[model_name]['accuracy'],
                self.results[model_name]['precision'],
                self.results[model_name]['recall'],
                self.results[model_name]['f1_score']
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=model_name,
                line_color=colors[idx % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Model Comparison - Radar Chart",
            title_font_size=18
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✓ Radar chart saved to {save_path}")
        
        return fig
    
    def generate_classification_report(self, model_name, y_test):
        """
        Generate detailed classification report
        
        Args:
            model_name (str): Name of the model
            y_test: True labels
            
        Returns:
            str: Classification report
        """
        if model_name not in self.results:
            raise ValueError(f"No results found for {model_name}")
        
        y_pred = self.results[model_name]['predictions']
        
        report = classification_report(
            y_test, y_pred,
            target_names=['Real', 'Fake'],
            digits=4
        )
        
        return report
    
    def create_summary_table(self):
        """
        Create a summary table of all model results
        
        Returns:
            pandas.DataFrame: Summary table
        """
        data = []
        
        for model_name, metrics in self.results.items():
            data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']*100:.2f}%",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
        
        df = pd.DataFrame(data)
        return df
    
    def plot_accuracy_bars(self, save_path=None):
        """
        Create a simple bar chart of model accuracies
        
        Args:
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Bar chart figure
        """
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] * 100 for model in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax.barh(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(acc + 0.5, i, f'{acc:.2f}%', va='center', fontweight='bold', fontsize=11)
        
        ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim([0, 105])
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Accuracy bars saved to {save_path}")
        
        return fig


def generate_all_visualizations(results, output_dir=None):
    """
    Generate all visualizations and save them
    
    Args:
        results (dict): Model results
        output_dir (str): Directory to save visualizations
        
    Returns:
        dict: Dictionary of all generated figures
    """
    if output_dir is None:
        output_dir = ASSETS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = ModelEvaluator(results)
    
    figures = {}
    
    # 1. Confusion matrices
    print("\nGenerating confusion matrices...")
    cm_figs = evaluator.plot_all_confusion_matrices(output_dir)
    figures['confusion_matrices'] = cm_figs
    
    # 2. Metrics comparison
    print("Generating metrics comparison...")
    metrics_path = os.path.join(output_dir, 'metrics_comparison.png')
    metrics_fig = evaluator.plot_metrics_comparison(metrics_path)
    figures['metrics_comparison'] = metrics_fig
    plt.close(metrics_fig)
    
    # 3. Accuracy bars
    print("Generating accuracy comparison...")
    accuracy_path = os.path.join(output_dir, 'accuracy_comparison.png')
    accuracy_fig = evaluator.plot_accuracy_bars(accuracy_path)
    figures['accuracy_comparison'] = accuracy_fig
    plt.close(accuracy_fig)
    
    # 4. Interactive radar chart
    print("Generating radar chart...")
    radar_path = os.path.join(output_dir, 'radar_comparison.html')
    radar_fig = evaluator.plot_model_comparison_radar(radar_path)
    figures['radar_chart'] = radar_fig
    
    # 5. Summary table
    print("Generating summary table...")
    summary_df = evaluator.create_summary_table()
    summary_path = os.path.join(output_dir, 'results_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary table saved to {summary_path}")
    
    print("\n✓ All visualizations generated successfully!")
    
    return figures


if __name__ == "__main__":
    print("Evaluation Module")
    print("This module should be imported and used with model results")
