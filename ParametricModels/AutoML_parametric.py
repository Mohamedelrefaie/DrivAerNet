#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

The module provides a comprehensive workflow for training and evaluating machine learning models on the parametric (tabular) DrivAerNet++ dataset.
It includes functionalities for data loading, preprocessing, model training, evaluation, results saving, and visualization.
The models trained in this workflow include AutoGluon, XGBoost, LightGBM, RandomForest, and GradientBoosting.
The results are saved in a JSON file and visualized using matplotlib and seaborn.
"""
import pandas as pd
import numpy as np
import json
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
from typing import Callable, Dict, Tuple, List


class ModelTrainer:
    """
    Class to handle model training and evaluation.
    """

    def __init__(self, model_callable: Callable, eval_metric: Callable = r2_score):
        """
        Initialize the ModelTrainer with a model callable and evaluation metric.

        Args:
            model_callable (Callable): A callable that returns an untrained model instance.
            eval_metric (Callable): A callable for evaluating model performance, default is r2_score.
        """
        self.model_callable = model_callable
        self.eval_metric = eval_metric

    def train_evaluate(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                       y_test: pd.Series) -> float:
        """
        Train a regression model and evaluate its performance using R² score.

        Args:
            X_train (pd.DataFrame): Training data features.
            X_test (pd.DataFrame): Test data features.
            y_train (pd.Series): Training data labels.
            y_test (pd.Series): Test data labels.

        Returns:
            float: R² score of the model on the test data.
        """
        # Initialize and train the model
        model = self.model_callable()
        model.fit(X_train, y_train)

        # Predict and evaluate the model performance
        y_pred = model.predict(X_test)
        return self.eval_metric(y_test, y_pred)


class DataHandler:
    """
    Class to handle data loading, splitting, and preprocessing.
    """

    def __init__(self, file_path: str):
        """
        Initialize the DataHandler with the path to the data file.

        Args:
            file_path (str): Path to the CSV file containing the dataset.
        """
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

        # Extract design category from 'Experiment' column
        self.data['Design_Category'] = self.data['Experiment'].apply(lambda x: x.split('_')[0])

    def get_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Get datasets split by design category.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of datasets.
        """
        datasets = {
            'Fastback_F': self.data[self.data['Design_Category'] == 'F'],
            'Combined_All': self.data
        }
        return datasets


class ResultSaver:
    """
    Class to handle saving and loading results.
    """

    @staticmethod
    def save_results(results: Dict[str, Dict[str, Dict[str, float]]], file_path: str) -> None:
        """
        Save results to a JSON file.

        Args:
            results (Dict[str, Dict[str, Dict[str, float]]]): Results to save.
            file_path (str): Path to save the results file.
        """
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {file_path}")

    @staticmethod
    def load_results(file_path: str) -> Dict:
        """
        Load results from a JSON file.

        Args:
            file_path (str): Path to the results file.

        Returns:
            Dict: Loaded results.
        """
        with open(file_path, 'r') as f:
            return json.load(f)


class Plotter:
    """
    Class to handle plotting of results.
    """

    def __init__(self, results: Dict[str, Dict[str, Dict[str, float]]]):
        """
        Initialize the Plotter with the results data.

        Args:
            results (Dict[str, Dict[str, Dict[str, float]]]): Results to plot.
        """
        self.results = results

    def plot_results(self) -> None:
        """
        Plot the results.
        """
        dataset_sizes = [0.2, 0.4, 0.6, 0.8, 0.95]
        sizes = [int(size * 100) for size in dataset_sizes]
        sizes[-1] = 100  # Replace 95 with 100 for x-axis labels

        # Mapping of dataset names for better readability
        category_names = {
            'Fastback_F': 'Fastback Only',
            'Combined_All': 'Fastback + Notchback + Estateback'
        }

        # Define plot settings
        mpl_settings = {
            "axes.labelsize": "large",
            "axes.titlesize": "xx-large",
            "xtick.labelsize": "large",
            "ytick.labelsize": "large",
        }

        with plt.rc_context(mpl_settings):
            fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)
            sns.set(style="white")

            # Loop through each dataset to plot the results
            for idx, (dataset_name, dataset_results) in enumerate(self.results.items()):
                ax = axes[idx]
                model_final_values = {}
                model_lines = {}

                # Loop through each model's results
                for model_name, results_summary in dataset_results.items():
                    print(f"Plotting results for {model_name} in {dataset_name}: {results_summary}")  # Debug print
                    mean_r2 = [results_summary[str(size)]['mean_r2'] for size in dataset_sizes]
                    ci_r2 = [results_summary[str(size)]['ci'] for size in dataset_sizes]
                    line, = ax.plot(sizes, mean_r2, marker='o', linewidth=2, label=model_name)
                    ax.fill_between(sizes, np.array(mean_r2) - np.array(ci_r2), np.array(mean_r2) + np.array(ci_r2),
                                    alpha=0.2)

                    model_final_values[model_name] = mean_r2[-1]
                    model_lines[model_name] = line.get_color()

                # Set plot labels and titles
                ax.set_xlabel('Percentage of Training Data (%)')
                ax.set_title(category_names[dataset_name], fontsize=22)
                ax.grid(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(True)
                ax.spines['left'].set_color('black')
                ax.spines['left'].set_linewidth(1)
                ax.spines['bottom'].set_visible(True)
                ax.spines['bottom'].set_color('black')
                ax.spines['bottom'].set_linewidth(1)
                ax.tick_params(axis='x', which='both', length=0)
                ax.tick_params(axis='y', which='both', length=0)

                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')

                # Set y-axis limits based on the range of R² values
                all_r2_values = [v['mean_r2'] for k, v in results_summary.items()]
                y_min = min(all_r2_values) - 0.15
                y_max = max(all_r2_values) + 0.1
                ax.set_ylim(y_min, y_max)

                y_range = y_max - y_min
                sep = y_range / 10

                final_values = pd.Series(model_final_values).sort_values(ascending=False)
                theoretical_values = pd.Series(
                    0.5 * (y_max + final_values.iloc[0]) - np.arange(len(final_values)) * sep,
                    index=final_values.index,
                    name="theoretical"
                )

                merged_values = pd.concat([final_values, theoretical_values], axis=1)

                # Add labels to the lines for each model
                for j, (k, v) in enumerate(merged_values.iterrows()):
                    ax.plot([sizes[-1], sizes[-1] + 5], [v[0], v["theoretical"]], linestyle=":", color=model_lines[k],
                            linewidth=1.5)
                    ax.text(sizes[-1] + 6, v["theoretical"], k, ha="left", va="center", size=16, color=model_lines[k])

            axes[0].set_ylabel('R²')

            plt.tight_layout()
            plt.savefig('../results_plot.png', dpi=300,
                        bbox_inches='tight')
            plt.show()


def main(file_path: str, output_path: str) -> None:
    """
    Main function to run the entire workflow.

    Args:
        file_path (str): Path to the input data file.
        output_path (str): Path to save the results.
    """
    # Initialize handlers
    data_handler = DataHandler(file_path)
    datasets = data_handler.get_datasets()

    models: Dict[str, Callable] = {
        'AutoGluon': lambda: TabularPredictor(label='Average_Cd', problem_type='regression', eval_metric='r2'),
        'XGBoost': lambda: xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        'LightGBM': lambda: lgb.LGBMRegressor(objective='regression', random_state=42),
        'RandomForest': lambda: RandomForestRegressor(random_state=42),
        'GradientBoosting': lambda: GradientBoostingRegressor(random_state=42)
    }

    dataset_sizes: List[float] = [0.2, 0.4, 0.6, 0.8, 0.95]
    n_splits: int = 20

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    # Loop through each dataset
    for dataset_name, dataset in datasets.items():
        print(f"\nProcessing dataset: {dataset_name}")

        # Split the dataset into training and test sets
        X = dataset.drop(columns=['Experiment', 'Average Cd', 'Std Cd', 'Design_Category'])
        y = dataset['Average Cd']
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        dataset_results: Dict[str, Dict[str, float]] = {}

        # Loop through each model
        for model_name, model_fn in models.items():
            results: Dict[float, List[float]] = {size: [] for size in dataset_sizes}

            # Loop through each dataset size
            for size in dataset_sizes:
                print(f"\nProcessing dataset size: {int(size * 100)}% for {model_name}")

                # Perform multiple splits for robust evaluation
                for split in range(n_splits):
                    if size < 1.0:
                        X_train, _, y_train, _ = train_test_split(X_train_full, y_train_full, train_size=size,
                                                                  random_state=split)
                    else:
                        X_train, _, y_train, _ = train_test_split(X_train_full, y_train_full, train_size=0.95,
                                                                  random_state=split)

                    trainer = ModelTrainer(model_fn)
                    if model_name == 'AutoGluon':
                        # Train and evaluate using AutoGluon
                        train_data = TabularDataset(pd.DataFrame(X_train).assign(Average_Cd=y_train))
                        test_data = TabularDataset(pd.DataFrame(X_test).assign(Average_Cd=y_test))
                        predictor = trainer.model_callable().fit(train_data)
                        y_pred = predictor.predict(test_data)
                        r2 = r2_score(y_test, y_pred)
                    else:
                        # Train and evaluate using other models
                        r2 = trainer.train_evaluate(X_train, X_test, y_train, y_test)

                    # Store the R² score
                    results[size].append(r2)
                    print(f"Split {split + 1}/{n_splits}, R²: {r2:.4f}")

            # Calculate summary statistics for the results
            results_summary: Dict[str, Dict[str, float]] = {
                str(size): {
                    'mean_r2': np.mean(results[size]),
                    'std_r2': np.std(results[size]),
                    'ci': t.ppf(0.975, n_splits - 1) * np.std(results[size]) / np.sqrt(n_splits)
                }
                for size in dataset_sizes
            }

            dataset_results[model_name] = results_summary

        all_results[dataset_name] = dataset_results

    # Save results
    ResultSaver.save_results(all_results, output_path)

    # Plot results
    plotter = Plotter(all_results)
    plotter.plot_results()
# Example usage
file_path = '../ParametricData/DrivAerNet_ParametricData.csv'
output_path = 'model_results_F_ALl.json'
main(file_path, output_path)
