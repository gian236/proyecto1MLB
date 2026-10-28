"""Demand forecasting pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from bike_sharing_forecast.pipelines.demand_forecasting.nodes import (
    build_features,
    train_and_evaluate,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the hourly demand forecasting pipeline."""
    return pipeline(
        [
            node(
                func=build_features,
                inputs="train_raw",
                outputs="model_features",
                name="build_demand_features",
            ),
            node(
                func=train_and_evaluate,
                inputs={
                    "features": "model_features",
                    "training_params": "params:training",
                },
                outputs=[
                    "evaluation_metrics",
                    "validation_predictions",
                    "trained_model",
                ],
                name="train_and_evaluate_catboost",
            ),
        ]
    )
