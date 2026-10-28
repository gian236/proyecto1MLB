"""Tests for the Kedro pipeline registry."""

from pathlib import Path

from kedro.framework.startup import bootstrap_project

from bike_sharing_forecast.pipeline_registry import register_pipelines

EXPECTED_PIPELINE_NODES = 2


def test_default_pipeline_is_registered():
    """The project must register a non-empty default pipeline."""
    bootstrap_project(Path.cwd())
    pipelines = register_pipelines()

    assert "__default__" in pipelines
    assert len(pipelines["__default__"].nodes) > 0
    assert "demand_forecasting" in pipelines
    assert len(pipelines["demand_forecasting"].nodes) == EXPECTED_PIPELINE_NODES
