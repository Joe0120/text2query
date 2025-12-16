"""Chart.js utilities for validation and data population."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, List, Optional, Union

from .json_utils import extract_json_from_string, extract_list_from_string, convert_to_dict

logger = logging.getLogger(__name__)

# Valid Chart.js chart types
VALID_CHART_TYPES = ['line', 'bar', 'radar', 'pie', 'doughnut', 'polarArea', 'scatter', 'bubble']


def check_chartjs(
    chartjs_config: Dict[str, Any],
    data_labels_column: str,
    datasets_labels_column: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Validate Chart.js configuration.

    Args:
        chartjs_config: Chart.js configuration dictionary
        data_labels_column: Column name for X-axis labels
        datasets_labels_column: List of column names for datasets

    Returns:
        Validated configuration dict if valid, None if invalid
    """
    errors = []

    # Check basic structure
    if not isinstance(chartjs_config, dict):
        logger.error("Chart.js config must be a dictionary")
        return None

    # Check required fields
    required_fields = ["type", "data"]
    for field in required_fields:
        if field not in chartjs_config:
            errors.append((field, f"Missing required field '{field}'"))

    # Check type
    if "type" in chartjs_config and chartjs_config["type"] not in VALID_CHART_TYPES:
        errors.append(("type", f"Invalid chart type. Valid types: {', '.join(VALID_CHART_TYPES)}"))

    # Check data structure
    if "data" in chartjs_config:
        data = chartjs_config["data"]
        if not isinstance(data, dict):
            errors.append(("data", "data must be a dictionary"))
        else:
            # Check labels (required for most chart types)
            if "labels" not in data and chartjs_config.get("type") not in ["scatter", "bubble"]:
                errors.append(("data.labels", "Missing labels"))
            elif "labels" in data and not isinstance(data["labels"], list):
                errors.append(("data.labels", "labels must be a list"))

            # Check datasets
            if "datasets" not in data:
                errors.append(("data.datasets", "Missing datasets"))
            elif not isinstance(data["datasets"], list):
                errors.append(("data.datasets", "datasets must be a list"))
            else:
                for i, dataset in enumerate(data["datasets"]):
                    if not isinstance(dataset, dict):
                        errors.append((f"data.datasets[{i}]", "Each dataset must be a dictionary"))
                    elif "data" not in dataset:
                        errors.append((f"data.datasets[{i}].data", "Each dataset must contain a data field"))

    # Check options (optional)
    if "options" in chartjs_config and not isinstance(chartjs_config["options"], dict):
        errors.append(("options", "options must be a dictionary"))

    if errors:
        logger.warning("Chart.js validation failed:")
        for field, error in errors:
            logger.warning(f"  - {field}: {error}")
        return None

    logger.debug("Chart.js validation passed")
    return {
        "chartjs_config": convert_to_dict(chartjs_config),
        "data_labels_column": data_labels_column,
        "datasets_labels_column": datasets_labels_column
    }


def populate_chartjs_data(
    chartjs_config: Dict[str, Any],
    data_labels_column: str,
    datasets_labels_column: List[str],
    columns: List[str],
    result: List[List[Any]]
) -> Optional[Dict[str, Any]]:
    """
    Populate Chart.js configuration with actual query result data.

    Args:
        chartjs_config: Chart.js configuration template
        data_labels_column: Column name for X-axis labels
        datasets_labels_column: List of column names for datasets
        columns: Query result column names
        result: Query result data rows

    Returns:
        Populated Chart.js configuration, or None if column names don't match
    """
    temp_chartjs = copy.deepcopy(chartjs_config)

    # 1. Check if data_labels_column exists in columns
    if data_labels_column not in columns:
        logger.error(f"data_labels_column '{data_labels_column}' not in query result columns")
        logger.error(f"Available columns: {columns}")
        return None

    # 2. Check if all datasets_labels_column exist in columns
    missing_columns = [col for col in datasets_labels_column if col not in columns]
    if missing_columns:
        logger.error(f"datasets_labels_column {missing_columns} not in query result columns")
        logger.error(f"Available columns: {columns}")
        return None

    # 3. Get column indices
    label_col_idx = columns.index(data_labels_column)
    dataset_col_indices = [columns.index(col) for col in datasets_labels_column]

    logger.debug(f"Column mapping:")
    logger.debug(f"  X-axis label '{data_labels_column}' -> column {label_col_idx}")
    logger.debug(f"  Data columns {datasets_labels_column} -> columns {dataset_col_indices}")

    # 4. Extract X-axis labels
    temp_chartjs['data']['labels'] = [row[label_col_idx] for row in result]

    # 5. Extract dataset data
    if 'datasets' not in temp_chartjs['data']:
        temp_chartjs['data']['datasets'] = []

    # Create dataset for each column
    for i, col_name in enumerate(datasets_labels_column):
        col_idx = dataset_col_indices[i]

        # Create new dataset if needed
        if i >= len(temp_chartjs['data']['datasets']):
            temp_chartjs['data']['datasets'].append({
                'label': col_name,
                'data': [],
                'backgroundColor': f'rgba({54 + i * 50}, {162 + i * 30}, {235 - i * 20}, 0.7)',
                'borderColor': f'rgba({54 + i * 50}, {162 + i * 30}, {235 - i * 20}, 1)',
                'borderWidth': 1
            })

        # Set dataset label and data
        temp_chartjs['data']['datasets'][i]['label'] = col_name
        temp_chartjs['data']['datasets'][i]['data'] = [row[col_idx] for row in result]

    # 6. Remove extra datasets
    temp_chartjs['data']['datasets'] = temp_chartjs['data']['datasets'][:len(datasets_labels_column)]

    logger.debug(f"Successfully populated Chart.js data:")
    logger.debug(f"  Labels count: {len(temp_chartjs['data']['labels'])}")
    logger.debug(f"  Datasets count: {len(temp_chartjs['data']['datasets'])}")

    return temp_chartjs


def safe_parse_chartjs_config(chartjs_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Safely parse Chart.js configuration from LLM response.

    Args:
        chartjs_data: Raw chartjs data from LLM

    Returns:
        Parsed configuration dict, or None if parsing fails
    """
    try:
        if not chartjs_data or not isinstance(chartjs_data, dict):
            return None

        chartjs_config = chartjs_data.get('chartjs_config')
        if not chartjs_config:
            return None

        # Already a dict
        if isinstance(chartjs_config, dict):
            return chartjs_config

        # String - try JSON parsing
        if isinstance(chartjs_config, str):
            try:
                cleaned_config = chartjs_config.replace('\\n', '\n').replace('\\t', '\t')
                try:
                    return json.loads(cleaned_config)
                except json.JSONDecodeError:
                    return extract_json_from_string(cleaned_config)
            except Exception as e:
                logger.warning(f"Chart.js config JSON parsing failed: {e}")
                return None

        return None

    except Exception as e:
        logger.error(f"Chart.js config parsing error: {e}")
        return None


def safe_parse_data_labels_column(chartjs_data: Dict[str, Any]) -> Optional[str]:
    """
    Safely parse data labels column from LLM response.

    Args:
        chartjs_data: Raw chartjs data from LLM

    Returns:
        Data labels column name, or None if parsing fails
    """
    try:
        if not chartjs_data or not isinstance(chartjs_data, dict):
            return None

        data_labels_column = chartjs_data.get('data_labels_column')
        if not data_labels_column:
            return None

        if isinstance(data_labels_column, str):
            return data_labels_column.replace('"', '').replace("'", '').strip()

        return None

    except Exception as e:
        logger.error(f"Data labels column parsing error: {e}")
        return None


def safe_parse_datasets_labels_column(chartjs_data: Dict[str, Any]) -> List[str]:
    """
    Safely parse datasets labels columns from LLM response.

    Args:
        chartjs_data: Raw chartjs data from LLM

    Returns:
        List of dataset column names, or empty list if parsing fails
    """
    try:
        if not chartjs_data or not isinstance(chartjs_data, dict):
            return []

        datasets_labels_column = chartjs_data.get('datasets_labels_column')
        if not datasets_labels_column:
            return []

        # Already a list
        if isinstance(datasets_labels_column, list):
            return datasets_labels_column

        # String - try to parse as list
        if isinstance(datasets_labels_column, str):
            try:
                return extract_list_from_string(datasets_labels_column) or []
            except Exception as e:
                logger.warning(f"Datasets labels column JSON parsing failed: {e}")
                return []

        return []

    except Exception as e:
        logger.error(f"Datasets labels column parsing error: {e}")
        return []
