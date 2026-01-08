"""
Chart.js configuration generator using LLM.

This module provides functionality to generate Chart.js configurations
based on query results and natural language questions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..utils.json_utils import extract_json_from_string
from ..utils.chart_js import (
    check_chartjs,
    populate_chartjs_data,
    safe_parse_chartjs_config,
    safe_parse_data_labels_column,
    safe_parse_datasets_labels_column,
    VALID_CHART_TYPES,
)


CHART_GENERATION_PROMPT = """You are a data visualization expert. Based on the user's query, SQL command, and query results,
generate an appropriate Chart.js configuration.

User Query: {question}

SQL Command: {sql_command}

Query Result Columns: {columns}

Sample Data (first {sample_size} rows):
{sample_data}

Instructions:
1. Analyze the data and determine the most appropriate chart type
2. Only use these chart types: {valid_types}
3. Choose columns that make sense for visualization:
   - data_labels_column: The column to use for X-axis labels (typically categories, dates, names)
   - datasets_labels_column: The columns to use for Y-axis data (typically numeric values)
4. The column names MUST exactly match the columns from the query result
5. Generate a valid Chart.js configuration

Response format (JSON only, no markdown):
{{
    "chartjs_config": {{
        "type": "<chart_type>",
        "data": {{
            "labels": [],
            "datasets": [
                {{
                    "label": "<dataset_label>",
                    "data": [],
                    "backgroundColor": "rgba(54, 162, 235, 0.7)",
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "borderWidth": 1
                }}
            ]
        }},
        "options": {{
            "responsive": true,
            "plugins": {{
                "title": {{
                    "display": true,
                    "text": "<chart_title>"
                }}
            }}
        }}
    }},
    "data_labels_column": "<column_name_for_x_axis>",
    "datasets_labels_column": ["<column_name_for_data_1>", "<column_name_for_data_2>"]
}}

Important:
- datasets_labels_column MUST be a list of strings, even if there's only one column
- Column names must exactly match the query result columns
- Do not include actual data in the config - only the structure
- If the data is not suitable for visualization, return null for chartjs_config
"""


class ChartGenerator:
    """
    Generate Chart.js configurations from query results using LiteLLM.

    This class analyzes query results and generates appropriate Chart.js
    configurations for data visualization.

    Example:
        >>> from text2query.core.utils.model_configs import create_llm_config
        >>> from text2query.llm.chart_generator import ChartGenerator
        >>>
        >>> llm_config = create_llm_config(
        ...     model_name="gpt-4o-mini",
        ...     apikey="your-api-key",
        ...     provider="openai"
        ... )
        >>> generator = ChartGenerator(llm_config=llm_config)
        >>>
        >>> result = await generator.generate_chart(
        ...     question="Show monthly sales",
        ...     sql_command="SELECT month, total FROM sales",
        ...     columns=["month", "total"],
        ...     rows=[["Jan", 100], ["Feb", 150], ["Mar", 200]]
        ... )
    """

    def __init__(
        self,
        llm_config: Any,
        sample_size: int = 5,
    ):
        """
        Initialize ChartGenerator.

        Args:
            llm_config: ModelConfig instance for LiteLLM.
                       Create via create_llm_config(model_name, apikey, provider)
            sample_size: Number of sample rows to show LLM (default: 5)
        """
        if llm_config is None:
            raise ValueError("llm_config is required. Create via create_llm_config(model_name, apikey, provider)")

        self.llm_config = llm_config
        self.sample_size = sample_size
        self.logger = logging.getLogger("text2query.chart_generator")

        # Import model utilities
        from ..core.utils.models import agenerate_chat
        self.agenerate_chat = agenerate_chat

    async def generate_chart(
        self,
        question: str,
        sql_command: str,
        columns: List[str],
        rows: List[List[Any]],
        populate_data: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate Chart.js configuration from query results.

        Args:
            question: User's natural language question
            sql_command: The SQL command that was executed
            columns: List of column names from query result
            rows: List of data rows from query result
            populate_data: Whether to populate the chart with actual data (default: True)

        Returns:
            Chart.js configuration dict with populated data, or None if generation fails.
            The returned dict contains:
            - chartjs_config: The Chart.js configuration object
            - data_labels_column: Column used for X-axis labels
            - datasets_labels_column: Columns used for datasets

        Raises:
            Exception: If chart generation fails
        """
        try:
            # Prepare sample data for LLM
            sample_rows = rows[:self.sample_size]
            sample_data = self._format_sample_data(columns, sample_rows)

            # Build prompt
            prompt = CHART_GENERATION_PROMPT.format(
                question=question,
                sql_command=sql_command,
                columns=columns,
                sample_data=sample_data,
                sample_size=min(self.sample_size, len(rows)),
                valid_types=', '.join(VALID_CHART_TYPES),
            )

            self.logger.info(f"Generating chart for question: {question[:50]}...")

            # Call LLM using LiteLLM
            messages = [{"role": "user", "content": prompt}]
            llm_response = await self.agenerate_chat(self.llm_config, messages)

            # Parse response
            json_resp = extract_json_from_string(llm_response)
            if not json_resp:
                self.logger.warning("Failed to parse LLM response as JSON")
                return None

            # Extract and validate chartjs components
            chartjs_config = safe_parse_chartjs_config(json_resp)
            data_labels_column = safe_parse_data_labels_column(json_resp)
            datasets_labels_column = safe_parse_datasets_labels_column(json_resp)

            if not chartjs_config:
                self.logger.warning("Failed to parse chartjs_config")
                return None

            if not data_labels_column:
                self.logger.warning("Failed to parse data_labels_column")
                return None

            if not datasets_labels_column:
                self.logger.warning("Failed to parse datasets_labels_column")
                return None

            # Validate configuration
            validated = check_chartjs(chartjs_config, data_labels_column, datasets_labels_column)
            if not validated:
                self.logger.warning("Chart.js validation failed")
                return None

            # Populate with actual data if requested
            if populate_data:
                populated_config = populate_chartjs_data(
                    validated['chartjs_config'],
                    validated['data_labels_column'],
                    validated['datasets_labels_column'],
                    columns,
                    rows,
                )
                if populated_config:
                    return {
                        'chartjs_config': populated_config,
                        'data_labels_column': validated['data_labels_column'],
                        'datasets_labels_column': validated['datasets_labels_column'],
                    }
                else:
                    self.logger.warning("Failed to populate chart data")
                    return None

            return validated

        except Exception as e:
            self.logger.error(f"Chart generation failed: {e}")
            raise Exception(f"Failed to generate chart: {str(e)}") from e

    def _format_sample_data(self, columns: List[str], rows: List[List[Any]]) -> str:
        """
        Format sample data for display in prompt.

        Args:
            columns: Column names
            rows: Data rows

        Returns:
            Formatted string representation of sample data
        """
        if not rows:
            return "No data available"

        lines = []
        # Header
        lines.append(" | ".join(str(col) for col in columns))
        lines.append("-" * len(lines[0]))

        # Data rows
        for row in rows:
            lines.append(" | ".join(str(val) for val in row))

        return "\n".join(lines)

    async def generate_chart_config_only(
        self,
        question: str,
        sql_command: str,
        columns: List[str],
        rows: List[List[Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Generate Chart.js configuration without populating data.

        This returns only the chart structure and column mappings,
        useful when you want to handle data population separately.

        Args:
            question: User's natural language question
            sql_command: The SQL command that was executed
            columns: List of column names from query result
            rows: List of data rows (only sample used for LLM context)

        Returns:
            Dict containing chartjs_config template, data_labels_column,
            and datasets_labels_column, or None if generation fails.
        """
        return await self.generate_chart(
            question=question,
            sql_command=sql_command,
            columns=columns,
            rows=rows,
            populate_data=False,
        )
