"""
Visualization Tools - Generate charts and graphs from query results
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..core.base import UserContext
from ..core.registry import (
    Tool,
    ToolCategory,
    ToolResult,
    ToolSchema,
    PermissionLevel,
)

logger = logging.getLogger(__name__)


class ChartType:
    """Supported chart types"""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    TABLE = "table"


class ChartGeneratorTool(Tool):
    """
    Generates Plotly chart specifications from query results

    Supports:
    - Automatic chart type inference
    - Multiple chart types (bar, line, pie, scatter, etc.)
    - Responsive design
    - Dark/light theme
    """

    name = "generate_chart"
    description = "Generate a visualization chart from query results"
    category = ToolCategory.VISUALIZATION
    permission_level = PermissionLevel.AUTHENTICATED

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "data": {
                    "type": "array",
                    "description": "Query result data as list of dicts",
                },
                "columns": {
                    "type": "array",
                    "description": "Column names",
                },
                "chart_type": {
                    "type": "string",
                    "description": "Chart type (bar, line, pie, scatter, area, histogram)",
                    "enum": ["bar", "line", "pie", "scatter", "area", "histogram", "auto"],
                    "default": "auto",
                },
                "x_column": {
                    "type": "string",
                    "description": "Column for x-axis",
                },
                "y_column": {
                    "type": "string",
                    "description": "Column for y-axis",
                },
                "title": {
                    "type": "string",
                    "description": "Chart title",
                },
                "theme": {
                    "type": "string",
                    "description": "Color theme",
                    "enum": ["light", "dark"],
                    "default": "light",
                },
            },
            required=["data", "columns"],
        )

    async def execute(
        self,
        user_context: UserContext,
        data: List[Dict],
        columns: List[str],
        chart_type: str = "auto",
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        title: Optional[str] = None,
        theme: str = "light",
    ) -> ToolResult:
        try:
            if not data:
                return ToolResult(
                    success=True,
                    data={"type": "empty", "message": "No data to visualize"},
                )

            # Auto-detect chart type if needed
            if chart_type == "auto":
                chart_type = self._infer_chart_type(data, columns)

            # Auto-select columns if not specified
            if not x_column or not y_column:
                x_column, y_column = self._select_columns(data, columns, chart_type)

            # Generate Plotly spec
            spec = self._generate_plotly_spec(
                data=data,
                chart_type=chart_type,
                x_column=x_column,
                y_column=y_column,
                title=title,
                theme=theme,
            )

            return ToolResult(
                success=True,
                data={
                    "type": chart_type,
                    "spec": spec,
                    "columns_used": {"x": x_column, "y": y_column},
                },
            )

        except Exception as e:
            logger.exception("Chart generation failed")
            return ToolResult(success=False, error=str(e))

    def _infer_chart_type(
        self,
        data: List[Dict],
        columns: List[str],
    ) -> str:
        """Infer the best chart type based on data characteristics"""
        if not data or not columns:
            return ChartType.TABLE

        row_count = len(data)
        col_count = len(columns)

        # Analyze column types
        numeric_cols = []
        categorical_cols = []
        date_cols = []

        for col in columns:
            sample_value = data[0].get(col)

            if isinstance(sample_value, (int, float)):
                numeric_cols.append(col)
            elif self._is_date_like(str(sample_value)):
                date_cols.append(col)
            else:
                categorical_cols.append(col)

        # Decision logic
        if date_cols and numeric_cols:
            return ChartType.LINE  # Time series

        if len(categorical_cols) == 1 and len(numeric_cols) == 1:
            if row_count <= 10:
                return ChartType.PIE
            return ChartType.BAR

        if len(numeric_cols) >= 2:
            return ChartType.SCATTER

        if categorical_cols and numeric_cols:
            return ChartType.BAR

        return ChartType.TABLE

    def _select_columns(
        self,
        data: List[Dict],
        columns: List[str],
        chart_type: str,
    ) -> tuple[str, str]:
        """Select appropriate x and y columns"""
        if not columns:
            return "", ""

        numeric_cols = []
        categorical_cols = []

        for col in columns:
            if data and isinstance(data[0].get(col), (int, float)):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        if chart_type in [ChartType.BAR, ChartType.PIE]:
            x_col = categorical_cols[0] if categorical_cols else columns[0]
            y_col = numeric_cols[0] if numeric_cols else columns[-1]
        elif chart_type == ChartType.LINE:
            x_col = columns[0]
            y_col = numeric_cols[0] if numeric_cols else columns[-1]
        elif chart_type == ChartType.SCATTER:
            x_col = numeric_cols[0] if numeric_cols else columns[0]
            y_col = numeric_cols[1] if len(numeric_cols) > 1 else columns[-1]
        else:
            x_col = columns[0]
            y_col = columns[-1] if len(columns) > 1 else columns[0]

        return x_col, y_col

    def _generate_plotly_spec(
        self,
        data: List[Dict],
        chart_type: str,
        x_column: str,
        y_column: str,
        title: Optional[str],
        theme: str,
    ) -> Dict[str, Any]:
        """Generate Plotly chart specification"""
        x_values = [row.get(x_column) for row in data]
        y_values = [row.get(y_column) for row in data]

        # Base layout
        layout = {
            "title": {"text": title or f"{y_column} by {x_column}"},
            "template": "plotly_dark" if theme == "dark" else "plotly_white",
            "autosize": True,
            "margin": {"l": 50, "r": 50, "t": 50, "b": 50},
        }

        # Chart-specific trace
        if chart_type == ChartType.BAR:
            trace = {
                "type": "bar",
                "x": x_values,
                "y": y_values,
                "marker": {"color": "#636EFA"},
            }
            layout["xaxis"] = {"title": x_column}
            layout["yaxis"] = {"title": y_column}

        elif chart_type == ChartType.LINE:
            trace = {
                "type": "scatter",
                "mode": "lines+markers",
                "x": x_values,
                "y": y_values,
                "line": {"color": "#636EFA"},
            }
            layout["xaxis"] = {"title": x_column}
            layout["yaxis"] = {"title": y_column}

        elif chart_type == ChartType.PIE:
            trace = {
                "type": "pie",
                "labels": x_values,
                "values": y_values,
                "hole": 0.3,  # Donut style
            }

        elif chart_type == ChartType.SCATTER:
            trace = {
                "type": "scatter",
                "mode": "markers",
                "x": x_values,
                "y": y_values,
                "marker": {"color": "#636EFA", "size": 10},
            }
            layout["xaxis"] = {"title": x_column}
            layout["yaxis"] = {"title": y_column}

        elif chart_type == ChartType.AREA:
            trace = {
                "type": "scatter",
                "mode": "lines",
                "fill": "tozeroy",
                "x": x_values,
                "y": y_values,
                "line": {"color": "#636EFA"},
            }
            layout["xaxis"] = {"title": x_column}
            layout["yaxis"] = {"title": y_column}

        elif chart_type == ChartType.HISTOGRAM:
            trace = {
                "type": "histogram",
                "x": y_values,  # Histogram uses single column
                "marker": {"color": "#636EFA"},
            }
            layout["xaxis"] = {"title": y_column}
            layout["yaxis"] = {"title": "Count"}

        else:
            # Default to bar
            trace = {
                "type": "bar",
                "x": x_values,
                "y": y_values,
            }

        return {
            "data": [trace],
            "layout": layout,
        }

    def _is_date_like(self, value: str) -> bool:
        """Check if a string looks like a date"""
        import re
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
            r"\d{2}/\d{2}/\d{4}",  # MM/DD/YYYY
            r"\d{2}-\d{2}-\d{4}",  # DD-MM-YYYY
        ]
        return any(re.match(p, value) for p in date_patterns)
