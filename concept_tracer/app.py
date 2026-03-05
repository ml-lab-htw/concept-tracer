import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
import torch

from dash import Dash, dcc, html, Input, Output, State, callback_context
from typing import Callable

from . import helpers
from .config import Config



def _graph_block(
        graph_id: str,
        height: int,
        graph_styles: dict
    ) -> html.Div:
    return html.Div(
        style={"height": f"{height}px"},
        children=[
            dcc.Graph(
                id=graph_id,
                style=graph_styles,
                config={
                    "responsive": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": [
                        "select2d",
                        "lasso2d",
                    ],
                    "toImageButtonOptions": {
                        "format": "svg",
                        "filename": graph_id,
                    },
                },
            )
        ],
    )

def _slider_default(max_k: int) -> int:
    return min(5, _slider_max(max_k))

def _slider_marks(max_k: int) -> dict:
    return {i: str(i) for i in range(1, max(1, max_k) + 1)}

def _slider_max(max_k: int) -> int:
    return max(1, min(10, max_k))


GetResultsFn = Callable[
    [Config, str | None, str | None],
    pd.DataFrame,
]


def run(cfg: Config | None = None, get_results_fn: GetResultsFn | None = None, task: str | None = None, granularity: str | None = None):
    
    if cfg is None:
        cfg = Config()
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    if get_results_fn is None:
        get_results_fn = helpers.get_results

    # =========================
    # DATA
    # =========================
    tasks = cfg.dataset_specs[cfg.dataset_name].get("tasks", []) or []
    # selection of first task by default if applicable, but can be changed
    if tasks and task is None:
        task = tasks[0]
    DF = get_results_fn(cfg, task=task, granularity=granularity)

    DF = DF[
        DF["p_saliency"].notna()
        & DF["p_selectivity"].notna()
        & DF["p_combined"].notna()
        # faster and more readable
        & (DF["p_saliency"] < cfg.significance_threshold)
        & (DF["p_selectivity"] < cfg.significance_threshold)
        & (DF["p_combined"] < cfg.significance_threshold)
    ].copy()



    # =========================
    # CONSTANTS
    # =========================
    STATS = {
        "saliency": {"min": DF["saliency"].min(), "max": DF["saliency"].max()},
        "selectivity": {"min": DF["selectivity"].min(), "max": DF["selectivity"].max()},
    }

    ALL_LAYERS = sorted(DF["layer"].unique())
    ALL_CONCEPTS = sorted(DF["concept"].unique())
    ROW_HEIGHT = max(20, int(cfg.heatmap_height / max(1, len(ALL_LAYERS))))



    # =========================
    # APP
    # =========================
    app = Dash(__name__)
    app.title = "ConceptTracer"

    app.index_string = app.index_string.replace(
        "</head>",
        f"""
            <style>
            #layer-checklist * {{ box-sizing: border-box !important; }}
            #layer-checklist {{ height: {cfg.heatmap_height}px !important; }}
            #layer-checklist > div {{ height: {ROW_HEIGHT}px !important; line-height: {ROW_HEIGHT}px !important; }}
            #layer-checklist label {{ height: {ROW_HEIGHT}px !important; line-height: {ROW_HEIGHT}px !important; }}
            </style>
            </head>
        """
    )

    app.layout = html.Div(
        style=cfg.styles["page"],
        children=[
            html.H2("ConceptTracer"),
            dcc.Store(id="selection-store", data={"layers": [], "neurons": []}),
            dcc.Store(id="context-all-concepts-store", data={"layers": [], "neurons": []}),
            dcc.Store(id="context-one-concept-store", data={"layers": [], "neurons": [], "concept": None}),
            html.Div(
                style={**cfg.styles["row"], "alignItems": "flex-start"},
                children=[
                    # LEFT
                    html.Div(
                        style=cfg.styles["pane"],
                        children=[
                            html.Div(
                                style=cfg.styles["align_center"],
                                children=[
                                    html.Label("Metric"),
                                    dcc.Dropdown(
                                        id="metric-dropdown",
                                        value="saliency",
                                        clearable=False,
                                        options=[
                                            {"label": "Max Saliency over Concepts", "value": "saliency"},
                                            {"label": "Max Selectivity over Concepts", "value": "selectivity"},
                                            {"label": "Max Saliency and Selectivity over Concepts", "value": "combined"},
                                        ],
                                        style={"flex": 1, "maxWidth": "520px"},
                                    ),
                                    html.Button("All layers", id="all-layers", n_clicks=0),
                                ],
                            ),
                            html.Div("Layers", style={"fontWeight": 600, "marginTop": "12px"}),
                            html.Div(
                                style=cfg.styles["align_top"],
                                children=[
                                    html.Div(
                                        style={"minWidth": "120px", "paddingBottom": "20px", "boxSizing": "border-box"},
                                        children=[
                                            dcc.Checklist(
                                                id="layer-checklist",
                                                options=[{"label": str(l), "value": int(l)} for l in reversed(ALL_LAYERS)],
                                                value=[],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={
                                            "flex": 1,
                                            "minWidth": 0,
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "marginLeft": "-60px",
                                        },
                                        children=[
                                            _graph_block("heatmap", cfg.heatmap_height, cfg.styles["graph"]),
                                            html.Div("Neurons", style={
                                                "fontWeight": 600,
                                                "textAlign": "center",
                                                "height": "20px",
                                                "lineHeight": "20px",
                                                "marginTop": "6px",
                                                },
                                            ),
                                        ],
                                    ),

                                ],
                            ),
                            html.Div(
                                f"Check one or more boxes to select layers (empty = ALL layers). Or click a heatmap cell to select a neuron. Note that insignificant values are excluded for improved speed and readability.",
                                style={"fontSize": f"{cfg.font_size - 2}px", "opacity": 0.7, "whiteSpace": "pre-line", "marginTop": "12px", "marginBottom": "12px"},
                            ),
                            html.Div(id="selection-summary", style={"fontSize": f"{cfg.font_size}px", "opacity": 0.9}),
                        ],
                    ),

                    # RIGHT
                    html.Div(
                        style=cfg.styles["pane"],
                        children=[
                            dcc.Tabs(
                                id="right-tabs",
                                value="all-concepts",
                                children=[
                                    dcc.Tab(
                                        label="All Concepts",
                                        value="all-concepts",
                                        children=html.Div(
                                            style=cfg.styles["tabs"],
                                            children=[
                                                html.Div(
                                                    style=cfg.styles["row"],
                                                    children=[
                                                        html.Div(
                                                            style={**cfg.styles["col"], "flex": "1 1 260px", "minWidth": 0},
                                                            children=[
                                                                _graph_block("saliency-hist-all-concepts", cfg.hist_height, cfg.styles["graph"]),
                                                                html.Div("Top-k saliencies:", style=cfg.styles["top_k_label"]),
                                                                dcc.Slider(
                                                                    id="top-k-saliency-all-concepts",
                                                                    min=1,
                                                                    max=10,
                                                                    step=1,
                                                                    value=5,
                                                                    marks=_slider_marks(10),
                                                                    updatemode="mouseup",
                                                                    included=False,
                                                                ),
                                                                html.Ol(id="top-saliency-all-concepts"),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            style={**cfg.styles["col"], "flex": "1 1 260px", "minWidth": 0},
                                                            children=[
                                                                _graph_block("selectivity-hist-all-concepts", cfg.hist_height, cfg.styles["graph"]),
                                                                html.Div("Top-k selectivities:", style=cfg.styles["top_k_label"]),
                                                                dcc.Slider(
                                                                    id="top-k-selectivity-all-concepts",
                                                                    min=1,
                                                                    max=10,
                                                                    step=1,
                                                                    value=5,
                                                                    marks=_slider_marks(10),
                                                                    updatemode="mouseup",
                                                                    included=False,
                                                                ),
                                                                html.Ol(id="top-selectivity-all-concepts"),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(),
                                                html.Div(
                                                    style={"minWidth": 0},
                                                    children=[_graph_block("scatter-all-concepts", cfg.scatter_height, cfg.styles["graph"])],
                                                ),
                                                html.Div("Top-k Pareto points:", style=cfg.styles["top_k_label"]),
                                                dcc.Slider(
                                                    id="top-k-pareto-all-concepts",
                                                    min=1,
                                                    max=10,
                                                    step=1,
                                                    value=5,
                                                    marks=_slider_marks(10),
                                                    updatemode="mouseup",
                                                    included=False,
                                                ),
                                                html.Ol(id="top-pareto-all-concepts"),
                                            ],
                                        ),
                                    ),
                                    dcc.Tab(
                                        label="Search concept",
                                        value="search-concept",
                                        children=html.Div(
                                            style=cfg.styles["tabs"],
                                            children=[
                                                html.Div(
                                                    style={**cfg.styles["align_center"], "minWidth": 0},
                                                    children=[
                                                        html.Div("Concept", style={"minWidth": "80px"}),
                                                        dcc.Dropdown(
                                                            id="concept-dropdown",
                                                            options=[{"label": c, "value": c} for c in ALL_CONCEPTS[:cfg.concept_limit]],
                                                            value=None,
                                                            placeholder="Search concept…",
                                                            searchable=True,
                                                            clearable=True,
                                                            style={"flex": 1, "minWidth": 0},
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    style=cfg.styles["row"],
                                                    children=[
                                                        html.Div(
                                                            style={**cfg.styles["col"], "flex": "1 1 260px", "minWidth": 0},
                                                            children=[
                                                                _graph_block("saliency-hist-one-concept", cfg.hist_height, cfg.styles["graph"]),
                                                                html.Div("Top-k saliencies:", style=cfg.styles["top_k_label"]),
                                                                dcc.Slider(
                                                                    id="top-k-saliency-one-concept",
                                                                    min=1,
                                                                    max=10,
                                                                    step=1,
                                                                    value=5,
                                                                    marks=_slider_marks(10),
                                                                    updatemode="mouseup",
                                                                    included=False,
                                                                ),
                                                                html.Ol(id="top-saliency-one-concept"),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            style={**cfg.styles["col"], "flex": "1 1 260px", "minWidth": 0},
                                                            children=[
                                                                _graph_block("selectivity-hist-one-concept", cfg.hist_height, cfg.styles["graph"]),
                                                                html.Div("Top-k selectivities:", style=cfg.styles["top_k_label"]),
                                                                dcc.Slider(
                                                                    id="top-k-selectivity-one-concept",
                                                                    min=1,
                                                                    max=10,
                                                                    step=1,
                                                                    value=5,
                                                                    marks=_slider_marks(10),
                                                                    updatemode="mouseup",
                                                                    included=False,
                                                                ),
                                                                html.Ol(id="top-selectivity-one-concept"),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(),
                                                html.Div(
                                                    style={"minWidth": 0},
                                                    children=[_graph_block("scatter-one-concept", cfg.scatter_height, cfg.styles["graph"])],
                                                ),
                                                html.Div("Top-k Pareto points:", style=cfg.styles["top_k_label"]),
                                                dcc.Slider(
                                                    id="top-k-pareto-one-concept",
                                                    min=1,
                                                    max=10,
                                                    step=1,
                                                    value=5,
                                                    marks=_slider_marks(10),
                                                    updatemode="mouseup",
                                                    included=False,
                                                ),
                                                html.Ol(id="top-pareto-one-concept"),
                                            ],
                                        ),
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )



    # =========================
    # CALLBACKS
    # =========================
    @app.callback(
        Output("selection-store", "data"),
        Output("selection-summary", "children"),
        Output("layer-checklist", "value"),
        Input("layer-checklist", "value"),
        Input("all-layers", "n_clicks"),
        Input("heatmap", "clickData"),
        State("selection-store", "data"),
    )
    def update_selection(layer_values, n_clicks, clickData, current):
        current = current or {"layers": [], "neurons": []}
        prop_id = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

        ui_layers = [int(x) for x in (layer_values or [])]
        layers = list(current.get("layers", []))
        neurons = list(current.get("neurons", []))

        if prop_id == "all-layers.n_clicks":
            ui_layers, layers, neurons = [], [], []

        elif prop_id == "layer-checklist.value":
            layers = ui_layers[:]
            neurons = []

        elif prop_id == "heatmap.clickData" and clickData and "points" in clickData:
            p = clickData["points"][0]
            layers = [int(p["y"])]
            neurons = [int(p["x"])]
            ui_layers = []

        layer_txt = ",".join(map(str, layers)) if layers else "ALL"
        neuron_txt = f"{neurons[0]}" if (len(layers) == 1 and len(neurons) == 1) else "ALL"
        summary = f"Selection → layers: {layer_txt} | neurons: {neuron_txt}"

        return {"layers": layers, "neurons": neurons}, summary, ui_layers


    @app.callback(
        Output("heatmap", "figure"),
        Input("metric-dropdown", "value"),
    )
    def update_heatmap(metric_mode):
        return helpers.get_heatmap_dashboard(DF, STATS, metric_mode, cfg)


    @app.callback(
        Output("concept-dropdown", "options"),
        Input("concept-dropdown", "search_value"),
        State("concept-dropdown", "value"),
    )
    def concept_options(search_value, current_value):
        if not search_value:
            base = ALL_CONCEPTS[: cfg.concept_limit]
            if current_value and current_value not in base:
                base = [current_value] + base
            return [{"label": concept, "value": concept} for concept in base[: cfg.concept_limit]]
        
        matches = [concept for concept in ALL_CONCEPTS if concept.lower().startswith(str(search_value).lower())]
        if current_value and current_value not in matches:
            matches = [current_value] + matches

        return [{"label": concept, "value": concept} for concept in matches[: cfg.concept_limit]]


    @app.callback(
        Output("saliency-hist-all-concepts", "figure"),
        Output("selectivity-hist-all-concepts", "figure"),
        Output("scatter-all-concepts", "figure"),
        Output("top-pareto-all-concepts", "children"),
        Output("top-saliency-all-concepts", "children"),
        Output("top-selectivity-all-concepts", "children"),
        Output("top-k-pareto-all-concepts", "max"),
        Output("top-k-pareto-all-concepts", "value"),
        Output("top-k-pareto-all-concepts", "marks"),
        Output("top-k-saliency-all-concepts", "max"),
        Output("top-k-saliency-all-concepts", "value"),
        Output("top-k-saliency-all-concepts", "marks"),
        Output("top-k-selectivity-all-concepts", "max"),
        Output("top-k-selectivity-all-concepts", "value"),
        Output("top-k-selectivity-all-concepts", "marks"),
        Output("context-all-concepts-store", "data"),
        Input("selection-store", "data"),
        Input("top-k-pareto-all-concepts", "value"),
        Input("top-k-saliency-all-concepts", "value"),
        Input("top-k-selectivity-all-concepts", "value"),
        State("context-all-concepts-store", "data"),
    )
    def update_all_concepts_view(selection, top_k_pareto, top_k_saliency, top_k_selectivity, prev_ctx):
        selection = selection or {"layers": [], "neurons": []}
        prev_ctx = prev_ctx or {"layers": [], "neurons": []}

        selection_changed = (
            sorted(selection.get("layers", [])) != sorted(prev_ctx.get("layers", []))
            or sorted(selection.get("neurons", [])) != sorted(prev_ctx.get("neurons", []))
        )

        df = DF
        layers = selection.get("layers", [])
        neurons = selection.get("neurons", [])
        if layers:
            df = df[df["layer"].isin(layers)]
        if neurons:
            df = df[df["neuron"].isin(neurons)]

        fig_saliency = helpers.get_hist_plot_dashboard(
            df["saliency"], title="Saliency over Concepts", xlabel="Saliency", cfg=cfg
        )
        fig_selectivity = helpers.get_hist_plot_dashboard(
            df["selectivity"], title="Selectivity over Concepts", xlabel="Selectivity", cfg=cfg
        )

        if df.empty:
            fig_scatter = go.Figure().update_layout(
                template=cfg.template, height=cfg.scatter_height, title="Saliency vs Selectivity"
            )

            k_max_pareto = k_max_saliency = k_max_selectivity = 1
            if selection_changed:
                k_val_pareto = k_val_saliency = k_val_selectivity = 1
            else:
                k_val_pareto = int(max(1, min(k_max_pareto, int(top_k_pareto or 1))))
                k_val_saliency = int(max(1, min(k_max_saliency, int(top_k_saliency or 1))))
                k_val_selectivity = int(max(1, min(k_max_selectivity, int(top_k_selectivity or 1))))

            return (
                fig_saliency,
                fig_selectivity,
                fig_scatter,
                [],
                [],
                [],
                k_max_pareto, k_val_pareto, _slider_marks(k_max_pareto),
                k_max_saliency, k_val_saliency, _slider_marks(k_max_saliency),
                k_max_selectivity, k_val_selectivity, _slider_marks(k_max_selectivity),
                {"layers": list(layers), "neurons": list(neurons)},
            )
        
        df_plot = df.copy()
        df_plot["id_scatter"] = (
            "Concept "
            + df_plot["concept"].astype(str)
            + ", layer "
            + df_plot["layer"].astype(str)
            + ", neuron "
            + df_plot["neuron"].astype(str)
        )

        df_scatter = df_plot.set_index("id_scatter")[["saliency", "selectivity"]]
        fig_scatter = helpers.get_scatter_plot_dashboard(df_scatter, STATS, title="Saliency vs Selectivity", cfg=cfg)

        k_max_pareto = _slider_max(len(helpers.get_pareto_front(df_scatter)))
        k_max_saliency = _slider_max(len(df_scatter))
        k_max_selectivity = _slider_max(len(df_scatter))
        if selection_changed:
            k_val_pareto = _slider_default(k_max_pareto)
            k_val_saliency = _slider_default(k_max_saliency)
            k_val_selectivity = _slider_default(k_max_selectivity)
        else:
            k_val_pareto = int(max(1, min(k_max_pareto, int(top_k_pareto or 1))))
            k_val_saliency = int(max(1, min(k_max_saliency, int(top_k_saliency or 1))))
            k_val_selectivity = int(max(1, min(k_max_selectivity, int(top_k_selectivity or 1))))

        top_k_items_pareto = helpers.get_top_k_items(df_scatter, STATS, k_val_pareto, "combined")
        top_k_items_saliency = helpers.get_top_k_items(df_scatter, STATS, k_val_saliency, "saliency")
        top_k_items_selectivity = helpers.get_top_k_items(df_scatter, STATS, k_val_selectivity, "selectivity")

        return (
            fig_saliency,
            fig_selectivity,
            fig_scatter,
            top_k_items_pareto,
            top_k_items_saliency,
            top_k_items_selectivity,
            k_max_pareto, k_val_pareto, _slider_marks(k_max_pareto),
            k_max_saliency, k_val_saliency, _slider_marks(k_max_saliency),
            k_max_selectivity, k_val_selectivity, _slider_marks(k_max_selectivity),
            {"layers": list(layers), "neurons": list(neurons)},
        )


    @app.callback(
        Output("saliency-hist-one-concept", "figure"),
        Output("selectivity-hist-one-concept", "figure"),
        Output("scatter-one-concept", "figure"),
        Output("top-pareto-one-concept", "children"),
        Output("top-saliency-one-concept", "children"),
        Output("top-selectivity-one-concept", "children"),
        Output("top-k-pareto-one-concept", "max"),
        Output("top-k-pareto-one-concept", "value"),
        Output("top-k-pareto-one-concept", "marks"),
        Output("top-k-saliency-one-concept", "max"),
        Output("top-k-saliency-one-concept", "value"),
        Output("top-k-saliency-one-concept", "marks"),
        Output("top-k-selectivity-one-concept", "max"),
        Output("top-k-selectivity-one-concept", "value"),
        Output("top-k-selectivity-one-concept", "marks"),
        Output("context-one-concept-store", "data"),
        Input("selection-store", "data"),
        Input("concept-dropdown", "value"),
        Input("top-k-pareto-one-concept", "value"),
        Input("top-k-saliency-one-concept", "value"),
        Input("top-k-selectivity-one-concept", "value"),
        State("context-one-concept-store", "data"),
    )
    def update_one_concept_view(selection, concept, top_k_pareto, top_k_saliency, top_k_selectivity, prev_ctx):
        selection = selection or {"layers": [], "neurons": []}
        prev_ctx = prev_ctx or {"layers": [], "neurons": [], "concept": None}

        selection_changed = (
            sorted(selection.get("layers", [])) != sorted(prev_ctx.get("layers", []))
            or sorted(selection.get("neurons", [])) != sorted(prev_ctx.get("neurons", []))
            or concept != prev_ctx.get("concept", None)
        )

        df = DF
        layers = selection.get("layers", [])
        neurons = selection.get("neurons", [])
        if layers:
            df = df[df["layer"].isin(layers)]
        if neurons:
            df = df[df["neuron"].isin(neurons)]

        ctx_out = {
            "layers": list(layers),
            "neurons": list(neurons),
            "concept": concept,
        }

        empty_title = None
        if not concept:
            empty_title = "Select a concept"
        else:
            df_concept = df[df["concept"] == str(concept)]
            if df_concept.empty:
                empty_title = "No data for concept"

        if empty_title is not None:
            empty_hist = go.Figure().update_layout(
                template=cfg.template, height=cfg.hist_height, title=empty_title,
            )
            empty_scatter = go.Figure().update_layout(
                template=cfg.template, height=cfg.scatter_height, title=empty_title,
            )

            k_max_pareto = k_max_saliency = k_max_selectivity = 1
            if selection_changed:
                k_val_pareto = k_val_saliency = k_val_selectivity = 1
            else:
                k_val_pareto = int(max(1, min(k_max_pareto, int(top_k_pareto or 1))))
                k_val_saliency = int(max(1, min(k_max_saliency, int(top_k_saliency or 1))))
                k_val_selectivity = int(max(1, min(k_max_selectivity, int(top_k_selectivity or 1))))

            return (
                empty_hist, empty_hist, empty_scatter,
                [], [], [],
                k_max_pareto, k_val_pareto, _slider_marks(k_max_pareto),
                k_max_saliency, k_val_saliency, _slider_marks(k_max_saliency),
                k_max_selectivity, k_val_selectivity, _slider_marks(k_max_selectivity),
                ctx_out,
            )
        
        df_plot = df_concept.copy()
        df_plot["id_scatter"] = (
            "Concept "
            + str(concept)
            + ", layer "
            + df_plot["layer"].astype(str)
            + ", neuron "
            + df_plot["neuron"].astype(str)
        )

        fig_saliency = helpers.get_hist_plot_dashboard(
            df_plot["saliency"], title="Saliency over Neurons", xlabel="Saliency", cfg=cfg
        )
        fig_selectivity = helpers.get_hist_plot_dashboard(
            df_plot["selectivity"], title="Selectivity over Neurons", xlabel="Selectivity", cfg=cfg
        )

        df_scatter = df_plot.set_index("id_scatter")[["saliency", "selectivity"]]
        fig_scatter = helpers.get_scatter_plot_dashboard(df_scatter, STATS, title="Saliency vs Selectivity", cfg=cfg)

        k_max_pareto = _slider_max(len(helpers.get_pareto_front(df_scatter)))
        k_max_saliency = _slider_max(len(df_scatter))
        k_max_selectivity = _slider_max(len(df_scatter))

        if selection_changed:
            k_val_pareto = _slider_default(k_max_pareto)
            k_val_saliency = _slider_default(k_max_saliency)
            k_val_selectivity = _slider_default(k_max_selectivity)
        else:
            k_val_pareto = int(max(1, min(k_max_pareto, int(top_k_pareto or 1))))
            k_val_saliency = int(max(1, min(k_max_saliency, int(top_k_saliency or 1))))
            k_val_selectivity = int(max(1, min(k_max_selectivity, int(top_k_selectivity or 1))))

        top_k_items_pareto = helpers.get_top_k_items(df_scatter, STATS, k_val_pareto, "combined")
        top_k_items_saliency = helpers.get_top_k_items(df_scatter, STATS, k_val_saliency, "saliency")
        top_k_items_selectivity = helpers.get_top_k_items(df_scatter, STATS, k_val_selectivity, "selectivity")

        return (
            fig_saliency,
            fig_selectivity,
            fig_scatter,
            top_k_items_pareto,
            top_k_items_saliency,
            top_k_items_selectivity,
            k_max_pareto, k_val_pareto, _slider_marks(k_max_pareto),
            k_max_saliency, k_val_saliency, _slider_marks(k_max_saliency),
            k_max_selectivity, k_val_selectivity, _slider_marks(k_max_selectivity),
            ctx_out,
        )
    


    # go to http://127.0.0.1:8050/ to view dashboard
    app.run(port=8050, host="127.0.0.1")



def main():
    run()



if __name__ == "__main__":
   main()
