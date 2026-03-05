import argparse
import os
import sys

from importlib import import_module

from .config import Config


def _load_callable(spec: str):
    try:
        if ":" in spec:
            module_name, fn_name = spec.split(":", 1)
        else:
            raise ValueError("Expected format MODULE:FUNCTION")
        module = import_module(module_name)
        fn = getattr(module, fn_name)

    except Exception as e:
        raise argparse.ArgumentTypeError(f"Could not load {spec!r}: {e}") from e
    
    if not callable(fn):
        raise argparse.ArgumentTypeError(f"{spec!r} is not callable")
    
    return fn


def main():

    parser = argparse.ArgumentParser(description="ConceptTracer: Interactive Analysis of Concept Saliency and Selectivity in Neural Representations")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # subparser for the calculations
    parser_calculations = subparsers.add_parser("calculations", help="Calculate saliency, selectivity, and p-values")
    parser_calculations.add_argument("--root", type=str, default=os.getcwd(), help="Root directory (default: os.getcwd())")
    parser_calculations.add_argument("--get_config_fn", type=str, default=None, help="Config loader, e.g., templates:get_config (default: Config in config.py)")
    parser_calculations.add_argument("--get_data_fn", type=str, default=None, help="Data loader, e.g., templates:get_data (default: get_data in helpers.py)")

    # subparser for the dashboard web app
    parser_app = subparsers.add_parser("app", help="Dashboard web app")
    parser_app.add_argument("--root", type=str, default=os.getcwd(), help="Root directory (default: os.getcwd())")
    parser_app.add_argument("--get_config_fn", type=str, default=None, help="Config loader, e.g., templates:get_config (default: Config in config.py)")
    parser_app.add_argument("--get_results_fn", type=str, default=None, help="Results loader, e.g., templates:get_results (default: get_results in helpers.py)")
    parser_app.add_argument("--task", type=str, default=None, help="Task name (default: first task in config)")
    parser_app.add_argument("--granularity", type=str, default=None, help="Granularity level (default: all granularities in config)")

    args = parser.parse_args()
    root_abs = os.path.abspath(args.root)
    if root_abs not in sys.path:
        sys.path.insert(0, root_abs)

    if args.command == "calculations":
        from . import calculations
        if args.get_config_fn:
            config_fn = _load_callable(args.get_config_fn)
            cfg = config_fn(root=root_abs)
        else:
            cfg = Config(root=root_abs)
        data_fn = _load_callable(args.get_data_fn) if args.get_data_fn else None
        calculations.run(cfg=cfg, get_data_fn=data_fn)
    elif args.command == "app":
        from . import app
        if args.get_config_fn:
            config_fn = _load_callable(args.get_config_fn)
            cfg = config_fn(root=root_abs)
        else:
            cfg = Config(root=root_abs)
        results_fn = _load_callable(args.get_results_fn) if args.get_results_fn else None
        app.run(cfg=cfg, get_results_fn=results_fn, task=args.task, granularity=args.granularity)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
