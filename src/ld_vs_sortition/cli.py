from __future__ import annotations

import argparse
from pathlib import Path

from .experiments import experiment1, experiment2, loss_bound


FULL_GAMMAS = [1e-9, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]
THEOREM_GAMMAS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]

PROFILES = {
    "quick": {
        "loss": {
            "n_list": [5, 10],
            "n_instances": 8,
            "graphs_per_instance": 120,
            "batch_size": 120,
        },
        "exp1": {
            "n_list": [10, 30],
            "gamma_list": [0.3, 1.0, 2.0],
            "n_mc": 120,
            "theorem_n_list": [10, 20],
            "theorem_gamma_list": [0.3, 1.0, 2.0],
            "theorem_graphs": 600,
            "theorem_mc": 600,
            "q_graphs": 1000,
        },
        "exp2": {
            "n_list": [30],
            "gamma_list": [0.3, 1.0, 2.0],
            "k_list": [1, 5],
            "n_mc": 15,
        },
    },
    "full": {
        "loss": {
            "n_list": [5, 10, 20, 50, 100],
            "n_instances": 500,
            "graphs_per_instance": 500,
            "batch_size": 500,
        },
        "exp1": {
            "n_list": [10, 50, 100, 200, 500],
            "gamma_list": FULL_GAMMAS,
            "n_mc": 1000,
            "theorem_n_list": [10, 20, 50, 100],
            "theorem_gamma_list": THEOREM_GAMMAS,
            "theorem_graphs": 100_000,
            "theorem_mc": 100_000,
            "q_graphs": 20_000,
        },
        "exp2": {
            "n_list": [50, 100, 200],
            "gamma_list": FULL_GAMMAS,
            "k_list": [1, 5, 10, 20],
            "n_mc": 200,
        },
    },
}


def parse_int_list(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("list may not be empty")
    return result


def parse_float_list(value: str) -> list[float]:
    try:
        result = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("list may not be empty")
    return result


def add_common_arguments(parser: argparse.ArgumentParser, default_output: str) -> None:
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--seed", type=int, default=2025)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ld-vs-sortition",
        description="Reproducible simulations for Liquid Democracy versus Sortition.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    loss = subparsers.add_parser("loss-bound", help="Check the singleton loss bound")
    add_common_arguments(loss, "outputs/loss_bound")
    loss.add_argument("--n-list", type=parse_int_list)
    loss.add_argument("--n-instances", type=int)
    loss.add_argument("--graphs-per-instance", type=int)
    loss.add_argument("--batch-size", type=int)

    exp1 = subparsers.add_parser("experiment1", help="Run the one-dimensional clique experiment")
    add_common_arguments(exp1, "outputs/experiment1")
    exp1.add_argument("--n-list", type=parse_int_list)
    exp1.add_argument("--gamma-list", type=parse_float_list)
    exp1.add_argument("--n-mc", type=int)
    exp1.add_argument("--theorem-n-list", type=parse_int_list)
    exp1.add_argument("--theorem-gamma-list", type=parse_float_list)
    exp1.add_argument("--theorem-graphs", type=int)
    exp1.add_argument("--theorem-mc", type=int)
    exp1.add_argument("--q-graphs", type=int)

    exp2 = subparsers.add_parser("experiment2", help="Run the two-dimensional SBM experiment")
    add_common_arguments(exp2, "outputs/experiment2")
    exp2.add_argument("--n-list", type=parse_int_list)
    exp2.add_argument("--gamma-list", type=parse_float_list)
    exp2.add_argument("--k-list", type=parse_int_list)
    exp2.add_argument("--n-mc", type=int)

    all_parser = subparsers.add_parser("all", help="Run all three experiment families")
    add_common_arguments(all_parser, "outputs/all")
    return parser


def override(profile: dict, arguments: argparse.Namespace, names: list[str]) -> dict:
    configured = dict(profile)
    for name in names:
        value = getattr(arguments, name, None)
        if value is not None:
            configured[name] = value
    return configured


def run_loss(arguments: argparse.Namespace, output_dir: str | Path | None = None) -> None:
    config = override(
        PROFILES[arguments.profile]["loss"],
        arguments,
        ["n_list", "n_instances", "graphs_per_instance", "batch_size"],
    )
    loss_bound.run_and_save(
        output_dir or arguments.output_dir,
        base_seed=arguments.seed,
        **config,
    )


def run_exp1(arguments: argparse.Namespace, output_dir: str | Path | None = None) -> None:
    config = override(
        PROFILES[arguments.profile]["exp1"],
        arguments,
        [
            "n_list",
            "gamma_list",
            "n_mc",
            "theorem_n_list",
            "theorem_gamma_list",
            "theorem_graphs",
            "theorem_mc",
            "q_graphs",
        ],
    )
    experiment1.run_and_save(
        output_dir or arguments.output_dir,
        base_seed=arguments.seed,
        **config,
    )


def run_exp2(arguments: argparse.Namespace, output_dir: str | Path | None = None) -> None:
    config = override(
        PROFILES[arguments.profile]["exp2"],
        arguments,
        ["n_list", "gamma_list", "k_list", "n_mc"],
    )
    experiment2.run_and_save(
        output_dir or arguments.output_dir,
        base_seed=arguments.seed,
        **config,
    )


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()

    if arguments.command == "loss-bound":
        run_loss(arguments)
    elif arguments.command == "experiment1":
        run_exp1(arguments)
    elif arguments.command == "experiment2":
        run_exp2(arguments)
    elif arguments.command == "all":
        root = Path(arguments.output_dir)
        run_loss(arguments, root / "loss_bound")
        run_exp1(arguments, root / "experiment1")
        run_exp2(arguments, root / "experiment2")
    else:
        parser.error(f"unknown command: {arguments.command}")


if __name__ == "__main__":
    main()
