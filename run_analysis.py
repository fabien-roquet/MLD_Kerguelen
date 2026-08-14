#!/usr/bin/env python3
"""Run the modular three-stage MLD Kerguelen analysis."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_SCRIPTS = {
    "CMA": PROJECT_ROOT / "1_data_processing" / "process_CMA.py",
    "GLORYS": PROJECT_ROOT / "1_data_processing" / "process_GLORYS.py",
    "GLORYS_CL": PROJECT_ROOT / "1_data_processing" / "process_GLORYS_CL.py",
    "GLORYS_SECTION": PROJECT_ROOT / "1_data_processing" / "process_GLORYS_section.py",
}

FPCA_SCRIPTS = {
    "CMA": PROJECT_ROOT / "2_compute_fPCA_R" / "script_PCA_CM_2026.R",
    "GLORYS": PROJECT_ROOT / "2_compute_fPCA_R" / "script_PCA_GLORYS_2026.R",
    "GLORYS_CL": PROJECT_ROOT / "2_compute_fPCA_R" / "script_PCA_GLORYS_CL_2026.R",
}

FIGURE_SCRIPTS = {
    "1": PROJECT_ROOT / "3_generate_figures" / "Figure_1_prof_dist_year.py",
    "2": PROJECT_ROOT / "3_generate_figures" / "Figure_2_map_mld.py",
    "3": PROJECT_ROOT / "3_generate_figures" / "Figure_3_seasonal_cycle.py",
    "4": PROJECT_ROOT / "3_generate_figures" / "Figure_4_MU_modes.py",
    "5": PROJECT_ROOT / "3_generate_figures" / "Figure_5_maps_cp.py",
    "6": PROJECT_ROOT / "3_generate_figures" / "Figure_6_quadratic_error.py",
    "7": PROJECT_ROOT / "3_generate_figures" / "Figure_7_1D_trends.py",
    "8": PROJECT_ROOT / "3_generate_figures" / "Figure_8_trend_maps.py",
    "9": PROJECT_ROOT / "3_generate_figures" / "Figure_9_sections.py",
    "10": PROJECT_ROOT / "3_generate_figures" / "Figure_10_KERFIX.py",
    "A1": PROJECT_ROOT / "3_generate_figures" / "Figure_A1_PACE_sampling_trends.py",
    "A2": PROJECT_ROOT / "3_generate_figures" / "Figure_A2_PACE_sampling_trend_maps.py",
    "A3": PROJECT_ROOT / "3_generate_figures" / "Figure_A3_deep_region_trends.py",
    "A4": PROJECT_ROOT / "3_generate_figures" / "Figure_A4_shallow_region_trends.py",
    "A5": PROJECT_ROOT / "3_generate_figures" / "Figure_A5_deep_north_saf_trends.py",
    "S1": PROJECT_ROOT / "3_generate_figures" / "Figure_S1_regions_trends.py",
    "S2": PROJECT_ROOT / "3_generate_figures" / "Figure_S2_PACE_K_diagnostics.py",
}

DEFAULT_FIGURES = ["2", "3", "4", "5", "6", "7", "8", "9", "10"]
TREND_TABLE_SCRIPT = PROJECT_ROOT / "3_generate_figures" / "Table_1_trends.py"
R_SETUP_SCRIPT = PROJECT_ROOT / "scripts" / "setup_r_packages.R"
GLORYS_RANDOM_SAMPLING_SCRIPT = PROJECT_ROOT / "2_compute_fPCA_R" / "script_PCA_GLORYS_random_sampling_2026.R"


def read_pyproject() -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        try:
            import tomli as tomllib
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing tomli. Run `uv sync` before launching the pipeline with Python 3.10.") from exc

    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)


def r_config() -> dict:
    pyproject = read_pyproject()
    config = pyproject.get("tool", {}).get("mld-kerguelen", {}).get("r", {})
    return {
        "packages": list(config.get("packages", [])),
        "repos": config.get("repos", "https://cloud.r-project.org"),
        "library": PROJECT_ROOT / config.get("library", ".r-lib"),
    }


def r_environment(config: dict | None = None) -> dict[str, str]:
    config = config or r_config()
    env = os.environ.copy()
    r_lib = str(Path(config["library"]).resolve())
    env["MLD_R_LIB"] = r_lib
    env["MLD_R_REPOS"] = str(config["repos"])
    existing_libs = env.get("R_LIBS_USER")
    env["R_LIBS_USER"] = r_lib if not existing_libs else os.pathsep.join([r_lib, existing_libs])
    return env


def run_command(command: list[str], cwd: Path = PROJECT_ROOT, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def ensure_r_packages() -> dict[str, str]:
    if shutil.which("Rscript") is None:
        raise SystemExit(
            "R is required for the fPCA stage, but `Rscript` was not found. "
            "Install R first, then rerun `uv run python run_analysis.py --stage setup`."
        )

    config = r_config()
    packages = config["packages"]
    env = r_environment(config)
    if packages:
        run_command(["Rscript", str(R_SETUP_SCRIPT), str(PROJECT_ROOT), str(config["repos"]), *packages], env=env)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=["setup", "data", "fpca", "sampling", "figures", "compare"],
        default=["data", "fpca", "figures", "compare"],
        help="Pipeline stages to run.",
    )
    parser.add_argument("--datasets", nargs="+", choices=DATA_SCRIPTS.keys(), default=list(DATA_SCRIPTS.keys()))
    parser.add_argument("--figures", nargs="+", choices=FIGURE_SCRIPTS.keys(), default=DEFAULT_FIGURES)
    parser.add_argument("--force-data", action="store_true", help="Recompute data-processing outputs.")
    parser.add_argument("--force-sampling", action="store_true", help="Recompute GLORYS random-sampling PACE outputs.")
    parser.add_argument("--sampling-replicates", type=int, default=30, help="Number of pseudo-random sampling replicates.")
    parser.add_argument("--sampling-levels", default="5,10,20", help="Comma-separated GLORYS sampling percentages.")
    parser.add_argument("--sampling-seed", type=int, default=20260526, help="Base seed for reproducible random sampling.")
    parser.add_argument("--skip-reference-compare", action="store_true", help="Alias for omitting the compare stage.")
    parser.add_argument("--skip-r-setup", action="store_true", help="Do not check/install R packages before fPCA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = set(args.stage)
    if args.skip_reference_compare:
        stages.discard("compare")

    r_env = None
    if ("setup" in stages or "fpca" in stages or "sampling" in stages) and not args.skip_r_setup:
        r_env = ensure_r_packages()
    elif "fpca" in stages or "sampling" in stages:
        r_env = r_environment()

    if "data" in stages:
        for dataset in args.datasets:
            command = [sys.executable, str(DATA_SCRIPTS[dataset]), "--project-root", str(PROJECT_ROOT)]
            if args.force_data:
                command.append("--force")
            run_command(command)

    if "fpca" in stages:
        for dataset in args.datasets:
            script = FPCA_SCRIPTS.get(dataset)
            if script is not None:
                run_command(["Rscript", str(script)], env=r_env)

    if "sampling" in stages:
        command = [
            "Rscript",
            str(GLORYS_RANDOM_SAMPLING_SCRIPT),
            "--project-root",
            str(PROJECT_ROOT),
            "--replicates",
            str(args.sampling_replicates),
            "--levels",
            args.sampling_levels,
            "--seed",
            str(args.sampling_seed),
        ]
        if args.force_sampling:
            command.append("--force")
        run_command(command, env=r_env)

    if "figures" in stages:
        for figure in args.figures:
            run_command([sys.executable, str(FIGURE_SCRIPTS[figure]), "--project-root", str(PROJECT_ROOT)])
        if "7" in args.figures:
            run_command([sys.executable, str(TREND_TABLE_SCRIPT), "--project-root", str(PROJECT_ROOT)])

    if "compare" in stages:
        generated_dir = PROJECT_ROOT / "processed"
        reference_dir = PROJECT_ROOT / "processed_reference"
        if reference_dir.exists():
            run_command([sys.executable, str(PROJECT_ROOT / "scripts" / "compare_processed_reference.py")])
        else:
            if not generated_dir.exists():
                raise FileNotFoundError(f"Generated processed directory not found: {generated_dir}")
            shutil.copytree(generated_dir, reference_dir)
            print(f"Created initial reference snapshot at {reference_dir} from {generated_dir}")


if __name__ == "__main__":
    main()
