"""
run_pipeline.py

Orchestrates the pipeline:
1) environment: load/generate traversability map + save .npy + save env png
2) graph_build: load .npy + build graph + save graph pngs
"""

from environment import main as env_main
from graph_build import main as graph_main
from environment import load_config


def main():
    cfg = load_config()
    pipe = cfg.get("pipeline", {})

    if bool(pipe.get("run_environment", True)):
        env_main()

    if bool(pipe.get("run_graph", True)):
        graph_main()


if __name__ == "__main__":
    main()
