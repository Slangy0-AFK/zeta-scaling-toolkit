"""Extended matched-seed tiny Shakespeare benchmark (1000 steps, 20 seeds by default)."""

import sys

from real_task import main as run_benchmark


def main() -> None:
    arguments = sys.argv[1:]
    if "--seeds" in arguments:
        index = arguments.index("--seeds")
        end = next(
            (
                position
                for position in range(index + 1, len(arguments))
                if arguments[position].startswith("--")
            ),
            len(arguments),
        )
        seed_arguments = arguments[index + 1 : end]
        if len(seed_arguments) == 1:
            try:
                seed_count = int(seed_arguments[0])
            except ValueError:
                seed_count = 0
            if seed_count >= 10:
                arguments[index + 1 : end] = [str(seed) for seed in range(42, 42 + seed_count)]
    sys.argv = [sys.argv[0], *arguments]
    run_benchmark(default_steps=1000, default_seeds=range(42, 62))


if __name__ == "__main__":
    main()
