#!/usr/bin/env python
import importlib
import traceback


def main() -> None:
    """
    Run the HSR MuJoCo lab demo without a viewer and exit non-zero with a traceback on
    failure.
    """
    try:
        # The hyphen in the file name rules out an import statement.
        demo_lab = importlib.import_module("demo-lab")

        demo_lab.main(headless=True)
    except Exception:
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
