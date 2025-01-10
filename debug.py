"""Debugging entrypoint"""

if __name__ == "__main__":
    from pcb.__main__ import main

    # CLI args and options as a list of str
    # pylint: disable=no-value-for-parameter
    main(["--logging-level", "debug"])
