"""`python -m voicefilter` -> start the web server (VF_HOST / VF_PORT)."""
from __future__ import annotations


def _main() -> None:
    import multiprocessing

    multiprocessing.freeze_support()
    from .server import main

    main()


if __name__ == "__main__":
    _main()
