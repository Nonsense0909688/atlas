"""Legacy top-level CLI import and launcher.

New code should import :func:`atlas.atlas_cli.app.main` instead.
"""
from atlas.atlas_cli.app import main

if __name__ == "__main__":
    raise SystemExit(main())
