"""Allow running Atlas as ``python -m atlas``."""

from atlas.atlas_cli.app import main

if __name__ == "__main__":
    raise SystemExit(main())
