"""Compatibility entry; prefer python -m tests.mla (E2E by default)."""

from tests.mla.support.mla_validation import main

if __name__ == "__main__":
    raise SystemExit(main())
