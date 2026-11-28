"""Opt-in hooks for test/prism. Inactive unless PRISM_KERNEL_GOLDEN is set (see kernel_golden.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import kernel_golden  # noqa: E402


def pytest_configure(config):
    if kernel_golden.install():
        config.pluginmanager.register(kernel_golden, "prism_kernel_golden")
