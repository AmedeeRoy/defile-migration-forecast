import pytest
import rootutils

# Makes `src` importable and resolves PROJECT_ROOT, the same way the entry points in `src/`
# do, so tests can be run from anywhere.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: test requires the live Open-Meteo API (deselected by default)"
    )


def pytest_collection_modifyitems(config, items):
    """Skips networked tests unless they are explicitly selected with `-m network`."""
    if "network" in config.getoption("-m"):
        return
    skip = pytest.mark.skip(reason="needs the live API; select it with -m network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
