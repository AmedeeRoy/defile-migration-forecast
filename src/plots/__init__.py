"""Plotting helpers.

Matplotlib is pinned to the non-interactive Agg backend here, at the single point every
plotting module is imported through. Training runs headless (CI, cron, ssh), where probing
for a GUI backend is at best wasted startup time and at worst an import-time failure, and
a run must never block on a window that nobody is there to close.
"""

import matplotlib

matplotlib.use("Agg")
