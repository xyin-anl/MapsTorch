import subprocess
import sys


def test_core_helpers_work_without_notebook_or_plotting_packages():
    """Block extras even in a developer environment that has them installed."""
    subprocess.run(
        [sys.executable, "-c", """
import importlib.abc
import sys

class NoExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'anywidget', 'traitlets', 'matplotlib', 'scienceplots',
            'marimo', 'plotly', 'pandas', 'scipy', 'xdrlib',
        }:
            raise ModuleNotFoundError(fullname, name=fullname)

sys.meta_path.insert(0, NoExtras())
import torch
import mapstorch
from mapstorch import default, map as model, opt, io, batchfit, util

assert util.get_peak_centers(['Fe'], 12.0, 90.0)['Fe_0'] > 0
for name in ('PeriodicTableWidget', 'BranchingRatioWidget'):
    try:
        getattr(util, name)
    except ImportError as exc:
        assert 'mapstorch[widget]' in str(exc)
    else:
        raise AssertionError('Widget access must require the widget extra')
try:
    import mapstorch.plot
except ImportError as exc:
    assert 'mapstorch[plot]' in str(exc)
else:
    raise AssertionError('Plotting must require the plot extra')
"""],
        check=True,
    )


def test_widgets_remain_usable_through_existing_app_imports():
    from mapstorch.util import BranchingRatioWidget, PeriodicTableWidget

    periodic = PeriodicTableWidget(initial_selected={"Fe": 0})
    assert periodic.selected_elements == {"Fe": 0}
    branching = BranchingRatioWidget(
        initial_lines="BRANCHING_RATIO_ADJUSTMENT_K: Fe, 1, 2, 3, 4"
    )
    assert branching.value["k"]["Fe"] == [1.0, 2.0, 3.0, 4.0]
