"""Experimental task-weighted ranking with the original strict protection set."""
import numpy as np
from .task_selection_v1 import select_task


def strict_weighted_guard(means,weights,reference=0):
    """Maximize explicit task gain among candidates with no training-axis loss.

    Compared with the historical guard this changes ranking, not feasibility.
    Training-objective domination does not imply heldout domination.
    """
    r=select_task(means,weights,np.zeros(np.asarray(means).shape[-1]),
                  'point_constrained',reference=reference)
    r['policy']='strict_weighted_guard'
    return r
