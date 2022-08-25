"""
Fast versions of equilibrium and calculate that "override" the equivalent
pycalphad functions for very fast performance.
"""

from collections import OrderedDict
from typing import Sequence, Dict, Optional
from numpy.typing import ArrayLike
import numpy as np
from pycalphad import Model, variables as v
from pycalphad.core.phase_rec import PhaseRecord
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.starting_point import starting_point
from pycalphad.core.eqsolver import _solve_eq_at_conditions
from pycalphad.core.equilibrium import _adjust_conditions
from pycalphad.core.utils import get_state_variables, unpack_kwarg, point_sample
from pycalphad.core.light_dataset import LightDataset
from pycalphad.core.calculate import _sample_phase_constitution, _compute_phase_values
from pycalphad.core.solver import Solver


def update_phase_record_parameters(phase_records: Dict[str, PhaseRecord], parameters: ArrayLike) -> None:
    if parameters.size > 0:
        for phase_name, phase_record in phase_records.items():
            # very important that these are floats, otherwise parameters can end up
            # with garbage data. `np.asarray` does not create a copy if the type is
            # correct
            phase_record.parameters[:] = np.asarray(parameters, dtype=np.float_)

def _single_phase_start_point(conditions, state_variables, phase_records, grid):
    """Return a single CompositionSet object to use in a point calculation

    Assumes the grid has includes only candidate phases. The starting point will be
    generated by taking the minimum energy, regardless of whether mass balance is
    satisfied.

    Parameters
    ----------
    conditions : Dict[v.StateVariable, ArrayLike]
        Conditions in this region. Assumes state variable conditions have size == 1.
    state_variables : List[v.StateVariable]
        Active state variables in the calculation
    phase_records : Dict[str, PhaseRecord]
        Phase records, can have more than just the phase of interest
    grid : LightDataset
        Sampled grid of points

    """
    # assumes state variables in the conditions have size == 1
    idx_min = grid.GM.argmin()
    # Assumes ordering of dimensions is [state variables, points, data_variable...]
    # Get phase record
    phase_name = str(grid.Phase[..., idx_min].squeeze())
    prx = phase_records[phase_name]
    Y = grid.Y[..., idx_min, :].squeeze()[:prx.phase_dof]
    # Get current state variables
    # TODO: can we assume sorting
    state_vars = np.array([conditions[sv][0] for sv in sorted(state_variables, key=str)])
    compset = CompositionSet(prx)
    compset.update(Y, 1.0, state_vars)
    return compset


def calculate_(species: Sequence[v.Species], phases: Sequence[str],
               str_statevar_dict: Dict[str, np.ndarray], models: Dict[str, Model],
               phase_records: Dict[str, PhaseRecord], output: Optional[str] = 'GM',
               points: Optional[Dict[str, np.ndarray]] = None,
               pdens: Optional[int] = 50, broadcast: Optional[bool] = True,
               fake_points: Optional[bool] = False,
               ) -> LightDataset:
    """
    Quickly sample phase internal degree of freedom with virtually no overhead.
    """
    points_dict = unpack_kwarg(points, default_arg=None)
    pdens_dict = unpack_kwarg(pdens, default_arg=50)
    nonvacant_components = [x for x in sorted(species) if x.number_of_atoms > 0]
    maximum_internal_dof = max(prx.phase_dof for prx in phase_records.values())
    all_phase_data = []
    for phase_name in sorted(phases):
        mod = models[phase_name]
        phase_record = phase_records[phase_name]
        points = points_dict[phase_name]
        if points is None:
            points = _sample_phase_constitution(mod, point_sample, True, pdens_dict[phase_name])
        points = np.atleast_2d(points)

        fp = fake_points and (phase_name == sorted(phases)[0])
        phase_ds = _compute_phase_values(nonvacant_components, str_statevar_dict,
                                         points, phase_record, output,
                                         maximum_internal_dof, broadcast=broadcast,
                                         largest_energy=float(1e10), fake_points=fp,
                                         parameters={})
        all_phase_data.append(phase_ds)

    # assumes phase_records all have the same nonvacant pure elements,
    # even if those elements are not present in this phase record
    fp_offset = len(tuple(phase_records.values())[0].nonvacant_elements) if fake_points else 0
    running_total = [fp_offset] + list(np.cumsum([phase_ds['X'].shape[-2] for phase_ds in all_phase_data]))
    islice_by_phase = {phase_name: slice(running_total[phase_idx], running_total[phase_idx+1], None)
                       for phase_idx, phase_name in enumerate(sorted(phases))}

    if len(all_phase_data) > 1:
        concatenated_coords = all_phase_data[0].coords

        data_vars = all_phase_data[0].data_vars
        concatenated_data_vars = {}
        for var in data_vars.keys():
            data_coords = data_vars[var][0]
            points_idx = data_coords.index('points')  # concatenation axis
            arrs = []
            for phase_data in all_phase_data:
                arrs.append(getattr(phase_data, var))
            concat_data = np.concatenate(arrs, axis=points_idx)
            concatenated_data_vars[var] = (data_coords, concat_data)
        final_ds = LightDataset(data_vars=concatenated_data_vars, coords=concatenated_coords)
    else:
        final_ds = all_phase_data[0]
    final_ds.attrs['phase_indices'] = islice_by_phase
    return final_ds


def constrained_equilibrium(phase_records: Dict[str, PhaseRecord],
                 conditions: Dict[v.StateVariable, np.ndarray], grid: LightDataset):
    """Perform an equilibrium calculation with just a single composition set that is constrained to the global composition condition"""
    statevars = get_state_variables(conds=conditions)
    conditions = _adjust_conditions(conditions)
    # Assume that all conditions keys are lists with exactly one element (point calculation)
    str_conds = OrderedDict([(str(ky), conditions[ky][0]) for ky in sorted(conditions.keys(), key=str)])
    compset = _single_phase_start_point(conditions, statevars, phase_records, grid)
    solution_compsets = [compset]
    solver = Solver()
    # modifies `solution_compsets` and `compset` in place
    solver_result = solver.solve(solution_compsets, str_conds)
    energy = compset.NP * compset.energy
    return solver_result.converged, energy

def equilibrium_(phase_records: Dict[str, PhaseRecord],
                 conditions: Dict[v.StateVariable, np.ndarray], grid: LightDataset
                 ) -> LightDataset:
    """
    Perform a fast equilibrium calculation with virtually no overhead.
    """
    statevars = sorted(get_state_variables(conds=conditions), key=str)
    conditions = _adjust_conditions(conditions)
    str_conds = OrderedDict([(str(ky), conditions[ky]) for ky in sorted(conditions.keys(), key=str)])
    start_point = starting_point(conditions, statevars, phase_records, grid)

    return _solve_eq_at_conditions(start_point, phase_records, grid, str_conds, statevars, False)


def no_op_equilibrium_(phase_records: Dict[str, PhaseRecord],
                       conditions: Dict[v.StateVariable, np.ndarray],
                       grid: LightDataset,
                       ) -> LightDataset:
    """
    Perform a fast "equilibrium" calculation with virtually no overhead that
    doesn't refine the solution or do global minimization, but just returns
    the starting point.

    Notes
    -----
    Uses a placeholder first argument for the same signature as
    ``_equilibrium``, but ``species`` are not needed.

    """
    statevars = get_state_variables(conds=conditions)
    conditions = _adjust_conditions(conditions)
    return starting_point(conditions, statevars, phase_records, grid)
