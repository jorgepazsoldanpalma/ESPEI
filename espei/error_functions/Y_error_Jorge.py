import yaml
YAML_LOADER = yaml.FullLoader
from tinydb import TinyDB, Query, where
from espei.utils import PickleableTinyDB, MemoryStorage, database_symbols_to_fit
from espei.datasets import DatasetError, load_datasets
from pycalphad import Database, calculate,Model, equilibrium, variables as v
from pycalphad.core.utils import instantiate_models, filter_phases, extract_parameters, unpack_components, unpack_condition
from espei.core_utils import ravel_conditions
from pycalphad.plot.eqplot import _map_coord_to_variable
from pycalphad.core.calculate import instantiate_models
from pycalphad.codegen.callables import build_phase_records
import numpy as np
import copy
from scipy.stats import norm
import logging
from espei.error_functions.residual_base import ResidualFunction, residual_function_registry
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Type, Union, Any
from espei.phase_models import PhaseModelSpecification
from espei.typing import SymbolName
import numpy.typing as npt
from espei.shadow_functions import equilibrium_, calculate_, no_op_equilibrium_, update_phase_record_parameters
from collections import OrderedDict


def get_Y_data(dbf: Database, comps: Sequence[str], phases: Sequence[str], datasets: PickleableTinyDB, parameters: Dict[str, float], model: Optional[Dict[str, Type[Model]]] = None):

    desired_data = datasets.search((where('output') == 'Y') &
                                   (where('components').test(lambda x: set(x).issubset(comps))) &
                                   (where('phases').test(lambda x: len(set(phases).intersection(x)) > 0)))
    Y_data=[]
    for data in desired_data:
        data_comps = list(set(data['components']).union({'VA'}))
        species = sorted(unpack_components(dbf, data_comps), key=str)
        phases= data['phases']
        ###Not sure if I need filter phases. Gotta come back to this one
        data_phases = filter_phases(dbf, species, candidate_phases=phases)
        ###################
        models = instantiate_models(dbf, species, data_phases, model=model, parameters=parameters)    
        site_fraction = data['values']
        conditions = data['conditions']
        model_phase=instantiate_models(dbf, data_comps, phases)
        sublattice=model_phase[phases[0]].site_fractions
        phase_recs = build_phase_records(dbf, species, data_phases, {v.N, v.P, v.T}, models, parameters=parameters, build_gradients=True, build_hessians=True)
        data_dict={
        'weight':data.get('weight', 1.0),
        'phase_records': phase_recs,
        'models': models,
        'species': species,
        'samples': site_fraction,
        'conditions': conditions,
         'components' : data_comps,
         'sublattice_info': sublattice,
        'phases': phases
        }
        Y_data.append(data_dict)
    return Y_data

def calculate_Y_difference(Y_data: Sequence[Dict[str, Any]],
                          parameters: np.ndarray,
                          approximate_equilibrium: Optional[bool] = False,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the sum of square error from site fraction data
    Parameters
    ----------
    dbf : pycalphad.Database
        Database to consider
    comps : list
        List of active component names
    phases : list
        List of phases to consider
    datasets : espei.utils.PickleableTinyDB
        Datasets that contain single phase data
    parameters : dict
        Dictionary of symbols that will be overridden in pycalphad.equilibrium
    phase_models : dict
        Phase models to pass to pycalphad calculations
    callables : dict
        Callables to pass to pycalphad
    data_weight : float
        Weight for standard deviation of activity measurements, dimensionless.
        Corresponds to the standard deviation of differences in chemical
        potential in typical measurements of activity, in J/mol.
    Returns
    -------
    float
        A single float of the sum of square errors
    Notes
    -----
    General procedure:
    1. Get the datasets
    2. For each dataset
        a. Calculate current site fraction 
        b. Find the target site fraction
        c. Calculate error due to site fraction
    """
    if len(Y_data) == 0:
        return 0.0
    if parameters is None:
        parameters = {}
        
    if approximate_equilibrium:
        _equilibrium = no_op_equilibrium_
    else:
        _equilibrium = equilibrium_
    Y_diff=[]
    weights_=[]  

    for data in Y_data:
        data_comps=data['components']
        conditions=data['conditions']
        conditions.setdefault('N', 1.0)
        species=data['species']
        phases=data['phases']
        phase_records=data['phase_records']
        models=data['models']
        sublattice=data['sublattice_info']
#        target_Y=np.array(data['samples']).flatten()
        target_Y=data['samples']
        
        composition_conds={key:val for key,val in conditions.items() if key!="T" and key!="P" and key!='N'}
        range_of_composition=[len(val) for key,val in conditions.items() if key!="T" and key!="P" and key!='N'][0]
        pot_conds = OrderedDict([(getattr(v, key), unpack_condition(data['conditions'][key]))\
                                 for key in sorted(conditions.keys())\
                                 if not key.startswith('X_')])
        str_statevar_dict = OrderedDict([(str(key), vals) for key, vals in pot_conds.items()])
        grid=calculate_(species, phases, str_statevar_dict
        , models, phase_records, pdens=50, fake_points=True)
        

        ######NEED TO ADD COMP CONDS TOMORROW. IF CAN CALCULATE AND EQUILIBRIUM IT WILL NOT BE TOO HARD###
        ###WILL NEED TO ACTUALLY TRY WITH GAS AFTERWARDS. MAYBE THINK ABOUT HOW TO IMPROVE DATASET#####
        for i in range(range_of_composition):
            finalized_conditions={}
            current_Y = []
            for element,val in composition_conds.items():
                finalized_conditions[element] = val[i]

            comp_conds= OrderedDict([(v.X(ele[2:]),val)\
                                     for ele,val in finalized_conditions.items()]) 
            
            cond_dict = OrderedDict(**pot_conds, **comp_conds)
            multi_eqdata = _equilibrium(phase_records, cond_dict, grid)
            eq_phases=np.squeeze(multi_eqdata.Phase) 
            eq_Y=np.squeeze(multi_eqdata.Y)


            result = eq_Y[np.logical_not(np.isnan(eq_Y))]
            if len(result)==0 or len(result)<len(sublattice):
                return -np.inf 
            elif len(result) > len(sublattice):
                result_st=eq_Y[0]
                result = result_st[np.logical_not(np.isnan(result_st))]

            current_Y=np.hstack((current_Y,result))
            
            current_target_Y=target_Y[0][0][i]

            weight = data.get('weight', 10.00)

            ind=[count for count,y_frac in enumerate(current_target_Y) if y_frac == None]
            current_target_Y=np.delete(current_target_Y,ind)
            current_Y=np.delete(current_Y,ind)
            current_result=current_target_Y - current_Y
            Y_diff.append(current_result)
            weights_.append(0.01)
#    pe =norm(loc=0, scale=0.01/(weight*data_weight)).logpdf(np.array(target_Y - current_Y, dtype=np.float64))
#    error += np.sum(pe)
#    logging.debug('Site_fraction error - data: {}, site_fraction difference: {}, probability: {}, reference: {}'.format(target_Y, current_Y-target_Y, pe, ds["reference"]))

        # TODO: write a test for this
 #   if np.any(np.isnan(np.array([error], dtype=np.float64))):  # must coerce sympy.core.numbers.Float to float64
#        return -np.inf

    return Y_diff,weights_


def calculate_Y_probability(Y_data: Sequence[Dict[str, Any]],
                                                     parameters: np.ndarray,
                                                     approximate_equilibrium: Optional[bool] = False,
                                                     ) -> float:
    """

    """
    if len(Y_data) == 0:
        return 0.0

    differences = []
    weights = []
    diffs, wts = calculate_Y_difference(Y_data, parameters, approximate_equilibrium)
    

    differences.append(diffs)
    weights.append(wts)

    differences = np.concatenate(differences, axis=0)
    weights = np.concatenate(weights, axis=0)
    likelihood=[]
    for i,j in zip(differences,weights):
        practice_like=sum([norm(loc=0.0, scale=j).logpdf(k) for k in i])
        if np.isnan(practice_like):
    #        # TODO: revisit this case and evaluate whether it is resonable for NaN
            # to show up here. When this comment was written, the test
            # test_subsystem_activity_probability would trigger a NaN.
            return -np.inf
        likelihood.append(practice_like)

    return np.sum(likelihood)

class EquilibriumSiteFractionResidual(ResidualFunction):
    def __init__(
        self,
        database: Database,
        datasets: PickleableTinyDB,
        phase_models: Union[PhaseModelSpecification, None],
        symbols_to_fit: Optional[List[SymbolName]] = None,
        weight: Optional[Dict[str, float]] = None,
        ):
        super().__init__(database, datasets, phase_models, symbols_to_fit)

        if weight is not None:
            self.weight = weight
        else:
            self.weight = {}

        if phase_models is not None:
            comps = sorted(phase_models.components)
            model_dict = phase_models.get_model_dict()
        else:
            comps = sorted(database.elements)
            model_dict = dict()
        phases = sorted(filter_phases(database, unpack_components(database, comps), database.phases.keys()))
        if symbols_to_fit is None:
            symbols_to_fit = database_symbols_to_fit(database)
        # okay if parameters are initialized to zero, we only need the symbol names
        parameters = dict(zip(symbols_to_fit, [0]*len(symbols_to_fit)))
        self.property_data = get_Y_data(database, comps, phases, datasets, parameters, model_dict)
    def get_residuals(self, parameters: npt.ArrayLike) -> Tuple[List[float], List[float]]:
        residuals = []
        weights = []
        dataset_residuals, dataset_weights = calculate_Y_difference(self.property_data, parameters)
        residuals.extend(dataset_residuals.tolist())
        weights.extend(dataset_weights.tolist())
        return residuals, weights

    def get_likelihood(self, parameters) -> float:
        likelihood = calculate_Y_probability(self.property_data, parameters)
        return likelihood


residual_function_registry.register(EquilibriumSiteFractionResidual)