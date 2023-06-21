"""
Calculate error due to measured activities.

The residual function implemented in this module needs to exist because it is
currently not possible to compute activity as a property via equilibrium
calculations because as PyCalphad does not yet have a suitable notion of a
reference state that could be used for equilibrium chemical potentials.

"""



import logging
from dataclasses import dataclass

from typing import Dict, List, NamedTuple, Optional, Any, Sequence, Tuple, Type, Union
from collections import OrderedDict

import numpy as np
import numpy.typing as npt
import tinydb
from pycalphad import Database, Model, equilibrium, variables as v
from pycalphad.plot.eqplot import _map_coord_to_variable
from pycalphad.core.utils import filter_phases, unpack_components, instantiate_models, extract_parameters, unpack_condition
from scipy.stats import norm

from sympy import exp, log, Abs, Add, And, Float, Mul, Piecewise, Pow, S
from espei.core_utils import ravel_conditions
from pycalphad.codegen.callables import build_phase_records
from espei.error_functions.residual_base import ResidualFunction, residual_function_registry
from espei.phase_models import PhaseModelSpecification
from espei.typing import SymbolName
from espei.utils import database_symbols_to_fit, PickleableTinyDB
from espei.shadow_functions import equilibrium_, calculate_, no_op_equilibrium_, update_phase_record_parameters
from pycalphad.core.phase_rec import PhaseRecord
from pycalphad.core.light_dataset import LightDataset
from pycalphad.core.equilibrium import _eqcalculate

_log = logging.getLogger(__name__)

@dataclass
class PotentialRegion:
    potential_conds : Dict[v.StateVariable, float]
    species:Sequence[v.Species]
    phases: Sequence[str]
    models: Dict[str,Model]


def get_fusion_data(dbf: Database, comps: Sequence[str],
                                        phases: Sequence[str],
                                        datasets: PickleableTinyDB,
                                        model: Optional[Dict[str, Model]] = None,
                                        parameters: Optional[Dict[str, float]] = None,
                                        data_weight_dict: Optional[Dict[str, float]] = None,
                                        ):
    """
    Get all the EqPropData for each matching equilibrium thermochemical dataset in the datasets

    Parameters
    ----------
    dbf : Database
        Database with parameters to fit
    comps : Sequence[str]
        List of pure element components used to find matching datasets.
    phases : Sequence[str]
        List of phases used to search for matching datasets.
    datasets : PickleableTinyDB
        Datasets that contain single phase data
    model : Optional[Dict[str, Type[Model]]]
        Dictionary phase names to pycalphad Model classes.
    parameters : Optional[Dict[str, float]]
        Mapping of parameter symbols to values.
    data_weight_dict : Optional[Dict[str, float]]
        Mapping of a data type (ACR for both pure and component references) to a weight.

    Notes
    -----
    Found datasets will be subsets of the components and phases. Equilibrium
    thermochemical data is assumed to be any data that does not have the
    `solver` key, and does not have an output of `ZPF` or `ACR` (which
    correspond to different data types than can be calculated here.)

    Returns
    -------
    Sequence[EqPropData]
    """

    desired_data = datasets.search(
        (tinydb.where('output').test(lambda x: 'FUSI' in x)) &
        (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
        (tinydb.where('phases').test(lambda x: set(x).issubset(set(phases))))        
        )

    fusion_data=[] 
    for data in desired_data:
        data_output=data['output'].split('_')[0]
        params_keys, _ = extract_parameters(parameters)
        def_comp=data['output'].split('_')
        if len(def_comp)>2:
            _COMP='COMP'
        else:
            _COMP='NONE'
        data_comps = list(set(data['components']))
        species = sorted(unpack_components(dbf, data_comps), key=str)
        if data['phases'] is None:
            data_phases = filter_phases(dbf, species, candidate_phases=phases)
        else:
            data_phases=data['phases']
        ref_data_phases=data['reference_state']['phases']
        samples=data['values']
        models = instantiate_models(dbf, species, data_phases, model=model, parameters=parameters)
        ref_models=instantiate_models(dbf, species, ref_data_phases, model=model, parameters=parameters)
        phase_recs = build_phase_records(dbf
        , species
        , data_phases
        , {v.N, v.P, v.T}
        , models
        , parameters=parameters
        , build_gradients=True
        , build_hessians=True)
        
        ref_phase_records= build_phase_records(dbf
        , species
        , ref_data_phases
        , {v.N, v.P, v.T}
        , ref_models
        , parameters=parameters
        , build_gradients=True
        , build_hessians=True)
        
        conditions=data['conditions']
        ref_conditions=data['reference_state']['conditions']

        potential_conds={key:val for key,val in conditions.items() if key=="T" or key=="P"}
        potential_conds.setdefault('N', 1.0)
        Potential_Reg=PotentialRegion(potential_conds,species,data_phases,models)
        
        if _COMP=='NONE':
            comp_conds={key:val for key,val in conditions.items() if key!="T" and key!="P"}
        else:
            def_components=data['defined_components']
            def_comp_conds={key.split('_')[1]:val for key,val in conditions.items() if key!="T" and key!="P"}
            comp_conds=calculating_pseudo_line(data_comps,def_components,def_comp_conds)     
            
        ref_potential={key:value for key,value in ref_conditions.items() if not key.startswith('X_')}
        ref_potential.setdefault('N', 1.0)
        ref_compositions={key:val for key,val in ref_conditions.items() if key!="T" and key!="P"}
        ref_Potential_Reg=PotentialRegion(ref_potential,species,ref_data_phases,ref_models)
        
        
        final_ref_potential=OrderedDict([(getattr(v, key), unpack_condition(ref_potential[key])) for key in sorted(ref_potential.keys()) if not key.startswith('X_')])  
        final_potential= OrderedDict([(getattr(v, key), unpack_condition(potential_conds[key])) for key in sorted(potential_conds.keys()) if not key.startswith('X_')])
        
        total_num_calculations=np.prod([len(vals) for vals in final_potential.values()])
        dataset_weights = np.array(data.get('weight', 1.0)) * np.ones(total_num_calculations)

        
        data_ref=data['reference']
        data_dict={
        'weight':dataset_weights,
        'database':dbf,
        'output':data_output,
        'Potential_region':Potential_Reg,
        'Ref_Potential_region':ref_Potential_Reg,
        'phase_records':phase_recs,
        'reference_phase_records':ref_phase_records,
        'dataset_reference':data_ref,
        'component_dict': comp_conds,
        'Defined_components':_COMP,
        'param_variables':params_keys,
        'samples':samples
        }
        
        fusion_data.append(data_dict)
    return fusion_data        
 
def calculating_pseudo_line(elemental_composition,defined_components,component_fractions):

    pseudo_line=defined_components
    checking_type=[True for i in component_fractions.values() if isinstance(i,list)==True][0]
    if checking_type==True:
        dependent_comp=1-sum([i[0] for i in component_fractions.values()])
    elif checking_type!=True:
        dependent_comp=1-sum([i for i in component_fractions.values()])
    for key,value in defined_components.items():
        if key not in component_fractions:   
            component_fractions[key]=dependent_comp
    final_amount={}
    tot_moles=S.Zero
    for i in elemental_composition:
        fun_list=[]
        for comp,value in component_fractions.items():
            if isinstance(value,list)==True:
                value=value[0]
            else:
                pass
            for new_comp,new_value in pseudo_line[comp].items():
                if i==new_comp:
                    fun_list.append(value*new_value)
                    final_fun_list=sum(fun_list)
                    final_amount['X_'+i]=final_fun_list
                    tot_moles+=value*new_value
    
    for final_comp,final_value in final_amount.items():
        final_amount[final_comp]=float(final_value/tot_moles)
    
    component_fractions.popitem()
    
    return final_amount

        
# TODO: roll this function into ActivityResidual
def calculate_fusion_residuals(fusion_data: Sequence[Dict[str, Any]],
                          parameters: np.ndarray,
                          approximate_equilibrium: Optional[bool] = False,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Notes
    -----
    General procedure:
    1. Get the datasets
    2. For each dataset

        a. Calculate reference state equilibrium
        b. Calculate current chemical potentials
        c. Find the target chemical potentials
        d. Calculate error due to chemical potentials
    """
    std_dev = 500  # J/mol
    
    if approximate_equilibrium:
        _equilibrium = no_op_equilibrium_
    else:
        _equilibrium = equilibrium_
    residuals=[]
#    for data in fusion_data:
    defined_comp=fusion_data['Defined_components']
    data_weight= fusion_data['weight']
    therm_property=fusion_data['output']  # the property of interest
    data_comps = fusion_data['component_dict']
    dbf=fusion_data['database']
    def_com_ref_condition={}
    phase_records=fusion_data['phase_records']
    ref_phase_records=fusion_data['reference_phase_records']
    update_phase_record_parameters(phase_records, parameters)
    update_phase_record_parameters(ref_phase_records, parameters)
    dataset_species=fusion_data['Potential_region'].species
    dataset_phases=fusion_data['Potential_region'].phases
    dataset_models=fusion_data['Potential_region'].models
    dataset_state_var=fusion_data['Potential_region'].potential_conds
    ref_dataset_species=fusion_data['Ref_Potential_region'].species
    ref_dataset_phases=fusion_data['Ref_Potential_region'].phases
    ref_dataset_models=fusion_data['Ref_Potential_region'].models
    ref_dataset_state_var=fusion_data['Ref_Potential_region'].potential_conds       
    samples=fusion_data['samples']
    param_keys=fusion_data['param_variables']
    params_dict = OrderedDict(zip(map(str, param_keys), parameters))
    pot_cond=OrderedDict([(getattr(v, key), unpack_condition(dataset_state_var[key])) for key in sorted(dataset_state_var.keys())])    
    dataset_state_var = OrderedDict([(str(key), vals) for key, vals in sorted(dataset_state_var.items())])
    grid=calculate_(dataset_species
    ,dataset_phases,dataset_state_var,dataset_models
    ,phase_records, pdens=50, fake_points=True)
    
    ref_grid=calculate_(ref_dataset_species
    ,ref_dataset_phases,ref_dataset_state_var,ref_dataset_models
    ,ref_phase_records, pdens=50, fake_points=True)
    
    
    ref_dataset_state_var=OrderedDict([(getattr(v, key), unpack_condition(ref_dataset_state_var[key])) for key in sorted(ref_dataset_state_var.keys())])    
    
    if defined_comp=='NONE':
        comp_cond=OrderedDict([(v.X(key[2:]), unpack_condition(data_comps[key])) for key,val in sorted(data_comps.items()) if key.startswith('X_')]) 
    elif defined_comp=='COMP':
        defined_unary_components=[key.split('_')[1] for key,val in data_comps.items() if val>0.0]
        depend_unary_copmponents=sorted(defined_unary_components)[:-1]     
        comp_cond = OrderedDict([(v.X(key[2:]), unpack_condition(data_comps[key])) \
        for key,val in sorted(data_comps.items()) \
        if key.startswith('X_') and val!=0.0 and key[2:] in depend_unary_copmponents])           
    cond_dict = OrderedDict(**pot_cond, **comp_cond)    
    ref_cond_dict= OrderedDict(**ref_dataset_state_var, **comp_cond)   
    
    multi_eqdata =_equilibrium(phase_records, 
        cond_dict, grid)  
    ref_multi_eqdata =_equilibrium(ref_phase_records, 
        ref_cond_dict, ref_grid)             
    propdata = _eqcalculate(dbf, dataset_species, dataset_phases, cond_dict
    , therm_property, data=multi_eqdata, per_phase=False, callables=None, parameters=params_dict, model=dataset_models)
    ref_propdata = _eqcalculate(dbf, ref_dataset_species, ref_dataset_phases, ref_cond_dict
    , therm_property, data=ref_multi_eqdata, per_phase=False, callables=None, parameters=params_dict, model=ref_dataset_models)
    ref_values=getattr(ref_propdata, therm_property).flatten().tolist()[0]
    values=getattr(propdata, therm_property).flatten().tolist()
    values=[val-ref_values for val in values]
    res=np.array(samples).flatten()[0]-values

    residuals.append(res)

    weights_ = (std_dev/data_weight).flatten()
        

    _log.debug('Data: %s, Fusion:q!: difference: %s, reference: %s', values, residuals,fusion_data['dataset_reference'])

    return residuals, weights_


# TODO: roll this function into ActivityResidual

def calculate_fusion_error(fusion_data: Sequence[Dict[str, Any]],
                        parameters: np.ndarray = None,
                        data_weight: int = 1.0,
                        approximate_equilibrium: bool = False) -> float:
                        
    if len(fusion_data) == 0:
        return 0.0
    differences=[]
    wts=[]
    for fus_dat in fusion_data:    
        residuals, weights = calculate_fusion_residuals(fus_dat, parameters)
        if np.any(np.isinf(residuals) | np.isnan(residuals)):
            # NaN or infinity are assumed calculation failures. If we are
            # calculating log-probability, just bail out and return -infinity.
            return -np.inf
        differences.append(residuals[0])
        wts.append(weights)       
    differences = np.concatenate(differences, axis=0)
    weights = np.concatenate(wts, axis=0)
    likelihood = norm(loc=0.0, scale=weights).logpdf(differences)
    if np.isnan(likelihood).any():
#        # TODO: revisit this case and evaluate whether it is resonable for NaN
        # to show up here. When this comment was written, the test
        # test_subsystem_activity_probability would trigger a NaN.
        return -np.inf
    return np.sum(likelihood)

class FusionResidual(ResidualFunction):
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
        self._symbols_to_fit = symbols_to_fit


###JORGE IS ADDING LINES HERE
        parameters = dict(zip(symbols_to_fit, [0]*len(symbols_to_fit)))
        self.fusion_data = get_fusion_data(database, comps, phases, datasets, model_dict, parameters, data_weight_dict=self.weight)
############################################        

    def get_residuals(self, parameters: npt.ArrayLike) -> Tuple[List[float], List[float]]:
        residuals, weights =calculate_fusion_residuals(self.fusion_data, parameters)
        return residuals, weights

    def get_likelihood(self, parameters: npt.NDArray) -> float:
        likelihood = calculate_fusion_error(self.fusion_data, parameters, data_weight=self.weight)
        return likelihood


residual_function_registry.register(FusionResidual)
