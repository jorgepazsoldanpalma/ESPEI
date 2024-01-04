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


def get_partial_pressure_data(dbf: Database, comps: Sequence[str],
                                        phases: Sequence[str],
                                        datasets: PickleableTinyDB,
                                        model: Optional[Dict[str, Model]] = None,
                                        parameters: Optional[Dict[str, float]] = None,
                                        data_weight_dict: Optional[Dict[str, float]] = None,
                                        approximate_equilibrium: Optional[bool] = False):
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
    if approximate_equilibrium:
        _equilibrium = no_op_equilibrium_
    else:
        _equilibrium = equilibrium_
    desired_data = datasets.search(
        (tinydb.where('output').test(lambda x: 'PP' in x)) &
        (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
        (tinydb.where('phases').test(lambda x: set(x).issubset(set(phases))))        
        )
    
    pp_data=[] 
    for data in desired_data:
        data_output=data['output'].split('_')[0]
        params_keys, _ = extract_parameters(parameters)
        params_dict = OrderedDict(zip(map(str, params_keys), parameters))

        def_comp=data['output'].split('_')
        if len(def_comp)>1:
            _COMP='COMP'
        else:
            _COMP='NONE'
   
        data_comps = list(set(data['components']))
        species = sorted(unpack_components(dbf, data_comps), key=str)        
        data_phases = filter_phases(dbf, species, candidate_phases=data['phases'])

        gas_species=data['reference_state']['species']
        filter_gas_species=[spec for spec in species if spec.charge==0]
        for name,cons in gas_species.items():
            filter_gas_species=[spec for spec in filter_gas_species if spec.name==name]
            total_num_mol=sum([sto for ele,sto in cons.items()])            
            gas_comp_conds={'X_'+ele:sto/total_num_mol for ele,sto in cons.items()}
            gas_comp_conds.popitem()
        gas_phase=data['reference_state']['phases']
#        data_ref_state={'phases':gas_phase, 'species':filter_gas_species}
        samples=data['values']
        reference_state=data['reference_state']
        defined_comp=data['defined_components']
        models = instantiate_models(dbf, species, data_phases, model=model,\
        parameters=parameters)
        gas_models= instantiate_models(dbf, filter_gas_species, gas_phase,model=model, parameters=parameters)
        
        
        for mod in gas_models.values():
            mod.shift_partial_pressure(reference_state, dbf)

        phase_recs = build_phase_records(dbf
        , species
        , data_phases
        , {v.N, v.P, v.T}
        , models
        , parameters=parameters
        , build_gradients=True
        , build_hessians=True)
 
        gas_phase_recs= build_phase_records(dbf
        , filter_gas_species
        , gas_phase
        , {v.N, v.P, v.T}
        , gas_models
        , parameters=parameters
        , build_gradients=True
        , build_hessians=True)
        
        conditions=data['conditions']
        gas_conditions=data['reference_state']['conditions']
        potential_conds={key:val for key,val in conditions.items() if key=="T" or key=="P"}
        gas_potential_conds={key:val for key,val in gas_conditions.items() if key=="T" or key=="P"}

        gas_comp_conds=OrderedDict([(v.X(key[2:]), unpack_condition(gas_comp_conds[key])) \
        for key,val in sorted(gas_comp_conds.items()) \
        if key.startswith('X_') and val!=0.0])      
        
        
        
        if _COMP=='NONE':
            comp_conds={key:val for key,val in conditions.items() if key!="T" and key!="P"}
        else:
            def_components=data['defined_components']
            def_comp_conds={key.split('_')[1]:val for key,val in conditions.items() if key!="T" and key!="P"}
            comp_conds=calculating_pseudo_line(data_comps,def_components,def_comp_conds)     
        final_potential= OrderedDict([(getattr(v, key), unpack_condition(potential_conds[key])) for key in sorted(conditions.keys()) if not key.startswith('X_')])
        
        
        
        total_num_calculations=np.prod([len(vals) for vals in final_potential.values()])
        dataset_weights = np.array(data.get('weight', 1.0)) * np.ones(total_num_calculations)

        data_ref=data['reference']
        data_dict={
        'weight':dataset_weights,
        'conditions':gas_potential_conds,
        'references':data_ref,
        'database':dbf,
        'gas_model':gas_models,
        'model':models,
        'gas_spec_dict':gas_species,
        'gas_species':filter_gas_species,
        'species':species,
        'output':data_output,
        'data_phases':data_phases,
        'gas_phase':gas_phase,
        'phase_records': phase_recs,
        'gas_phase_records': gas_phase_recs,        
        'reference_state':reference_state,
        'component_dict': comp_conds,
        'gas_composition':gas_comp_conds,
        'Defined_components':_COMP,
        'param_variables':params_keys,
        'samples':samples
        }
        
        pp_data.append(data_dict)
    return pp_data        
 
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
def calculate_PP_difference(partial_pressure_data: Sequence[Dict[str, Any]],
                          parameters: np.ndarray,
                          approximate_equilibrium: Optional[bool] = False,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Notes
    -----
    General procedure:
    1. Get the datasets
    2. For each dataset

        a. Calculate the partial pressure of species
        b. Will also calculate total vapor pressure of the system 
        c. Compare with target partial pressure/Gibbs energy

    """
    std_dev = 500  # J/mol
    
    if approximate_equilibrium:
        _equilibrium = no_op_equilibrium_
    else:
        _equilibrium = equilibrium_
    residuals=[]
    defined_comp=partial_pressure_data['Defined_components']
    data_weight= partial_pressure_data['weight']
    therm_property='GM'  # the property of interest
    data_comps = partial_pressure_data['component_dict']
    gas_comps = partial_pressure_data['gas_composition']
    dbf=partial_pressure_data['database']
    potential_conds=partial_pressure_data['conditions']
    temp_list=potential_conds['T']
    gas_models=partial_pressure_data['gas_model']
    models=partial_pressure_data['model']
    phase_records=partial_pressure_data['phase_records']
    gas_phase_records=partial_pressure_data['gas_phase_records']
    update_phase_record_parameters(phase_records, parameters)
    update_phase_record_parameters(gas_phase_records, parameters)
    samples=partial_pressure_data['samples']
    param_keys=partial_pressure_data['param_variables']
    params_dict = OrderedDict(zip(map(str, param_keys), parameters))
    references=partial_pressure_data['references']
    filter_gas_species=partial_pressure_data['gas_species']
    species=partial_pressure_data['species']
    data_phases=partial_pressure_data['data_phases']
    gas_phase=partial_pressure_data['gas_phase']
    gas_spec=partial_pressure_data['gas_spec_dict']
    if defined_comp=='NONE':
        comp_cond=OrderedDict([(v.X(key[2:]), unpack_condition(data_comps[key])) for key,val in sorted(data_comps.items()) if key.startswith('X_')]) 
    elif defined_comp=='COMP':
        defined_unary_components=[key.split('_')[1] for key,val in data_comps.items() if val>0.0]
        depend_unary_components=sorted(defined_unary_components)[:-1]     
        comp_cond = OrderedDict([(v.X(key[2:]), unpack_condition(data_comps[key])) \
        for key,val in sorted(data_comps.items()) \
        if key.startswith('X_') and val!=0.0 and key[2:] in depend_unary_components]) 
        
    
    values=[]
    dataset_state_var=potential_conds
    pot_cond=OrderedDict([(getattr(v, key), unpack_condition(dataset_state_var[key])) for key in sorted(dataset_state_var.keys())])    
    dataset_state_var = OrderedDict([(str(key), vals) for key, vals in \
    sorted(dataset_state_var.items())])
    
    for temp in range(len(temp_list)):        
        potential_conds.setdefault('N', 1.0)
        potential_conds['T']=temp_list[temp]
        state_var=OrderedDict([(getattr(v, key), unpack_condition(potential_conds[key]))\
        for key in sorted(potential_conds.keys())])   
        ref_cond_dict= OrderedDict(**state_var, **gas_comps) 
    
        cond_dict = OrderedDict(**state_var, **comp_cond)
    
        grid=calculate_(species
        ,data_phases,potential_conds,models
        ,phase_records, pdens=50, fake_points=True)  
        
        ref_grid=calculate_(filter_gas_species
        ,gas_phase,potential_conds,gas_models
        ,gas_phase_records, pdens=50, fake_points=True)  


        multi_eqdata =_equilibrium(phase_records, 
            cond_dict, grid)  
        ref_multi_eqdata =_equilibrium(gas_phase_records, 
            ref_cond_dict, ref_grid)
            
        propdata = _eqcalculate(dbf, filter_gas_species, gas_phase, ref_cond_dict, 'GMR', data=ref_multi_eqdata, per_phase=False, callables=None, parameters=params_dict, model=gas_models)
        ref_gas_species_mu=getattr(propdata,'GMR').flatten().tolist()


        Chemical_Potentials=multi_eqdata.MU.squeeze()
        Components=multi_eqdata.coords['component']
        Dict_chem_pot={ele:mu for ele,mu in zip(Components,Chemical_Potentials)}
        num_of_specie=sum([stoi for j,i in gas_spec.items() for ele,stoi in i.items()])
        Chem_pot_spec=sum([Dict_chem_pot[ele]*stoi/num_of_specie for j,i in gas_spec.items() for ele,stoi in i.items()])
        RTemp= v.R*temp_list[temp]
        pp_spec=float(num_of_specie*(Chem_pot_spec-ref_gas_species_mu[0])/RTemp)
        pp_spec=np.exp(pp_spec)
        
        values.append(pp_spec)
        
    potential_conds['T']=temp_list



    
#    GAS_dataset_state_var=OrderedDict([(getattr(v, key), unpack_condition(dataset_state_var[key]))\
#    for key in sorted(dataset_state_var.keys())])    
#    cond_dict = OrderedDict(**pot_cond, **comp_cond)
#    ref_cond_dict= OrderedDict(**GAS_dataset_state_var, **gas_comp_conds)  

#    multi_eqdata =_equilibrium(phase_records, 
#        cond_dict, grid)  
#    ref_multi_eqdata =_equilibrium(Gas_phase_records, 
#        ref_cond_dict, ref_grid)
#    print('what is going on Part 2',ref_multi_eqdata.Y.squeeze())
#    propdata = _eqcalculate(dbf, gas_dataset_species, gas_dataset_phases, ref_cond_dict, 'GMR', data=ref_multi_eqdata, per_phase=False, callables=None, parameters=params_dict, model=gas_dataset_models)
#    ref_gas_species_mu=getattr(propdata,'GMR').flatten().tolist()
#    print('Gas Gibbs energy',ref_gas_species_mu)
        

    
    samples=np.array(samples).flatten()
    res=samples-values


    residuals.append(res)

    weights_ = (std_dev/data_weight).flatten()
        
    _log.debug('Data: %s, Partial_Pressure: difference: %s, reference: %s', values, residuals,references)

    return residuals, weights_


# TODO: roll this function into ActivityResidual

def calculate_PP_probability(partial_pressure_data: Sequence[Dict[str, Any]],
                        parameters: np.ndarray = None,
                        data_weight: int = 1.0,
                        approximate_equilibrium: bool = False) -> float:
                        
    if len(partial_pressure_data) == 0:
        return 0.0
    differences=[]
    wts=[]
    residuals, weights = calculate_PP_difference(partial_pressure_data[0], parameters)
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

class PartialPressurResidual(ResidualFunction):
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
        self.partial_pressure_data = get_partial_pressure_data(database, comps, phases, datasets, model_dict, parameters, data_weight_dict=self.weight)
############################################        

    def get_residuals(self, parameters: npt.ArrayLike) -> Tuple[List[float], List[float]]:
        residuals, weights =calculate_PP_difference(self.partial_pressure_data, parameters)
        return residuals, weights

    def get_likelihood(self, parameters: npt.NDArray) -> float:
        likelihood = calculate_PP_probability(self.partial_pressure_data, parameters, data_weight=self.weight)
        return likelihood


residual_function_registry.register(PartialPressurResidual)
