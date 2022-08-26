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

_log = logging.getLogger(__name__)

@dataclass
class ChemPotentialRegion:
    potential_conds : Dict[v.StateVariable, float]
    species:Sequence[v.Species]
    phases: Sequence[str]
    models: Dict[str,Model]

def get_activity_data(dbf: Database, comps: Sequence[str],
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
        (tinydb.where('output').test(lambda x: 'ACR' in x)) &
        (tinydb.where('components').test(lambda x: set(x).issubset(comps))))

    activity_data=[] 
    stoich_ref=[]
    for data in desired_data:
        data_defined_components=data['output'].split('_')[1]
        data_comps = list(set(data['components']))
        species = sorted(unpack_components(dbf, data_comps), key=str)

        if data['phases'] is None:
            data_phases = filter_phases(dbf, species, candidate_phases=phases)
        else:
            data_phases=data['phases']
            
        samples=data['values']
        models = instantiate_models(dbf, species, data_phases, model=model, parameters=parameters)
                
        phase_recs = build_phase_records(dbf
        , species
        , data_phases
        , {v.N, v.P, v.T}
        , models
        , parameters=parameters
        , build_gradients=True
        , build_hessians=True)
        activity_conditions=data['conditions']
        potential_conds={key:val for key,val in activity_conditions.items() if key=="T" or key=="P"}
        potential_conds.setdefault('N', 1.0)
        comp_conds={key:val for key,val in activity_conditions.items() if key!="T" and key!="P"}
        
        ref_conditions=data['reference_state']['conditions']
        ref_compositions={key.split('_')[1]:val for key,val in ref_conditions.items() if key!="T" and key!="P"}
        Chem_Pot_ref_potential={key:value for key,value in ref_conditions.items() if not key.startswith('X_')}
        Chem_Pot_ref_potential.setdefault('N', 1.0)
        ref_potential=OrderedDict([(getattr(v, key), unpack_condition(Chem_Pot_ref_potential[key])) for key in sorted(Chem_Pot_ref_potential.keys()) if not key.startswith('X_')])
#        ref_defined_component=[key for key,val in ref_compositions.items() if val==1]
        
  
        
####The reason Jorge added len_components is in cade multiple different defined compositions are provided###
        len_components=list(set([len(comps) for comps in comp_conds.values()]))[0]
        lst_def_component_comps=[]
        if data_defined_components=='COMP':
            defined_components=data['defined_components']
            first_defined_component=[key for key in defined_components.keys()][0]
            reference_stoich=defined_components[first_defined_component]
            converted_ref_compositions=calculating_pseudo_line(data_comps,defined_components,ref_compositions)
            defined_unary_components=[key.split('_')[1] for key,val in converted_ref_compositions.items() if val>0.0]
            depend_unary_copmponents=sorted(defined_unary_components)[:-1]     
            ref_comp_conds = OrderedDict([(v.X(key[2:]), unpack_condition(converted_ref_compositions[key])) for key,val in sorted(converted_ref_compositions.items()) if key.startswith('X_') and val!=0.0 and key[2:] in depend_unary_copmponents])
            def_comp_species=sorted(unpack_components(dbf, defined_unary_components), key=str)
            def_comp_data_phases = filter_phases(dbf, def_comp_species, candidate_phases=phases)
            def_comp_models = instantiate_models(dbf, def_comp_species, def_comp_data_phases, model=model, parameters=parameters)
            ref_chem_pot_reg=ChemPotentialRegion(Chem_Pot_ref_potential,def_comp_species
            ,def_comp_data_phases,def_comp_models)
            
            chem_pot=ChemPotentialRegion(potential_conds,species,data_phases,models)
            def_comp_phase_recs = build_phase_records(dbf
            , def_comp_species
            , def_comp_data_phases
            , {v.N, v.P, v.T}
            , def_comp_models
            , parameters=parameters
            , build_gradients=True
            , build_hessians=True)
            for i in range(len_components):
                def_component_comps={}
                for comps,quant in comp_conds.items():
                    def_component_comps[comps[2:]]=quant[i]
                    def_comp_dat_file=calculating_pseudo_line(data_comps
                    ,defined_components,def_component_comps)
                lst_def_component_comps.append(def_comp_dat_file)
            ref_cond_dict = OrderedDict(**ref_potential, **ref_comp_conds)
        else:
            
            chem_pot=ChemPotentialRegion(potential_conds
            ,species,data_phases,models)

            
            defined_unary_components=[data_defined_components]
            ref_comp_species=sorted(unpack_components(dbf, defined_unary_components), key=str)
            lst_def_component_comps.append(comp_conds)
            ref_comp_data_phases = filter_phases(dbf, ref_comp_species, candidate_phases=phases)
            ref_cond_dict = OrderedDict(**ref_potential)
            ref_comp_models = instantiate_models(dbf, ref_comp_species, ref_comp_data_phases
            , model=model, parameters=parameters)
            def_comp_models = instantiate_models(dbf, ref_comp_species
            , ref_comp_data_phases, model=model, parameters=parameters)
            ref_chem_pot_reg=ChemPotentialRegion(Chem_Pot_ref_potential,ref_comp_species
            ,ref_comp_data_phases,def_comp_models)    
            reference_stoich=None
            def_comp_phase_recs = build_phase_records(dbf
            , ref_comp_species
            , ref_comp_data_phases
            , {v.N, v.P, v.T}
            , def_comp_models
            , parameters=parameters
            , build_gradients=True
            , build_hessians=True)            
            
        data_ref=data['reference']
        
        
        data_dict={
        'weight':data.get('weight', 1.0),
        'defined_components':data_defined_components,
        'phase_records':phase_recs,
        'reference_phase_records':def_comp_phase_recs,
        'ref_Chem_Potential':ref_chem_pot_reg,
        'Chem_Potential':chem_pot,
        'reference_stoich':reference_stoich,
        'dataset_reference':data_ref,
        'list_con_dict': lst_def_component_comps,
        'ref_cond_dict':ref_cond_dict,
        'samples':samples
        }
        
        activity_data.append(data_dict)
    return activity_data        
 
def calculating_pseudo_line(elemental_composition,defined_components,component_fractions):

#    pseudo_line=component_ratio(defined_components)    
    pseudo_line=defined_components
    dependent_comp=1-sum([i for i in component_fractions.values()])
    for key,value in defined_components.items():
        if key not in component_fractions:   
            component_fractions[key]=dependent_comp
    final_amount={}
    tot_moles=S.Zero
    for i in elemental_composition:
        fun_list=[]
        for comp,value in component_fractions.items():
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

def target_chempots_from_activity(component, target_activity, temperatures, reference_result):
    """
    Return an array of experimental chemical potentials for the component

    Parameters
    ----------
    component : str
        Name of the component
    target_activity : numpy.ndarray
        Array of experimental activities
    temperatures : numpy.ndarray
        Ravelled array of temperatures (of same size as ``exp_activity``).
    reference_result : xarray.Dataset
        Dataset of the equilibrium reference state. Should contain a singe point calculation.

    Returns
    -------
    numpy.ndarray
        Array of experimental chemical potentials
    """
    # acr_i = exp((mu_i - mu_i^{ref})/(RT))
    # so mu_i = R*T*ln(acr_i) + mu_i^{ref}
    if isinstance(component,str)==True:
        ref_chempot = reference_result["MU"].sel(component=component).values.flatten()
    else:
        ref_chempot=[sto*reference_result.MU.sel(component=spec).values.flatten()[0] for spec,sto in component[list(set(component.keys()))[0]].items()] 
#    print(v.R,temperatures,np.log(target_activity),sum(ref_chempot))
    return v.R * temperatures * np.log(target_activity) + sum(ref_chempot)

def calc_difference_activity(activity_data: Sequence[Dict[str, Any]],
                          parameters: np.ndarray,
                          approximate_equilibrium: Optional[bool] = False,
                          ) -> Tuple[np.ndarray, np.ndarray]:

    """
    Calculate differences between the expected and calculated values for a property

    Parameters
    ----------
    eqpropdata : EqPropData
        Data corresponding to equilibrium calculations for a single datasets.
    parameters : np.ndarray
        Array of parameters to fit. Must be sorted in the same symbol sorted
        order used to create the PhaseRecords.
    approximate_equilibrium : Optional[bool]
        Whether or not to use an approximate version of equilibrium that does
        not refine the solution and uses ``starting_point`` instead.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Pair of
        * differences between the calculated property and expected property
        * weights for this dataset

    """
    if approximate_equilibrium:
        _equilibrium = no_op_equilibrium_
    else:
        _equilibrium = equilibrium_

    for data in activity_data:
        activity_error=[]
        data_weights=[]
        weight=data['weight']
        defined_components=data['defined_components']
        dataset_ref=data['dataset_reference']
        list_component_dataset=data['list_con_dict']
        phase_records=data['phase_records']
        update_phase_record_parameters(phase_records, parameters)
        dataset_species=data['Chem_Potential'].species
        dataset_phases=data['Chem_Potential'].phases
        dataset_models=data['Chem_Potential'].models
        dataset_state_var=data['Chem_Potential'].potential_conds
        Temp=dataset_state_var['T']
        samples=data['samples']
        print('I am her enow',samples)
        samples=[v.R*Temp*np.log(i) for i in samples]
        
        ref_cond_dict=data['ref_cond_dict']        
        ref_phase_records=data['reference_phase_records']
        ref_state_var=data['ref_Chem_Potential'].potential_conds
        ref_species=data['ref_Chem_Potential'].species
        ref_phases=data['ref_Chem_Potential'].phases
        ref_models=data['ref_Chem_Potential'].models
        reference_stoichiometric=data['reference_stoich']
        ref_grid = calculate_(ref_species, ref_phases, ref_state_var
        , ref_models, ref_phase_records, pdens=50, fake_points=True)
        Ref_multi_eqdata = _equilibrium(ref_phase_records, ref_cond_dict, ref_grid)
        Ref_Chem_Potentials=Ref_multi_eqdata.MU.squeeze()
        Ref_Chem_components=Ref_multi_eqdata.coords['component']
        if defined_components=='COMP':
            Ref_Chem_Potential=sum([reference_stoichiometric[comp]*mu for comp,mu in zip(Ref_Chem_components,Ref_Chem_Potentials)])
        else:
            if Ref_Chem_Potentials.ndim==0:
                Ref_Chem_Potentials=[Ref_Chem_Potentials.tolist()]
            else:
                pass
            Ref_Chem_Potential=[mu for comp,mu in zip(Ref_Chem_components,Ref_Chem_Potentials) if comp==defined_components][0]
        grid=calculate_(dataset_species
        ,dataset_phases,dataset_state_var,dataset_models
        ,phase_records, pdens=50, fake_points=True)

        dataset_state_var=OrderedDict([(getattr(v, key), unpack_condition(dataset_state_var[key])) for key in sorted(dataset_state_var.keys())])           
        calculated_data=[]
        for cond in list_component_dataset:
            if len(cond)>1:
                dep_comp=[i for i in cond.keys()][:-1]
            else:
                dep_comp=cond
            comp_cond=OrderedDict([(v.X(key[2:]), unpack_condition(cond[key])) for key,val in sorted(cond.items()) if key.startswith('X_')
            and key in dep_comp])
            cond_dict = OrderedDict(**dataset_state_var, **comp_cond)
            multi_eqdata =_equilibrium(phase_records, 
            cond_dict, grid)
            Chem_Pot=multi_eqdata.MU.squeeze()

            if defined_components=='COMP':
                Chem_components=list(sorted([i for i in reference_stoichiometric.keys()]))
                Chem_Potential=[sum([reference_stoichiometric[comp]*mu 
                for comp,mu in zip(Chem_components,Chem_Pot)])]
            else:
                Chem_components=multi_eqdata.coords['component']    
                Chem_Potential=[mu for chem_pot in Chem_Pot for comp,mu in zip(Chem_components,chem_pot) if comp==defined_components]
            
            activity= [(mu - Ref_Chem_Potential) for mu in Chem_Potential]
            calculated_data.append(activity)   
            

        calculated_data = np.array(calculated_data, dtype=np.float_)
        samples=np.array(samples,dtype=np.float)
    
####CHECK THIS AGAIN FOR ARRAY SHAPE THAT WILL BE IMPORTANT####
#    assert calculated_data.shape == samples.shape, f"Calculated data shape {calculated_data.shape} does not match samples shape {samples.shape}"
#    assert calculated_data.shape == weight.shape, f"Calculated data shape {calculated_data.shape} does not match weights shape {weights.shape}"
##############################################################
        differences = calculated_data - samples
        output=calculated_data.flatten().tolist()
#        print('These are the differences my dude',differences)
        _log.debug('Output: %s differences: %s, weights: %s, reference: %s', output, differences, weight, dataset_ref)
    return differences, weight
        
# TODO: roll this function into ActivityResidual
def calculate_activity_residuals(dbf, comps, phases, datasets, parameters=None, phase_models=None, callables=None, data_weight=1.0) -> Tuple[List[float], List[float]]:
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

    if parameters is None:
        parameters = {}

    activity_datasets = datasets.search(
        (tinydb.where('output').test(lambda x: 'ACR' in x)) &
        (tinydb.where('components').test(lambda x: set(x).issubset(comps))))

    residuals = []
    weights = []
    
    for ds in activity_datasets:
        acr_def_component=ds['output'].split('_')[1]  # the component of interest
        data_comps = ds['components']
        def_com_ref_condition={}
        if acr_def_component!='COMP':
            # calculate the reference state equilibrium
            ref = ds['reference_state']
            ref_conditions = {_map_coord_to_variable(coord): val for coord, val in ref['conditions'].items()}
            ref_result = equilibrium(dbf, data_comps, ref['phases'], ref_conditions,
                                     model=phase_models, parameters=parameters,
                                     callables=callables)
        else:

            checking_comment=ds['comment']
            checking_comment=checking_comment.split('_')
            ele_stoich=[]
            ele=[]
            for i in checking_comment:

                if i not in ds['components']:
                    ele_stoich.append(float(i))
                else:
                    ele.append(i)

            def_com_ref_condition[v.N]=1
            def_com_ref_condition[v.T]=ds['conditions']['T']
            def_com_ref_condition[v.P]=ds['conditions']['P']
            defined_comp=ds['output'].split('_')
            del defined_comp[0]
            del defined_comp[0]
            end_member_comp=[i/sum(ele_stoich) for i in ele_stoich]
            for index in range(len(ele)-1):                
                def_com_ref_condition[v.X(ele[index])]=end_member_comp[index]
            ref_result = equilibrium(dbf, ele, ds['phases'], def_com_ref_condition,
                                     model=phase_models, parameters=parameters,
                                     callables=callables, calc_opts={'pdens': 500})
            ref_result_check=np.squeeze(ref_result.GM.values).tolist()

        # data_comps and data_phases ensures that we only do calculations on
        # the subsystem of the system defining the data.
        data_phases = filter_phases(dbf, unpack_components(dbf, data_comps), candidate_phases=phases)

        # calculate current chemical potentials
        # get the conditions
        conditions = {}
        # first make sure the conditions are paired
        # only get the compositions, P and T are special cased
        conds_list = [(cond, value) for cond, value in ds['conditions'].items() if cond not in ('P', 'T')]
        # ravel the conditions
        # we will ravel each composition individually, since they all must have the same shape
        dataset_computed_chempots = []
        dataset_weights = []
        for comp_name, comp_x in conds_list:
            P, T, X = ravel_conditions(ds['values'], ds['conditions']['P'], ds['conditions']['T'], comp_x)
            conditions[v.P] = P
            conditions[v.T] = T
            conditions[_map_coord_to_variable(comp_name)] = X
        # do the calculations
        # we cannot currently turn broadcasting off, so we have to do equilibrium one by one
        # invert the conditions dicts to make a list of condition dicts rather than a condition dict of lists
        # assume now that the ravelled conditions all have the same size
        conditions_list = [{c: conditions[c][i] for c in conditions.keys()} for i in range(len(conditions[v.T]))]
        trouble_shooting_reference_phase=ds['phases']
        acr_component={}
        if acr_def_component=='COMP':
            acr_component[ds['output'].split('_')[2]]={}
            for k,l in zip(ele,ele_stoich):
                acr_component[ds['output'].split('_')[2]][k]=l
        else:
            pass
#        def_comp_ele_chem_pot=new_components[defined_comp[0]]
        if acr_def_component!='COMP':
            for conds in conditions_list:
                sample_eq_res = equilibrium(dbf, data_comps, data_phases, conds,model=phase_models, parameters=parameters,
                            callables=callables)
                dataset_computed_chempots.append(sample_eq_res.MU.sel(component=acr_def_component).values.flatten()[0]) 
        else:
            for conds in conditions_list:
                sample_eq_res = equilibrium(dbf, data_comps, data_phases, conds,model=phase_models, parameters=parameters,
                            callables=callables, calc_opts={'pdens': 500})
                if np.isnan(sample_eq_res.GM)==True:
                    sample_eq_res = equilibrium(dbf, data_comps, trouble_shooting_reference_phase, conds,model=phase_models, parameters=parameters,
                            callables=callables, calc_opts={'pdens': 500})
                chem_pot_defined_comp=[sto*sample_eq_res.MU.sel(component=spec).values.flatten()[0] for spec,sto in zip(ele,ele_stoich)]
#            dataset_computed_chempots.append(float(sample_eq_res.MU.sel(component=acr_component).values.flatten()[0]))
            dataset_weights.append(std_dev / data_weight / ds.get("weight", 1.0))
            
            dataset_computed_chempots.append(sum(chem_pot_defined_comp))
        # calculate target chempots
             
        dataset_activities = np.array(ds['values']).flatten()
        if acr_def_component=='COMP': 
            dataset_target_chempots = target_chempots_from_activity(acr_component, dataset_activities, conditions[v.T], ref_result)
        else:
            dataset_target_chempots = target_chempots_from_activity(acr_def_component
                                                            , dataset_activities, conditions[v.T], ref_result)           # calculate the error
            
        dataset_residuals = (np.asarray(dataset_computed_chempots) - np.asarray(dataset_target_chempots, dtype=float)).tolist()
        _log.debug('Data: %s, chemical potential difference: %s, reference: %s', dataset_activities, dataset_residuals, ds["reference"])
        residuals.extend(dataset_residuals)
        weights.extend(dataset_weights)
        
    return residuals, weights


# TODO: roll this function into ActivityResidual

def calculate_activity_error(activity_data: Sequence[Dict[str, Any]],
                        parameters: np.ndarray = None,
                        data_weight: int = 1.0,
                        approximate_equilibrium: bool = False) -> float:
                        
    if len(activity_data) == 0:
        return 0.0
                        
    residuals, weights = calc_difference_activity(activity_data, parameters)
    likelihood = np.sum(norm(0, scale=weights).logpdf(residuals))
    if np.isnan(likelihood):
#        # TODO: revisit this case and evaluate whether it is resonable for NaN
        # to show up here. When this comment was written, the test
        # test_subsystem_activity_probability would trigger a NaN.
        return -np.inf
    return likelihood

                        
################JORGE IS EDITING THIS OUT 08-15-22########
#def calculate_activity_error(dbf, comps, phases, datasets, parameters=None, phase_models=None, callables=None, data_weight=1.0) -> float:
#    """
#    Return the sum of square error from activity data
#
#    Parameters
#    ----------
 #   dbf : pycalphad.Database
#        Database to consider
#    comps : list
#        List of active component names
#    phases : list
#        List of phases to consider
#    datasets : espei.utils.PickleableTinyDB
#        Datasets that contain single phase data
#    parameters : dict
#        Dictionary of symbols that will be overridden in pycalphad.equilibrium
#    phase_models : dict
#        Phase models to pass to pycalphad calculations
#    callables : dict
#        Callables to pass to pycalphad
#    data_weight : float
#        Weight for standard deviation of activity measurements, dimensionless.
#        Corresponds to the standard deviation of differences in chemical
#        potential in typical measurements of activity, in J/mol.
#
#    Returns
#    -------
#    float
#        A single float of the likelihood
#
#
#    """
#    residuals, weights = calculate_activity_residuals(dbf, comps, phases, datasets, parameters=None, phase_models=None, callables=None, data_weight=1.0)
#    likelihood = np.sum(norm(0, scale=weights).logpdf(residuals))
#    if np.isnan(likelihood):
        # TODO: revisit this case and evaluate whether it is resonable for NaN
        # to show up here. When this comment was written, the test
        # test_subsystem_activity_probability would trigger a NaN.
#        return -np.inf
#    return likelihood





# TODO: the __init__ method should pre-compute Model and PhaseRecord objects
#       similar to the other residual functions, which will be much more performant.
# TODO: it seems possible (likely?) that "global" callables that were used
#       previously could be incorrect if there are activity datasets with
#       different sets of active components. Usually models, callables, and
#       phase records are tied 1:1 with a set of components. For now, callables
#       will never be built, but this will almost certainly cause a performance
#       regression. Model will also not be pre-built so we can properly use
#       custom user models
class ActivityResidual(ResidualFunction):
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
            self.weight = weight.get("ACR", 1.0)
        else:
            self.weight = 1.0

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


#        self._activity_likelihood_kwargs = {
#            "dbf": database, "comps": comps, "phases": phases, "datasets": datasets,
#            "phase_models": model_dict,
#            "callables": None,
#            "data_weight": self.weight,
#        }


###JORGE IS ADDING LINES HERE
        parameters = dict(zip(symbols_to_fit, [0]*len(symbols_to_fit)))
        self.activity_data = get_activity_data(database, comps, phases, datasets, model_dict, parameters, data_weight_dict=self.weight)
############################################        

    def get_residuals(self, parameters: npt.ArrayLike) -> Tuple[List[float], List[float]]:
        residuals, weights =calc_difference_activity(self.activity_data, parameters)
        return residuals, weights

    def get_likelihood(self, parameters: npt.NDArray) -> float:
        likelihood = calculate_activity_error(self.activity_data,parameters=parameters, data_weight=self.weight)
        return likelihood


residual_function_registry.register(ActivityResidual)
