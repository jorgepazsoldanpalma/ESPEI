"""
Calculate error due to measured activities.
"""

import logging

import numpy as np
import tinydb
from pycalphad import equilibrium, variables as v
from pycalphad.plot.eqplot import _map_coord_to_variable
from pycalphad.core.utils import filter_phases, unpack_components, instantiate_models
from scipy.stats import norm
from sympy import exp, log, Abs, Add, And, Float, Mul, Piecewise, Pow, S
from espei.core_utils import ravel_conditions
from pycalphad.codegen.callables import build_phase_records

_log = logging.getLogger(__name__)

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
                    comp_pseu=v.X(i)
#                    comp_pseu=literal_eval(comp_pseu)
                    fun_list.append(value*new_value)
                    final_fun_list=sum(fun_list)
                    final_amount[i]=final_fun_list
                    tot_moles+=value*new_value
    
    for final_comp,final_value in final_amount.items():
        final_amount[final_comp]=final_value/tot_moles
    
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


def chempot_error(sample_chempots, target_chempots, std_dev=10.0):
    """
    Return the sum of square error from chemical potentials

    sample_chempots : numpy.ndarray
        Calculated chemical potentials
    target_activity : numpy.ndarray
        Chemical potentials to target
    std_dev : float
        Standard deviation of activity measurements in J/mol. Corresponds to the
        standard deviation of differences in chemical potential in typical
        measurements of activity.

    Returns
    -------
    float
        Error due to chemical potentials
    """
    # coerce the chemical potentials to float64s, fixes an issue where SymPy NaNs don't work
    return norm(loc=0, scale=std_dev).logpdf(np.array(target_chempots - sample_chempots, dtype=np.float64))


def calculate_activity_error(dbf, comps, phases, datasets, parameters=None, phase_models=None, callables=None, data_weight=1.0):
    """
    Return the sum of square error from activity data

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

    error = 0
    if len(activity_datasets) == 0:
        return error
    
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
        current_chempots = []
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
                current_chempots.append(sample_eq_res.MU.sel(component=acr_def_component).values.flatten()[0]) 
        else:
            for conds in conditions_list:
                sample_eq_res = equilibrium(dbf, data_comps, data_phases, conds,model=phase_models, parameters=parameters,
                            callables=callables, calc_opts={'pdens': 500})
                if np.isnan(sample_eq_res.GM)==True:
                    sample_eq_res = equilibrium(dbf, data_comps, trouble_shooting_reference_phase, conds,model=phase_models, parameters=parameters,
                            callables=callables, calc_opts={'pdens': 500})
#                acr_component=def_comp_ele_chem_pot
                chem_pot_defined_comp=[sto*sample_eq_res.MU.sel(component=spec).values.flatten()[0] for spec,sto in zip(ele,ele_stoich)]
            current_chempots.append(sum(chem_pot_defined_comp))
            current_chempots = np.array(current_chempots)
        # calculate target chempots
             
        samples = np.array(ds['values']).flatten()
        if acr_def_component=='COMP': 
            target_chempots = target_chempots_from_activity(acr_component, samples, conditions[v.T], ref_result)
        else:
            target_chempots = target_chempots_from_activity(acr_def_component
                                                            , samples, conditions[v.T], ref_result)           # calculate the error
        weight = ds.get('weight', 1.0)
        pe = chempot_error(current_chempots, target_chempots, std_dev=std_dev/data_weight/weight)
        error += np.sum(pe)

        _log.debug('Data: %s, chemical potential difference: %s, probability: %s, reference: %s', samples, current_chempots-target_chempots, pe, ds["reference"])

    # TODO: write a test for this
    if np.any(np.isnan(np.array([error], dtype=np.float64))):  # must coerce sympy.core.numbers.Float to float64
        return -np.inf
    return error
