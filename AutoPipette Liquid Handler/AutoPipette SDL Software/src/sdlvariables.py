import pandas as pd
import numpy as np

def set_variables():
    '''
    This function initializes the variables and datasets that will be referenced and used  
    throughout the program and experiment.

    Returns:
        - data_bank: contains formulations and collected data, like reagent type concentrations and absorbance slope

        - dispense_mixtures: dataset used by the liquid handler, which contains volumes at every reagent 
        concentration and pH, as well as the coordinate positions of the wells

        - full_dispense_mixtures: stores all the values in dispense_mixtures for reference

        - desired_activity: the desired vMax you are looking for in your enzyme assay. We used 0.0005,
        but it is updated by the GUI

        - types_of_reagents: contains the unique types of reagents, essentially whatever is inputted on the
        "Name" row in the home tab of the GUI.  
        Format: ['Reagent 1', 'Reagent 2', ...]

        - reagent_names: contains a list of reagent identification names. Typically, it is in the form 
        '[Reagent Type] [Concentration] [pH (optional)]'  
        Format: ['[Reagent Type] [Concentration] [pH (optional)]', ...]

        - reagents_list: contains list of unique types of reagents, with buffers being distributed to different pH's.  
        Format: ['Reagent Type', 'Buffer Type pH1', ...]

        - reagents_list_w_conc: contains dictionary of unique types of reagents, with buffers being distributed to different pH's,  
        with corresponding concentrations  
        Format: {'Reagent Type': [Concentration], 'Buffer Type pH': [Concentration], ...}

        - discrete_variables and continues_variables: currently not being used, but would organize the types of reagents in
        whether they are treated as discrete or continuous. Discrete means you have set concentrations, continuous is just a full
        range of concentrations possible

        - buffer_ph: a dictionary of buffers with their corresponding pHs. It is in the format {'Buffer': [pH1, pH2, pH3]}

        - group: contains a dictionary of the organized groups, which reagents correspond to which groups.  
        Format: {'Group 1': ['Reagent 1', 'Reagent 2'], 'Group 2': ['Reagent 3', 'Reagent 4']}
        
        - reagent_coords: dataframe with coordinate and well information for all the reagents at different concentrations and pHs

        - reagent_dict: reagent directory which stores all reagent information. Primarily used for parameter bounds since there are
        distinct lists and dictionaries for the other properties

        - num_samples: The number of samples for the initial seed library
    '''
    global types_of_reagents, reagent_names,reagent_list, discrete_variables,continuous_variables,buffer_ph,group,data_bank,dispense_mixtures,reagent_dict,reagent_coords,desired_activity,full_dispense_mixtures,reagents_list_w_conc, num_samples 
    data_bank = [['{}{}'.format(chr(i + 65), j + 1),False] for j in range(12) for i in range(8)]
    data_bank = pd.DataFrame(data_bank, columns=['Well','Status'])
    data_bank.set_index('Well', inplace=True)

    dispense_mixtures = [['{}{}'.format(chr(i + 65), j + 1), 209 - 9 * i, 26 + 9 * j, False, False] for j in range(12) for i in range(8)]
    dispense_mixtures = pd.DataFrame(dispense_mixtures, columns=['Well', 'X', 'Y', 'Status','Current'])
    dispense_mixtures.set_index('Well', inplace=True)
    dispense_mixtures['Iteration'] = np.nan

    full_dispense_mixtures = []
    desired_activity = 0.0005
    types_of_reagents = []
    reagent_names = []
    reagent_list = []
    reagents_list_w_conc = {}
    discrete_variables = []
    continuous_variables = []
    buffer_ph = {}
    group = {}
    num_samples = []

    # Reagent coordinates for reference:
    # A1: (125, 111), B1: (144, 111), C1: (163, 111), D1: (182, 111)
    # A2: (125, 91), B2: (144, 91), C2: (163, 91), D2: (182, 91)
    # A3: (125, 72), B3: (144, 72), C3: (163, 72), D3: (182, 72)
    # A4: (125, 53), B4: (144, 53), C4: (163, 53), D4: (182, 53)
    # A5: (125, 34), B5: (144, 34), C5: (163, 34), D5: (182, 34)
    # A6: (125, 15), B6: (144, 15), C6: (163, 15), D6: (182, 15)
    # Printed 1: (104,70), Printed 2: (104,27), Printed 3: (104,0)

    reagent_coords = pd.DataFrame({
        'reagent': [f'reagent {i+1}' for i in range(23)],  # 23 reagent names
        'Well': ['B1','C1','D1', 
                 'A2','B2','C2','D2',
                 'A3','B3','C3','D3',
                 'A4','B4','C4','D4',
                 'A5','B5','C5','D5',
                 'A6','B6','C6','D6'],
        'X': [75,50,27,              # "A1" is kept for water
              4,144,163,182,
              125,144,163,182,
              125,144,163,182,
              125,144,163,182,
              125,144,163,182],
        'Y': [248.3,248.3,248.3,
              248.3,91,91,91,
              72,72,72,72,
              53,53,53,53,
              34,34,34,34,
              15,15,15,15]
    })

    reagent_coords.set_index('reagent', inplace=True)

    reagent_dict = pd.DataFrame({
        'reagent': [f'reagent {i+1}' for i in range(10)],
        'Type': ['Continuous'] * 10,
        'Group': ['No Group'] * 10,
        'Stock': [np.nan] * 10,
        'Lower Bounds': [np.nan] * 10,
        'Upper Bounds': [np.nan] * 10
    })
    reagent_dict.set_index('reagent', inplace=True)

    return types_of_reagents, reagent_names,reagent_list, discrete_variables,continuous_variables,buffer_ph,group,data_bank,dispense_mixtures,reagent_dict,reagent_coords,desired_activity,full_dispense_mixtures,reagents_list_w_conc

set_variables()