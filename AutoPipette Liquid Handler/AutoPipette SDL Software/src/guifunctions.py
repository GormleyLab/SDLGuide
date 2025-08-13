import csv
import itertools
import time
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
import os


import numpy as np
import pandas as pd
from pyDOE import lhs

import src.dataprocessing as dp
import src.sdlvariables as var
import src.liquidhandler as lh
import src.sdlgui as sg

pd.set_option('display.max_columns', 30)

# --------------------- Control GUI / Collect inputs from GUI ---------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sample_composition_path = os.path.join(BASE_DIR, '..', 'data', 'sample_compositions.csv')
dispense_mixtures_path = os.path.join(BASE_DIR, '..', 'data', 'dispense_mixtures.csv')
combinations_path = os.path.join(BASE_DIR, '..', 'data', 'all_combinations.csv')
data_bank_path = os.path.join(BASE_DIR, '..', 'data', 'data_bank.csv')
absorbance_path = os.path.join(BASE_DIR, '..', 'data', 'absorbance.xml')

def populate_table_from_csv(entries, filename):
    '''
    Populate the table in the home page from a csv file to prepare for automated 
    dispensing using the liquid handler
    '''
    try:
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip the first row
            for row_index, row_data in enumerate(csv_reader):
                for col_index, value in enumerate(row_data, start=0):  # Start from 0 to include all columns
                    if row_index < 15 and col_index < 10:  # Make sure to stay within table bounds
                        entries[row_index][col_index].delete(0, tk.END)
                        entries[row_index][col_index].insert(0, value)
    except Exception:
        messagebox.showerror("Error", "Invalid CSV File")

def upload_csv(entries):
    '''
    Select and upload the csv used to populate the table in the home page to prepare 
    for automated dispensing using the liquid handler
    '''
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filename:
        populate_table_from_csv(entries, filename)

def clear_entries(entries):
    '''
    Clears the custom entries inputted in the home page
    '''
    for row_entries in entries:
        for entry in row_entries:
            entry.delete(0, tk.END)

def streamlined_reagent_calculator(reagent_list, formula, target_volume):
    """
    Calculates the required volumes of reagents to achieve a specified target volume for a formulation.
    
    Parameters:
    - reagent_list (list of str): A list of reagent names to consider in the calculation.
    - formula (dict): A dictionary mapping reagent names (str) to their required concentration (float) in the final mixture.
    - target_volume (int or float): The desired total volume of the final mixture.
    
    Returns:
    - dict or None: A dictionary mapping each reagent (including 'Water') to its 'Stock Concentration' and required 'Volume'
      for the formulation if a suitable combination is found. Returns None if no combination meets the criteria.
    """
    
    # Filter the input reagent list to include only those present in the formula
    reagent_list = [reagent for reagent in reagent_list if reagent in formula]

    # Generate all possible combinations of specified reagent concentrations using their bounds from reagent_dict
    reagent_combos = list(itertools.product(*[var.reagents_list_w_conc[reagent] for reagent in reagent_list]))
    
    # Iterate through each combination of reagent concentrations
    for combo in reagent_combos:
        # Calculate the volume for each reagent based on its proportion in the formula and the combination's concentration
        reagent_volumes = [target_volume * formula[reagent] / combo[i] for i, reagent in enumerate(reagent_list)]
        total_volume = sum(reagent_volumes)  # Calculate the total volume of all reagents
        
        # Check if the total volume is within the target and individual volumes are within practical limits
        if total_volume <= target_volume and all((volume == 0 or 10 <= volume < 300) for volume in reagent_volumes):
            water_volume = target_volume - total_volume
            
            # Construct and return a dictionary with each reagent's 'Stock Concentration' and 'Volume'
            output_dic = {f'{reagent_list[i]} {combo[i]}': reagent_volumes[i] for i in range(len(reagent_list))}
            output_dic['water'] = water_volume  # Add 'Water' details
            
            return output_dic  # Return the dictionary if a suitable combination is found
    
    # If no suitable combination is found, print a message and return None
    print("No combination of reagent concentrations is less than the target volume")
    # Print the formulation details
    
    return None

def retrieve_initial_mixtures_data(entries, reagents, stock_concentrations,ph_inputs,group_vars,bounds_widgets,variable_type_widgets):
    '''
    Retrieves all the data from the GUI inputs
    This includes:
        -   Manual formulation inputs using the array of input text boxes. This is converted directly into instructions
            for the liquid handler to dispense
        -   Reagent Information
            -   Group Type (No Group, Buffer [x], or Group [x]) - This indicates how the reagent will be processed and
                how the program will organize the experiment.
                -   No Group: Will always include this reagent in every formulation
                -   Group [x]: Will select a reagent from the list in each group. Meaning, if there is reagent 1,
                    reagent 2, and reagent 3 in Group 1, it will randomly select one of these 3 possibilities and add
                    it to a certain percentage of formulations as inputted
                -   Buffer: Each reagent that is considered a buffer, or has an alterring pH must be organized as a buffer
                    to be processed correctly. Program will vary the buffer type, ph, and concentration for every formulation
        -   Reagent concentration parameter space (This is inputted under the Enzyme Assay Tab)
        -   Reagent Type: Whether it is continuous or discrete. This was an idea that was abandoned, so every reagent type
            is considered as a continuous concentration.
    '''
    initial_data = []
    stock_data = []

    # Extract the entries from input fields
    for row in entries:
        row_data = []
        for entry in row:
            row_data.append(entry.get())
        initial_data.append(row_data)
    # Iterate through each reagent and its corresponding stock concentrations
    for i, (reagent, stock_entry, ph_entry) in enumerate(zip(reagents, stock_concentrations,ph_inputs)):
        reagent_name = reagent.get().strip() or f'reagent {i + 1}'
        stock_values = stock_entry.get().strip().split(',')  # Assuming stock concentrations are comma-separated
        ph_values = ph_entry.get().strip().split(',')

        # Create new reagent for each stock concentration
        for stock_value in stock_values:
            stock_value = stock_value.strip()
            if stock_value:
                # Create new reagent for each ph
                for ph_value in ph_values:
                    ph_value = ph_value.strip()
                    if ph_value:
                        ph_value = float(ph_value) 
                        if ph_value.is_integer():
                            ph_value = int(ph_value)
                        column_name = f'{reagent_name} {ph_value} {float(stock_value)}'
                        reagent_name_ph = f'{reagent_name} {ph_value}'
                        var.reagent_names.append(column_name)
                        var.types_of_reagents.append(reagent_name)
                        if reagent_name_ph in var.reagents_list_w_conc:
                        # Append the stock value to the list
                            var.reagents_list_w_conc[reagent_name_ph].append(float(stock_value))
                        else:
                            # Initialize with a list containing the first stock value
                            var.reagents_list_w_conc[reagent_name_ph] = [float(stock_value)]
                        if reagent_name_ph not in var.reagent_list:
                            var.reagent_list.append(reagent_name_ph)
                    else:
                        column_name = f'{reagent_name} {float(stock_value)}'
                        var.reagent_names.append(column_name)
                        var.types_of_reagents.append(reagent_name)
                        if reagent_name in var.reagents_list_w_conc:
                        # Append the stock value to the list
                            var.reagents_list_w_conc[reagent_name].append(float(stock_value))
                        else:
                            # Initialize with a list containing the first stock value
                            var.reagents_list_w_conc[reagent_name] = [float(stock_value)]

                        if reagent_name not in var.reagent_list:
                        # Append the stock value to the list
                            var.reagent_list.append(reagent_name)
            else:
                stock_data.append(np.nan)

    # Retrieve number of samples
    var.num_samples = int(sg.get_num_samples_value())

    # Remove duplicate reagent names from appending step
    var.types_of_reagents = list(dict.fromkeys(var.types_of_reagents))

    # Update the reagent_coords with reagent names and stock concentrations, and set name (reagent type + concentration + ph) as index
    var.reagent_coords = var.reagent_coords.iloc[:len(var.reagent_names)].copy()
    var.reagent_coords.index = var.reagent_names

    # Update the reagent_dict with the new reagent names and stock concentrations, and set name (just reagent type) as index
    var.reagent_dict = var.reagent_dict.iloc[:len(var.types_of_reagents)].copy()
    var.reagent_dict.index = var.types_of_reagents

    # Add water reagent coordinate to top of reagent_dict
    new_row = pd.DataFrame({'X': [99], 'Y': [248.3], 'Well': ['A1']}, index=['water'])
    var.reagent_coords = pd.concat([new_row, var.reagent_coords])

    # Iterates through group dropdowns and assigns group in reagent directory. 
    # Also creates list of buffers with corresponding pH, and list of groups with corresponding reagents
    for i, group_var in enumerate(group_vars):
        if i < len(var.types_of_reagents):
            selected_value = group_var.get()  # Get the selected value from the dropdown
            type_of_reagent = var.types_of_reagents[i]
            var.reagent_dict.at[type_of_reagent, 'Group'] = selected_value
            # Look for buffer groups, and add to buffer_ph with corresponding pH values
            if selected_value.lower().startswith('buffer'):
                var.buffer_ph[type_of_reagent] = [float(x) for x in ph_inputs[i].get().split(',')]
            
            if selected_value.lower().startswith('group'):
                if selected_value not in var.group:
                    var.group[selected_value] = []
                
                var.group[selected_value].append(type_of_reagent)
                

    # Extract lower and upper bounds from entries and update the reagent_dict DataFrame
    for i, reagent_name in enumerate(var.types_of_reagents):
        # Retrieve the bounds entry associated with the current reagent
        bounds_entry = bounds_widgets.get(i)
        
        if bounds_entry:
            bound_str = bounds_entry.get()
            try:
                # Extract lower and upper bounds
                lower_bound, upper_bound = map(float, bound_str.strip("()").split(","))
            except ValueError:
                # Handle invalid input
                lower_bound, upper_bound = np.nan, np.nan
            
            if reagent_name in var.reagent_dict.index:
                var.reagent_dict.at[reagent_name, 'Lower Bounds'] = lower_bound
                var.reagent_dict.at[reagent_name, 'Upper Bounds'] = upper_bound
            else:
                print(f"Warning: {reagent_name} not found in reagent_dict.")
        else:
            print(f"Warning: Bounds entry for index {i} not found.")
        
        variable_type = variable_type_widgets.get(i)
        if variable_type:
            selected_type = variable_type.get()
            # Update the type in var.reagent_dict with the current name from UI
            if reagent_name in var.reagent_dict.index:
                var.reagent_dict.at[reagent_name, 'Type'] = selected_type
            else:
                print(f"Warning: {reagent_name} not found in var.reagent_dict.")
        else:
            print(f"Warning: Combobox for index {i} not found.")

    # Filter and clean up the dispense mixtures
    for reagent in var.reagent_names:
        var.dispense_mixtures.insert(var.dispense_mixtures.columns.get_loc('Current'), reagent, np.nan)
    var.dispense_mixtures.insert(var.dispense_mixtures.columns.get_loc('Current'), 'water', np.nan)         ###fix this in a bit ---------------------------------------------



# --------------------- Format inputs into usable dataframe form ---------------------

def organize_mixtures():
    '''
    Constructs a list of formulation components based on reagent group classifications.
    
    This list will be used as the formula when generating dispense mixtures

    - "No Group" reagents: Treated as continuous variables and included individually.
    - "Buffer" reagents: Represented as a single molarity variable. Buffer type and pH are assigned separately.
    - "Group X" reagents: Each group is assigned a molarity variable and a corresponding reagent type variable 
      to track which reagent from the group is selected. The actual assignment occurs during sample generation.

    Returns:
        formulation (list): List of formulation component names to be used in sample generation.
    '''
    # Initialize formulation list
    formulation = []
    
    # Add all "No Group" reagents (treated as continuous variables)
    no_group_reagents = var.reagent_dict[var.reagent_dict['Group'].str.startswith('No', na=False)].index.tolist()
    if no_group_reagents:
        formulation.extend(no_group_reagents)
    
    # If buffers present, add molarity for concentration of buffer
    if var.buffer_ph:
        formulation.append('Molarity')
    
    # If user-defined groups present:
    if var.group:
        
        # For each group, add Molarity and Reagent Type
        group_names = var.reagent_dict['Group'].dropna().unique()
        group_names = [g for g in group_names if isinstance(g, str) and g.startswith('Group')]

        for group_name in group_names:
            formulation.append(f'{group_name} Molarity')         # Add Group X Molarity to the formulation
            formulation.append(f'{group_name} Reagent Type')     # Add Group X Reagent Type to the formulation
            
    return formulation

def generate_dispense_mixtures(data, formulation):
    '''
    Generates liquid handling instructions based on a sample design matrix and a formulation definition.

    For each sample (row in `data`), it calculates the volume of each reagent required to reach the desired
    target concentration and volume. It supports individual reagents, buffer systems (with pH variants), and
    grouped reagents defined by molarity and type.

    Inputs:
        - data (DataFrame): Design matrix containing reagent names, molarities, buffer selections, and pH values.
        - formulation (list): List of reagent-related columns used to construct each sample's formulation.

    Returns:
        Creates a dataframe in the format:
        Well    X   Y   Status  [Reagent 1 Conc. 1] [Reagent 1 Conc. 2] [Reagent 2 Conc. 1] Current

        Where Well is the well position, X and Y are the well coordinates, Status is whether it has been
        dispensed, current is which wells will get dispensed, and then between Status and Current, are all the
        reagents, where each reagent at each concentration, at each pH get's its own column.
    '''
    # Dictionary to store computed volumes for each sample row
    formulation_volumes = {}

    for index, row in data.iterrows():
        formula = {}

        # Add individual 'No group' reagents
        for key in formulation:
            if key in row and all(exclude not in key for exclude in ['Molarity', 'Group']):
                formula[key] = row[key]

        # Add buffer and pH (if applicable)
        if var.buffer_ph:
            buffer = row.get('Buffer', '')
            pH = row.get('pH', None)
            if buffer in var.buffer_ph and pH in var.buffer_ph[buffer]:
                pH_str = str(int(pH)) if isinstance(pH, (int, float)) and pH.is_integer() else str(pH)
                formula[f'{buffer} {pH_str}'] = row.get('Molarity', 0)

        # Add user-defined group reagent information (if applicable)
        if var.group:
            for group_name, reagent_list in var.group.items():
                reagent_key = f'{group_name}'
                molarity_key = f'{group_name} Molarity'

                if reagent_key in row and molarity_key in row:
                    chosen_reagent = row[reagent_key]
                    molarity_value = row[molarity_key]

                    if pd.notna(chosen_reagent) and pd.notna(molarity_value) and chosen_reagent in reagent_list:
                        formula[chosen_reagent] = molarity_value

        # Final mixture target volume in µL
        target_volume = 200

        # Remove any zero or missing concentrations before calculating volumes
        formula = {k: v for k, v in formula.items() if v not in [0, 0.0, None, '']}

        # Input formula as the reagent and concentration, and output a formulation with how much volume of
        # corresponding reagent at each concentration/ph
        result = streamlined_reagent_calculator(var.reagent_list, formula, target_volume)
        formulation_volumes[index] = result

    # Populate dispense_mixtures table
    for idx, volume_dict in formulation_volumes.items():
        if not isinstance(volume_dict, dict):
            continue
        for reagent, volume in volume_dict.items():
            if reagent in var.dispense_mixtures.columns:
                if isinstance(volume, (int, float)):
                    var.dispense_mixtures.loc[var.dispense_mixtures.index[idx], reagent] = volume / 1000

def create_lhs_samples(sample_number, bounds):
    '''
    Generates a set of parameter samples using Latin Hypercube Sampling (LHS),
    scaled according to the specified lower and upper bounds.

    Parameters:
        sample_number (int): Number of samples to generate.
        bounds (ndarray): A 2D NumPy array of shape (n_parameters, 2),
                          where each row contains [lower_bound, upper_bound] 
                          for a given parameter.

    Returns:
        scaled_samples (ndarray): A NumPy array of shape (sample_number, n_parameters),
                                  containing LHS-sampled and scaled values.
    '''

    lhs_sample = lhs(len(bounds), samples=sample_number, criterion='center')
    scaled_samples = lhs_sample * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    return scaled_samples

def seed_library_generator():
    '''
    Generates a seed library of sample formulations using Latin Hypercube Sampling (LHS)
    based on user-defined reagent types and formulation bounds.

    Function workflow:
    - Collects formulation bounds from reagents that are not assigned to any group.
    - If buffer options are specified, includes buffer molarity bounds and randomly assigns
      buffer types and pH values to samples.
    - If user-defined groups exist, calculates group-specific molarity bounds and randomly
      assigns group reagents to a subset of the samples.
    - Uses LHS to generate diverse and efficient sampling of parameter combinations.

    Returns:
        - scaled_samples_df (DataFrame): A DataFrame of generated sample compositions by LHS, 
          including any randomly assigned buffer and group reagents.
        - formulation (DataFrame): The organized mixture information from the input reagents.
    '''

    formulation = organize_mixtures()

    # Extract 'No group' bounds
    no_group_bounds = var.reagent_dict[var.reagent_dict['Group'].str.startswith('No', na=False)][['Lower Bounds', 'Upper Bounds']]
    
    # Start with 'No group' bounds
    all_bounds = no_group_bounds.copy()

    # Extract Buffer group bounds
    buffer_bounds = var.reagent_dict[var.reagent_dict['Group'].str.startswith('Buffer', na=False)][['Lower Bounds', 'Upper Bounds']]
    if var.buffer_ph:
        if not buffer_bounds.empty:
            molarity_bounds = pd.DataFrame({
                'Lower Bounds': [buffer_bounds['Lower Bounds'].min()],
                'Upper Bounds': [buffer_bounds['Upper Bounds'].max()]
            }, index=['Molarity'])
        else:
            molarity_bounds = pd.DataFrame(columns=['Lower Bounds', 'Upper Bounds'], index=['Molarity'])

        # Add buffer bounds to no group bounds
        all_bounds = pd.concat([all_bounds, molarity_bounds])

    # Initialize user-defined group bounds list
    group_bounds_list = []

    # Extract user-defined group bounds
    if var.group:
        group_names = var.reagent_dict['Group'].dropna().unique()
        group_names = [g for g in group_names if isinstance(g, str) and g.startswith('Group')]

        for group_name in group_names:
            group_df = var.reagent_dict[var.reagent_dict['Group'] == group_name][['Lower Bounds', 'Upper Bounds']]
            if not group_df.empty:
                molarity_lower = group_df['Lower Bounds'].min()
                molarity_upper = group_df['Upper Bounds'].max()
                molarity_bounds = pd.DataFrame({
                    'Lower Bounds': [molarity_lower],
                    'Upper Bounds': [molarity_upper]
                }, index=[f'{group_name} Molarity'])

                group_bounds_list.append(group_df)  # Save for later reagent selection
            else:
                molarity_bounds = pd.DataFrame(columns=['Lower Bounds', 'Upper Bounds'], index=[f'{group_name} Molarity'])

            # Add user-defined bounds to no group/buffer bounds
            all_bounds = pd.concat([all_bounds, molarity_bounds])

    # Generate LHS samples scaled to reagent bounds
    scaled_samples = create_lhs_samples(var.num_samples, all_bounds.to_numpy())
    scaled_samples_df = pd.DataFrame(scaled_samples, columns=all_bounds.index)

    # Randomly assign buffer IDs and pH (if applicable)
    if not buffer_bounds.empty:
        buffer_samples = np.random.choice(buffer_bounds.index.tolist(), size=var.num_samples)
        scaled_samples_df['Buffer'] = buffer_samples
        for idx, buffer_type in scaled_samples_df['Buffer'].items():
            if buffer_type in var.buffer_ph:
                random_ph = np.random.choice(var.buffer_ph[buffer_type])
                scaled_samples_df.at[idx, 'pH'] = random_ph

    # Randomly assign user-defined group reagents to a subset of samples(if applicable)
    full_group_df = pd.concat(group_bounds_list) if group_bounds_list else pd.DataFrame()
    if not full_group_df.empty:
        assigned_samples_count = var.num_samples // 2   # 50% of samples will contain groups
        none_samples_count = var.num_samples - assigned_samples_count

        assigned_samples = np.random.choice(full_group_df.index.tolist(), size=assigned_samples_count)
        none_samples = [None] * none_samples_count
        group_samples = np.array(list(assigned_samples) + none_samples)
        np.random.shuffle(group_samples)
        
        for group_name in group_names:
            group_reagents = var.reagent_dict[var.reagent_dict['Group'] == group_name].index.tolist()

            # Decide which rows will get a reagent for this group
            assigned_indices = np.random.choice(var.num_samples, size=var.num_samples // 2, replace=False)
            group_column = [None] * var.num_samples

            for idx in assigned_indices:
                chosen_reagent = np.random.choice(group_reagents)
                group_column[idx] = chosen_reagent

            scaled_samples_df[group_name] = group_column

    return scaled_samples_df,formulation
# --------------------- Experiment specific section ---------------------

def enzyme_assay_run():
    '''
    Perform's the steps necessary for automated and self-driven experiments
        1. Creates seed library using the seed_library_generator function
        2. Converts the seed library generated, which is scaled_samples_df, and
        with the formulation list, that is sent to generate_dispense_mixtures to
        convert the formulation concentrations to volume instructions for the 
        liquid handler.
        3. Data cleanup is performed, and the DBTL loop begins.

            4. BUILD: Dispensing is performed
            5. TEST: Spectramax is closed and it starts reading
            6. LEARN: find_next function is called, which trains a model, and finds
            next set of formulations to to test
            7. DESIGN: Converts the expected improvement formulations to instructions
            the liquid handler can use to synthesize the samples

            Steps 4-7 are looped until sufficient r^2 is achieved, or a sample 
            that matches the desired activity is found

    '''
    # Reset everything
    var.set_variables()

    # Retrieve inputs from GUI
    retrieve_initial_mixtures_data(sg.entries, sg.reagents,sg.stock_concentrations,sg.ph_inputs,sg.group_vars,sg.bounds_widgets,sg.variable_type_widgets)
    
    # Generate seed library
    scaled_samples_df, formulation = seed_library_generator()

    # Generate dispensing instructions using seed library
    generate_dispense_mixtures(scaled_samples_df, formulation)

    # Mark samples to be synthesized
    var.dispense_mixtures.loc[var.dispense_mixtures.index[:var.num_samples], 'Current'] = True

    # Move 'water' column after 'Status'
    cols = list(var.dispense_mixtures.columns)
    if 'water' in cols and 'Status' in cols:
        cols.insert(cols.index('Status') + 1, cols.pop(cols.index('water')))
        var.dispense_mixtures = var.dispense_mixtures[cols]

    # Save outputs
    scaled_samples_df.to_csv(sample_composition_path)
    var.dispense_mixtures.to_csv(dispense_mixtures_path)

    # Prepare 'var.data_bank' for compositions
    for component in scaled_samples_df.columns:
        if component not in var.data_bank.columns:
            var.data_bank[component] = np.nan  # Initialize new columns with NaN values
        var.data_bank[component] = var.data_bank[component].astype(object) # Ensure it can handle all variable types
    ei_comp = scaled_samples_df

    dp.generate_combinations_and_save_to_csv(var.reagent_dict, combinations_path)
    
    #result = dp.concentration_test("all_combinations.csv",1000000)
    #print(result)

    user_continue = sg.ask_to_continue_with_dataframe(var.reagent_coords)
    if user_continue:
        
        #dp.spectramax_initialize() # Uncomment if integrating with Spectramax
        r2_score = 0
        prev_r2 = -np.inf
        threshold_improvement = 0.03
        activity_tolerance = 0.03
        iteration_count = 1

        # DBTL LOOP CODE
        while True:
            # Calculate relative change in r2
            rel_change = abs(r2_score - prev_r2) / max(abs(prev_r2), 1e-8)

            if r2_score >= 0.8 or rel_change < threshold_improvement:
                break
             # Break if any slope is within tolerance of desired activity
            elif any(abs(s - var.desired_activity) < activity_tolerance for s in var.data_bank['Slope']):
                break

            synthesized_wells = []

            # Store compositions in 'var.data_bank'
            for _, row in ei_comp.iterrows():
                # Find the first open well
                open_well = select_first_open_well(var.data_bank)
                synthesized_wells.extend([open_well])
                if open_well is None:
                    print("No open wells available.")
                    break

                # Populate the open well with the row data from `scaled_samples_df`
                for component, value in row.items():
                    if component in var.data_bank.columns:  # Check if component exists in `data_bank`
                        var.data_bank.at[open_well, component] = value  # Use .at to set value by index

                # Mark the open well as used
                var.data_bank.at[open_well, 'Status'] = True

                # Set iteration to 1
                var.data_bank.at[open_well, 'Iteration'] = iteration_count

            var.data_bank.to_csv(data_bank_path)
            iteration_count = iteration_count + 1

            # Synthesize samples
            #dp.spectramax_opendrawer() # Uncomment if integrating with Spectramax

            lh.perform_dispensing(var.dispense_mixtures, var.reagent_coords)

            # Uncomment if looping with spectramax, and remove the break statement
            break
            '''
            # Collect data using spectramax
            dp.spectramax_closedrawer() # Uncomment if integrating with Spectramax
            dp.collect_data() # Uncomment if integrating with Spectramax

            ei_comp,r2_score = dp.find_next(absorbance_path, synthesized_wells, combinations_path)
            print(r2_score)
            print(ei_comp)
            var.full_dispense_mixtures = pd.DataFrame(columns=var.dispense_mixtures.columns)
            dp.process_dispense_mixtures()
            generate_dispense_mixtures(ei_comp,formulation)
            # Assign True value to 'Current' column to indicate which samples must be synthesized
            var.dispense_mixtures.loc[var.dispense_mixtures.index[:6], 'Current'] = True
            cols = list(var.dispense_mixtures.columns)
            '''
        dp.visualize_output()

    else:
        var.set_variables()

def generate_custom_input_formulations():
    print('WIP')

def performdispensing():
    lh.perform_dispensing(var.dispense_mixtures, var.reagent_coords)

def select_first_open_well(data_bank):
    for idx, row in data_bank.iterrows():  # Iterate through each row
        if not row['Status']:  # Check if 'Status' is False
            return idx  # Return the index of the open well
    return None  # Return None if no open well is found