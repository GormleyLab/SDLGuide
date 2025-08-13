import os
import time
import csv
from io import StringIO
from itertools import product
import itertools
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import shuffle

from scipy.stats import norm
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Local Modules
import src.sdlvariables as var
import src.guifunctions as gf

# Spectramax SDK
# Uncomment if using Spectramax
"""
import clr
clr.AddReference('C:\\Program Files\\Molecular Devices\\SoftMax Pro 7.2 Automation SDK\\SoftMaxPro.AutomationClient.dll')
from SoftMaxPro.AutomationClient import SMPAutomationClient

spectramax = SMPAutomationClient()
"""
# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sda_path = os.path.join(BASE_DIR, '..', 'data', 'absorbance.sda')
xml_path = os.path.join(BASE_DIR, '..', 'data', 'absorbance.xml')
combinations_path = os.path.join(BASE_DIR, '..', 'data', 'all_combinations.csv')

def spectramax_initialize():
    spectramax.Initialize()

def spectramax_opendrawer():
    spectramax.OpenDrawer()

def spectramax_closedrawer():
    spectramax.CloseDrawer()

def concentration_test(csv_path, chunk_size):
    '''
    Tests parameter space to ensure experiment will run smoothly.
    Ensures provided stock concentrations are able to generate formulations that
    can be synthesized without going out of bounds.
    '''
    results = []

    def process_row(row):
        formula = pd.Series(row._asdict())  # convert namedtuple to Series
        result = gf.streamlined_reagent_calculator(var.reagent_list, formula, 200)
        if result is None:
            return {"formula": formula.to_dict(), "result": None}
        return None

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        alternating_rows = chunk.iloc[::3]
        with ThreadPoolExecutor() as executor:
            chunk_results = list(executor.map(process_row, alternating_rows.itertuples(index=False)))
            results.extend(filter(None, chunk_results))  # remove None entries

    return results

def generate_combinations_and_save_to_csv(df, output_csv_path):
    '''
    Generate all reagent concentration combinations from the provided dataframe,
    including buffer pH ranges and grouped reagents, and save the results to a CSV file
    to save system RAM.

    Parameters:
    - df (pd.DataFrame): Input dataframe containing reagents, bounds, groups, and buffer info.
    - output_csv_path (str): Path to save the generated combinations as a CSV.
    '''

     # Initialize Buffer_ID column
    df["Buffer_ID"] = None
    buffer_rows = df[df["Group"].str.startswith("Buffer", na=False)]

    # Assign unique buffer IDs
    for idx, (index, _) in enumerate(buffer_rows.iterrows(), start=1):
        df.loc[index, "Buffer_ID"] = idx

    # Handle "No Group" reagents: independently varied
    no_group_df = df[df["Group"] == "No Group"]
    bounds = no_group_df[["Lower Bounds", "Upper Bounds"]]

    ranges = [
        np.linspace(row["Lower Bounds"], row["Upper Bounds"], num=50, dtype=np.float32)
        for _, row in bounds.iterrows()
    ]

    mesh = np.meshgrid(*ranges, indexing='ij')
    no_group_matrix = np.stack(mesh, axis=-1).reshape(-1, len(ranges))

    # Handle buffer molarity-pH combinations
    buffer_combinations = []

    if not buffer_rows.empty:
        for index, row in buffer_rows.iterrows():
            buffer_id = row["Buffer_ID"]
            if pd.isna(buffer_id):
                continue

            # Molarity range for buffer
            values = np.linspace(row["Lower Bounds"], row["Upper Bounds"], num=50, dtype=np.float32)
            ph_values = var.buffer_ph.get(index, [])
            if len(ph_values) == 0:
                continue

            # Create all molarity-pH combinations
            ph_array = np.array(ph_values, dtype=np.float32)
            mesh_mol, mesh_ph = np.meshgrid(values, ph_array, indexing="ij")
            mol_flat = mesh_mol.ravel()
            ph_flat = mesh_ph.ravel()
            buffer_id_array = np.full_like(mol_flat, buffer_id, dtype=np.float32)

            buffer_combinations.append(np.stack([mol_flat, buffer_id_array, ph_flat], axis=1))

    # Combine buffer combinations into single array
    all_buffer_combos = np.vstack(buffer_combinations) if buffer_combinations else None

    # Handle grouped reagents like "Group1", "Group2", etc.
    group_names = sorted(set(df["Group"].dropna().unique()) - {"No Group"})
    group_ranges = []
    group_labels = []

    for group in group_names:
        if group.startswith("Group"):
            group_df = df[df["Group"] == group]
            if not group_df.empty:
                lower = group_df["Lower Bounds"].min()
                upper = group_df["Upper Bounds"].max()
                group_ranges.append(np.linspace(lower, upper, num=10, dtype=np.float32))
                group_labels.append(f"{group} Molarity")

    # Create mesh of all group molarity combinations
    group_matrix = (
        np.stack(np.meshgrid(*group_ranges, indexing='ij'), axis=-1).reshape(-1, len(group_ranges))
        if group_ranges else None
    )

    # Determine CSV header columns
    columns = list(no_group_df.index)
    if all_buffer_combos is not None:
        columns += ["Molarity", "Buffer_ID", "pH"]
    if group_labels:
        columns += group_labels

    # Write to CSV
    with open(output_csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)

        base_matrix = no_group_matrix

        # If buffers are present, repeat no_group values across all buffer entries
        if all_buffer_combos is not None:
            base_matrix = np.hstack([
                np.tile(no_group_matrix, (all_buffer_combos.shape[0], 1)),
                np.repeat(all_buffer_combos, no_group_matrix.shape[0], axis=0)
            ])

        # If group values are present, repeat base_matrix across all group combinations
        if group_matrix is not None:
            combo_block = np.hstack([
                np.tile(base_matrix, (group_matrix.shape[0], 1)),
                np.repeat(group_matrix, base_matrix.shape[0], axis=0)
            ])
            writer.writerows(combo_block)
        else:
            writer.writerows(base_matrix)

    # Save final DataFrame for future reference
    var.reagent_dict = df
    print(f"Combinations saved to {output_csv_path}")

def extract_data(file_path, wells):
    '''
    Extracts absorbance data from an XML file for specified wells.

    Parameters:
    - file_path (str): Path to the XML file containing plate reader data.
    - wells (list of str): List of well names to extract (e.g., ['A1', 'B2', 'C3']).

    Returns:
    - df_pivoted (pd.DataFrame): A pivoted DataFrame where rows are time points,
                                 and columns are wells with corresponding raw data values.
    - time (pd.Series): A Series of time points (shared across all wells).
    '''

    # Remove any plate clone data from the XML to avoid duplicates or interference
    remove_plate_clones(file_path)

    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = []

    # Iterate over all wells in the XML
    for well in root.findall(".//Well"):
        name = well.get('Name')
        if name in wells:
            # Look for raw data and time data under this well
            raw_data_element = well.find('RawData')
            time_data_element = well.find('TimeData')

            if raw_data_element is not None and time_data_element is not None:
                # Extract and clean text data
                raw_data = raw_data_element.text.strip()
                time_data = time_data_element.text.strip()

                # Convert string data into lists of floats
                raw_data_list = list(map(float, raw_data.split()))
                time_data_list = list(map(float, time_data.split()))

                # Store the data in a structured format
                data.append({
                    'Well': name,
                    'RawData': raw_data_list,
                    'TimeData': time_data_list
                })

    if data:
        # Initialize lists to build a flattened DataFrame
        well_names_list = []
        raw_data_list = []
        time_data_list = []

        # Flatten data so each row corresponds to a single time point per well
        for entry in data:
            well = entry['Well']
            raw_data = entry['RawData']
            time_data = entry['TimeData']

            well_names_list.extend([well] * len(time_data))
            raw_data_list.extend(raw_data)
            time_data_list.extend(time_data)

        # Create long-form DataFrame: one row per time point per well
        df = pd.DataFrame({
            'Well': well_names_list,
            'RawData': raw_data_list,
            'TimeData': time_data_list
        }).drop_duplicates()  # Remove duplicate entries if any

        # Pivot to wide format: TimeData as index, one column per well
        df_pivoted = df.pivot(index='TimeData', columns='Well', values='RawData').reset_index()

        # Optionally convert TimeData to seconds (commented out)
        # df_pivoted['TimeData'] = pd.to_timedelta(df_pivoted['TimeData']).dt.total_seconds()

        # Extract TimeData separately and drop from the DataFrame
        time = df_pivoted['TimeData']
        df_pivoted = df_pivoted.drop(labels='TimeData', axis=1)

        return df_pivoted, time
    
def remove_plate_clones(file_path):
    """
    Removes the <PlateClones> section from an XML file, if it exists.

    This is useful for cleaning plate reader XML files by eliminating 
    duplicate or irrelevant sections that might interfere with data extraction.
    Some devices don't export XML files with ,PlateClones.
    Parameters:
    - file_path (str): Path to the XML file to be modified.
    """

    # Parse the XML file into a tree structure
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Iterate through the immediate children of the root element
    for child in root:
        # Look for the <PlateClones> tag
        if child.tag == "PlateClones":
            # Remove the <PlateClones> section from the XML
            root.remove(child)
            break  # Only remove the first occurrence (should only be one)

    # Save the modified XML back to the same file path
    tree.write(file_path)

def collect_data():
    """
    Controls the SpectraMax plate reader to collect absorbance data,
    saves the result as both SDA and XML formats, waits until the SDA file
    is saved, and then removes the SDA file. This is to detect when reading
    has completed.
    """

    # Select the plate section to read
    spectramax.SelectSection("Plate1")

    # Start the plate reading process
    global saveas_xml
    Results = spectramax.StartRead()
    print('Reading Plate')

    # Save raw results in SDA format (SpectraMax native)
    spectramax.SaveAs(sda_path)

    # Export results to XML format for further processing
    spectramax.ExportAs(xml_path, spectramax.ExportAsFormat.XML)

    # Wait until the SDA file exists before proceeding
    waitUnit = 1.0  # Check every 1 second
    taskCompleted = False
    while not taskCompleted:
        fileExists = os.path.exists(sda_path)
        if fileExists:
            taskCompleted = True
            break
        time.sleep(waitUnit)

    # Delete the SDA file after confirming it's been saved
    os.remove(sda_path)

def poly2(x, a, b):
    return a * x + b

def fit_line(absorption_data, time):
    """
    Fits a 2-parameter model (assumed linear) to absorption data for each well.
    Stores and merges the slope values into var.data_bank.
    
    Parameters:
        absorption_data (pd.DataFrame): Absorbance data with wells as columns.
        time (pd.Series or array): Time points corresponding to absorption readings.
    
    Returns:
        parameters_df (pd.DataFrame): DataFrame containing slope values for each well.
    """

    fitted_params = {}

    # Loop over each well in the data
    for well in absorption_data.columns:
        well_data = absorption_data[well].values  # Extract absorbance values

        # Fit a polynomial function (assumes poly2 is predefined as a linear or quadratic)
        params, _ = curve_fit(poly2, time, well_data)
        fitted_params[well] = params  # Store the fitted parameters for each well

    # Only keep the first coefficient (e.g., slope if it's linear)
    # Loop is unnecessary if you're only extracting `a, b`, so this may be simplified
    for well, params in fitted_params.items():
        a, b = params  # Extract params, though not used here directly again

    # Convert fitted parameters dictionary to DataFrame
    fitted_parameters_df = pd.DataFrame(fitted_params)

    # Transpose to get wells as rows
    transposed_parameters = fitted_parameters_df.transpose().reset_index()
    transposed_parameters = transposed_parameters.rename(columns={'index': 'Well'})

    # Drop the second coefficient column (e.g., intercept if linear) to keep only slope
    transposed_parameters = transposed_parameters.drop(labels=1, axis=1)
    parameters_df = transposed_parameters.set_index('Well')

    # Rename column for clarity
    parameters_df = parameters_df.rename(columns={0: 'Slope'})

    # Prepare to merge slope values into var.data_bank
    parameters_df = parameters_df.reset_index()
    var.data_bank = var.data_bank.reset_index()

    # Merge or update slope values into var.data_bank
    if 'Slope' not in var.data_bank.columns:
        # First time: add the new column
        var.data_bank = var.data_bank.merge(parameters_df[['Well', 'Slope']], on='Well', how='left')
    else:
        # Already exists: update existing values
        for well in parameters_df['Well']:
            if well in var.data_bank['Well'].values:
                var.data_bank.loc[var.data_bank['Well'] == well, 'Slope'] = parameters_df.loc[
                    parameters_df['Well'] == well, 'Slope'
                ].values[0]

    # Restore original index after merge/update
    var.data_bank.set_index('Well', inplace=True)

    return parameters_df


def expected_improvement(y_pred, y_std, best_y):

    # Calculate expected improvement
    z = (y_pred - best_y) / y_std
    ei = (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)

    return ei

def find_next(file_name, well_names, combination_file_path): 
    """
    Uses experimental results to train a Gaussian Process model and identify the 
    next promising formulations to test using Expected Improvement as the acquisition function.
    
    Parameters:
        file_name (str): Path to the XML file with current experimental results.
        well_names (list): List of wells that were used in the experiment.
        combination_file_path (str): Path to CSV containing all possible formulation combinations.

    Returns:
        ei_comp (pd.DataFrame): Top 6 selected formulations for next iteration.
        cv_results_r2 (float): Cross-validated R² score of the GP model.
    """

    # Extract data
    absorbance_data, time = extract_data(file_name, well_names)

    # Calculate vMax and store in var.data_bank
    fit_line(absorbance_data,time)

    # Prepare model output data
    # Set standard scaler to scale the data for model training
    scaler = StandardScaler()

    # Isolate the synthesized formulations
    compositions_unscaled = var.data_bank.loc[var.data_bank['Status'] == True]
    absorbance_slope = pd.DataFrame(compositions_unscaled['Slope'])
    compositions_unscaled = compositions_unscaled.drop(labels=['Slope','Status'],axis=1)

    # Scale and shuffle to reduce bias
    compositions = scaler.fit_transform(compositions_unscaled)
    compositions, absorbance_slope = shuffle(compositions, absorbance_slope, random_state=42)

    #Model generation
    kernel = DotProduct() + RBF(length_scale=1.0) + WhiteKernel(0.001, noise_level_bounds=(1e-7, 1))
    gp_model = GaussianProcessRegressor(kernel=kernel)
    gp_model.fit(compositions,absorbance_slope)
    cv_results_r2 = np.mean(cross_val_score(gp_model, compositions, absorbance_slope, cv=4, scoring = 'r2'))

    ei_records = []
    global_index_offset = 0  # keeps track of absolute row index
    # Process CSV in chunks
    for chunk in pd.read_csv(combination_file_path, chunksize=1000000):
        
        # Optionally skip every 3rd row (your previous code)
        alternating_rows = chunk.iloc[::3].copy()
        alternating_rows.reset_index(drop=True, inplace=True)

        # Predict from gp_model
        y_pred, y_std = gp_model.predict(alternating_rows.values, return_std=True)
        ei = expected_improvement(y_pred, y_std, var.desired_activity).flatten()

        # Store EI with global index and row data
        for i, score in enumerate(ei):
            ei_records.append({
                "global_index": global_index_offset + (i * 3),
                "ei": score
            })

        # Update offset
        global_index_offset += len(chunk)

    ei_df = pd.DataFrame(ei_records)

    ## Smooth and identify peaks in EI
    flattened_ei = ei_df['ei'].values.astype(float)

    # Light smoothing first
    window_size = 1
    smoothed_y = np.convolve(flattened_ei, np.ones(window_size) / window_size, mode='valid')

    # First round of peak detection
    peaks, _ = find_peaks(smoothed_y)

    smoothed_peaks = smoothed_y[peaks]

    # Pad to detect boundary peaks
    modified_smoothed_peaks = np.concatenate([[0], smoothed_peaks, [0]])

    # Second peak detection
    peaks1, _ = find_peaks(modified_smoothed_peaks)

    ## Select top 6 smoothed peaks
    max_ei_indices = []
    temp_peaks = modified_smoothed_peaks.copy()
    for _ in range(20):  # look through top 20 peaks first
        max_index = np.argmax(temp_peaks[peaks1])
        max_ei_indices.append(peaks1[max_index])
        temp_peaks[peaks1[max_index]] = -np.inf

    # Map these to original EI rows
    original_peak_indices = peaks[max_ei_indices]  # These are row indices in ei_df

    # Keep only global_index of selected rows
    top_global_indices = ei_df.iloc[original_peak_indices]['global_index'].tolist()

    ## Reload combinations from CSV using global_index
    selected_rows = []
    with open(combinations_path, "r") as f:
        header = next(f)
        for i, line in enumerate(f):
            if i in top_global_indices:
                selected_rows.append((i, line))

    # Build DataFrame of selected combinations
    ei_comp = pd.read_csv(StringIO(header + "".join([line for _, line in selected_rows])))
    ei_comp = shuffle(ei_comp, random_state=42).reset_index(drop=True)
    ei_comp = ei_comp.iloc[:6]

    return ei_comp,cv_results_r2

def visualize_output():
    # Define the kernel for Gaussian Process: DotProduct + RBF + WhiteKernel (for noise)
    kernel = DotProduct() + RBF(length_scale=1.0) + WhiteKernel(0.001, noise_level_bounds=(1e-5, 1))

    # Optional dictionary mapping iteration numbers to readable generation names
    model_names = {
        1: "Seed",
        2: "First Gen",
        3: "Second Gen",
        4: "Third Gen",
        5: "Fourth Gen",
        6: "Fifth Gen",
        7: "Sixth Gen",
        8: "Seventh Gen",
        9: "Eighth Gen",
        10: "Ninth Gen",
        11: "Tenth Gen",
    }
    
    # Initialize containers
    gp_models = []        # List to hold trained GP models
    cv_results = {}       # Dict to hold cross-validation results
    n_folds = 4           # Number of folds for cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=38)  # KFold splitter
    full_iterations = {}  # Dict to hold Vmax data per generation
    scaler = StandardScaler()  # Standard scaler for feature normalization

    # Loop over each unique iteration in the data bank
    for iteration in sorted(var.data_bank['Iteration'].dropna().unique()):
        # Filter subset up to and including current iteration with active status
        subset = var.data_bank[(var.data_bank['Iteration'] <= iteration) & (var.data_bank['Status'] == True)]
        if subset.empty:
            continue  # Skip if no valid data
        
        # Prepare training labels (Vmax) and features (compositions)
        absorbance_slope = subset['Slope'] * 1000  # Convert Slope to Vmax scale
        compositions = subset.drop(columns=['Slope', 'Status','Iteration'])

        # Shuffle the data
        compositions, absorbance_slope = shuffle(compositions, absorbance_slope, random_state=42)

        # Store current iteration's data for plotting boxplot later
        subset_iteration = var.data_bank[(var.data_bank['Iteration'] == iteration) & (var.data_bank['Status'] == True)]
        absorbance_slope_iteration = subset_iteration['Slope'] * 1000
        full_iterations[model_names.get(iteration, f'Iter {iteration}')] = absorbance_slope_iteration

        # Train the GPR model
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(compositions, absorbance_slope)
        gp_models.append((iteration, model))  # Save model for later use

        # Normalize features and scale target for cross-validation
        compositions1 = scaler.fit_transform(compositions)
        absorbance_slope1 = absorbance_slope / 1000

        # Initialize cross-validation metrics
        r2_scores = []
        mse_scores = []
        mae_scores = []

        # Cross-validation loop
        for train_index, test_index in kf.split(compositions1):
            X_train, X_test = compositions1[train_index], compositions1[test_index]
            y_train, y_test = absorbance_slope1.iloc[train_index], absorbance_slope1.iloc[test_index]

            # Train model on training fold
            gp_model = GaussianProcessRegressor(kernel=kernel)
            gp_model.fit(X_train, y_train)

            # Predict on test fold
            y_pred = gp_model.predict(X_test)

            # Calculate performance metrics
            r2_scores.append(r2_score(y_test, y_pred))
            mse_scores.append(-np.mean((y_test - y_pred) ** 2))   # Negative MSE (optional)
            mae_scores.append(-np.mean(np.abs(y_test - y_pred)))  # Negative MAE (optional)

        # Save mean CV metrics for the generation
        cv_results[model_names.get(iteration, f'Iter {iteration}')] = {
            'R2': np.mean(r2_scores),
            'MSE': np.mean(mse_scores),
            'MAE': np.mean(mae_scores)
        }

    # Convert results to DataFrame for easier plotting
    cv_df = pd.DataFrame.from_dict(cv_results, orient='index')

    # === First Plot: Predicted vs Measured ===
    num_plots = len(gp_models)
    fig, axes = plt.subplots(1, num_plots, figsize=(21, 5), sharex=True, sharey=True)

    for i, (iteration, gp_model) in enumerate(gp_models[:num_plots]):
        predicted_vmax, predicted_std = gp_model.predict(compositions, return_std=True)

        ax = axes[i]
        ax.scatter(absorbance_slope, predicted_vmax, s=80, color='#800080', edgecolors='black', alpha=0.9)
        ax.plot([0, 0.7], [0, 0.7], 'k--', linewidth=2)

        title = model_names.get(iteration, f'Iter {iteration}')
        ax.set_title(f'GPR {title}', fontsize=14, fontweight='bold')

        legend_text = f"$R^2$: {round(cv_results[title]['R2'], 2)}\nMAE: {round(cv_results[title]['MAE']*-100000, 3)} E-5"
        ax.text(0.05, 0.7, legend_text, fontsize=10, ha='left', va='top', backgroundcolor='white')

    # Add global labels
    fig.supxlabel('Measured VMax ($10^{-3}$)', fontsize=14, fontweight='bold')
    fig.supylabel('Predicted VMax ($10^{-3}$)', fontsize=14, fontweight='bold', x=0.01)
    plt.tight_layout()
    plt.savefig("Prediction_Capabilities_All_Models.svg", format='svg', bbox_inches='tight')
    plt.show()

    # === Second Plot: Box and Strip plots + R² score ===
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Prepare data for box and strip plots
    df_box = pd.DataFrame({key: pd.Series(value) for key, value in full_iterations.items()})
    df_strip = pd.DataFrame([(key, value) for key, values in full_iterations.items() for value in values],
                            columns=["Generation", "Slope"])

    # Set consistent color palette
    palette = sns.color_palette("Paired", n_colors=len(full_iterations))

    # Plot boxplot with median/whisker styling
    ax1.axhline(var.desired_activity * 1000, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label="Desired Activity")
    sns.boxplot(
        data=df_box,
        width=0.5,
        showcaps=True,
        palette=palette,
        medianprops={'color': 'black', 'linewidth': 1.5},
        whiskerprops={'color': 'black', 'linewidth': 1.2},
        ax=ax1
    )

    # Add strip plot (individual data points) over boxplot
    sns.stripplot(
        data=df_strip,
        x="Generation",
        y="Slope",
        jitter=0.25,
        palette=palette,
        alpha=0.7,
        size=6,
        edgecolor="black",
        linewidth=1.2,
        ax=ax1
    )

    # Axis labeling
    ax1.set_ylabel(r'$\mathbf{V}_{\mathbf{max}}$ ($10^{-3}$)', fontsize=12, fontweight='bold')
    ax1.set_title(r'$\mathbf{V}_{\mathbf{max}}$ Gen-on-Gen Improvement and R² Scores', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', frameon=False, fontsize=11)

    # R² line plot in lower subplot
    sns.lineplot(data=cv_df["R2"], marker='o', linestyle='-', color='b', ax=ax2)
    ax2.set_ylabel("R² Score", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Generation", fontsize=12, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=11)

    # Final layout adjustment and save
    plt.tight_layout()
    plt.savefig("GenonGen_R2_Separated.svg", format='svg', bbox_inches='tight')
    plt.show()

    # === SHAP Analysis for Final Model ===
    compositions_shap = var.data_bank.loc[var.data_bank['Status'] == True].drop(labels=['Slope','Status','Iteration'], axis=1)

    # Use final model for SHAP
    iteration, gp_shap = gp_models[-1]

    # KernelExplainer is general-purpose and slow
    explainer = shap.KernelExplainer(gp_shap.predict, compositions_shap)

    # Compute SHAP values
    shap_values = explainer.shap_values(compositions_shap)

    # Summary plot to visualize feature contributions
    shap.summary_plot(shap_values, compositions_shap, show=False)
    plt.title("Shapely Additive Explanations (SHAP) Analysis")
    plt.show()


def select_first_open_well(data_bank): 
    '''
    Find an open well in the order of A1,B1, etc
    '''
    for well_data in data_bank:
        if not well_data[3]:  # Check if the well is not marked as used
            return well_data  # Return the well data if it's not used
    return None  # Return None if all wells are used

def process_dispense_mixtures():
    # Identify rows where Current is True
    true_rows = var.dispense_mixtures[var.dispense_mixtures["Current"]]

    # add these rows to full_dispense_mixtures
    var.full_dispense_mixtures = pd.concat([var.full_dispense_mixtures, true_rows], ignore_index=False)

    # Remove these rows from var.dispense_mixtures
    var.dispense_mixtures = var.dispense_mixtures[~var.dispense_mixtures["Current"]]
