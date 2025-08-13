import numpy as np
import pandas as pd
import time
from datetime import datetime
import pyvisa
import SDLVariables as var
import clr
import sys, os

# ------------------------------------------
# Load the RemoteControl DLL for the Integra VIAFLO pipette
# ------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(BASE_DIR, '..', 'resources', 'RemoteControl.dll')

clr.AddReference(filepath)
from RemoteControl import PythonAPI

# Create a pipette object to access commands from RemoteControl.dll
pipette = PythonAPI();
# Global flags and counters to keep track of hardware connections and tip usage
global chemyx_TF
global pipette_TF
chemyx_TF = False  # Flag to indicate if the Chemyx pump is connected
pipette_TF = False # Flag to indicate if the VIAFLO pipette is connected
tip_TF = False     # Flag to indicate if we currently have a tip attached
tip_count = 0      # Keep track of how many tips have been picked up
speed = 8
mixcycles = 3
calibrationfactor = 0.95
x = 0
y = 0
z = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tip_positions_path = os.path.join(BASE_DIR, '..', 'resources', 'Tip Positions.csv.csv')

tip_positions = pd.read_csv(tip_positions_path, index_col='name')

def filter_next_mixtures(data_bank):
    dispense_mixtures = data_bank[(data_bank['Current'] == True) & (data_bank['Status'] == False)]
    
    return dispense_mixtures

# Initialize ChemyX Fusion 200X Syringe Pump
def initialize_chemyx():
    global chemyx
    global rm
    rm = pyvisa.ResourceManager()
    chemyx = rm.open_resource('ASRL7::INSTR') #, timeout = 25000, write_termination = '\r\n', readtermination = '\r\n', baud_rate =19200, data_bits =8)

def chemyx_command(command):
    global chemyx
    chemyx.query(command)
    response = chemyx.read()
    print (response)

# --------------------------------------- Duet Board Connection ---------------------------
def duet_connect():
    """
    Establishes a pyvisa connection to a Duet controller (or similar device)
    for controlling X-Y-Z motion on a custom liquid handling robot.
    Homes the axes upon connection.
    """
    global duet
    global rm
    rm = pyvisa.ResourceManager()
    # Note: 'ASRL3::INSTR' must match the COM port for the Duet (or other motion controller)
    duet = rm.open_resource('ASRL4::INSTR', timeout=60000, write_termination='\r\n', 
                            baud_rate=9600, data_bits=8)
    duet_command("G28")

    # change the idle timeout to 6000
    duet_command('M84 S999999')

def duet_command(code):
    """
    Sends a command to the Duet (or other motion controller)
    and waits for M400 (moves complete) before reporting success.
    """
    global duet
    duet.query(code)
    duet.query('M400')  # Ensures the move completes
    print('move successful ' + code)

def quit_program():
    """
    A convenience function to return the robot to a safe position
    and then quit the Tkinter application.
    """
    # Move to X0, Y0, Z0 first, then move up to a safe Z=107 (bottom)
    pipette.PowerOff()
    duet_command("G28")
    duet_command("G1 Z107")
    duet_command("M18")  # Disable stepper motors

#----------------------------- Pipette -------------------------------

def pipette_connect():
    """
    Connects to the VIAFLO pipette on the specified COM port.
    Sets the pipette_TF flag to True upon successful connection.
    """
    global pipette_TF 
    pipette_TF = True
    pipette.Connect('COM7')  # Update 'COM6' to the appropriate port for your system

#----------------------------- Connect to devices -------------------------------
pipette_connect()
duet_connect()

#initialize_chemyx()

#----------------------------- Manual Tab Movement Control -------------------------------
def ad_left(step):
    global x
    if x - step >= 0:
        x = round(x - step,2)
        cmd = f'G1 X{x}'
        duet_command(cmd)

def ad_right(step):
    global x
    if x + step <= 350:
        x = round(x + step,2)
        cmd = f'G1 X{x}'
        duet_command(cmd)

def ad_forward(step):
    global y
    if y - step >= 0:
        y = round(y - step,2)
        cmd = f'G1 Y{y}'
        duet_command(cmd)

def ad_backward(step):
    global y
    if y + step <= 255:
        y = round(y + step,2)
        cmd = f'G1 Y{y}'
        duet_command(cmd)

def ad_up(step):
    global z
    if z - step >= 0:
        z = round(z - step,2)
        cmd = f'G1 Z{z}'
        duet_command(cmd)

def ad_down(step):
    global z
    if z + step <= 107:
        z = round(z + step,2)
        cmd = f'G1 Z{z}'
        duet_command(cmd)

def ad_move(x_val,y_val,z_val):
    cmd = f'G1 X{x_val} Y{y_val} Z{z_val}'
    duet_command(cmd)

def tip_pickup():
    """
    Moves to the next tip in the tip rack (using x_A1, y_A1 as reference),
    performs a down/up motion (Z10 -> Z44 -> Z10) to pick up a tip.
    Increments tip_count to track the next tip position.
    """
    global tip_count

    # Starting reference (A1) from the positions CSV
    x_A1 = tip_positions._get_value('Tips_A1', 'x')
    y_A1 = tip_positions._get_value('Tips_A1', 'y')

    # Calculate next tip’s XY location. Each row offset is 9 mm in Y, 
    # and each column offset is 9 mm in X.
    x = x_A1 - 9 * int((tip_count - tip_count % 8) / 8)
    y = y_A1 - 9 * (tip_count % 8)

    # Move above the tip, go down to Z35 and then further to Z44 (grab tip), then back to Z10
    cmd = 'G1 X' + str(x) + ' Y' + str(y) + ' Z0' + ' F18000'
    duet_command(cmd)
    cmd = 'G1 X' + str(x) + ' Y' + str(y) + ' Z105.1' + ' F18000'
    duet_command(cmd)
    cmd = 'G1 X' + str(x) + ' Y' + str(y) + ' Z0' + ' F18000'
    duet_command(cmd)

    tip_count += 1

def tip_discard():
    """
    Moves to a designated discard location, 
    performs a motion to drop the tip into a waste container,
    and then retracts back to a safe Z height.
    """
    # -7 on z to press the eject
    # Move from the current position to the disposal area
    cmd = 'G1 X286.2 Y236 Z0' + ' F18000'
    duet_command(cmd)
    cmd = 'G1 X286.2 Y236 Z79.7' + ' F18000'
    duet_command(cmd)
    cmd = 'G1 X286.2 Y252.5 Z79.7' + ' F18000'
    duet_command(cmd)

    # Additional small motions to shake off or drop the tip
    cmd = 'G1 Y252.5 Z73' 
    duet_command(cmd)
    cmd = 'G1 Y252.5 Z79.7'
    duet_command(cmd)
    cmd = 'G1 Y242'
    duet_command(cmd)
    cmd = 'G1 Y252.5'
    duet_command(cmd)
    cmd = 'G1 Y236'
    duet_command(cmd)
    cmd = 'G1 Z0' + ' F18000'
    duet_command(cmd)

def get_false_status_coords(data_bank, reservoir_coords):
    # Step 1: Filter the data_bank to find rows where Status is False
    false_status_rows = data_bank[data_bank['Status'] == False]
    
    # Step 2: Initialize a dictionary to store the coordinates
    coords = {}

    # Step 3: Loop through each row with Status as False
    for index, row in false_status_rows.iterrows():
        for col in data_bank.columns:
            if col.startswith('reservoir') and not pd.isna(row[col]):
                # Step 4: Extract the column name and corresponding coordinates
                x_coord = reservoir_coords.loc[col, 'X']
                y_coord = reservoir_coords.loc[col, 'Y']
                coords[index] = (x_coord, y_coord)
    
    return coords

# ----------------------------- Dispensing Procedure -------------------------------
'''
General Information:
ChemyX: 
    The syringe pump is controlled by sending serial commands, and uses the supplied python API
'''

def perform_dispensing(dispense_mixtures, reservoir_coords):
    global tip_TF
    start_idx = list(dispense_mixtures.columns).index("Status") + 1
    end_idx = list(dispense_mixtures.columns).index("Current")
    dynamic_columns = dispense_mixtures.columns[start_idx:end_idx]

    # Iterate over each column (component/ingredient)
    for column in dynamic_columns:
        reservoir = column  # Assume the column name matches the reservoir name
        if reservoir == 'Current':
            break
        x_reservoir_coord, y_reservoir_coord = reservoir_coords.loc[reservoir, ['X', 'Y']]
        column_volumes = dispense_mixtures[column].values
        drawvolume = [vol for vol in column_volumes if pd.notna(vol)]
        if sum(drawvolume) > 0:
            skip = False
        else:
            skip = True
        
        if not skip:
            # Iterate over each well to dispense the current component
            for index, row in dispense_mixtures.iterrows():
                xposition, yposition = dispense_mixtures.loc[index, ['X', 'Y']]  # Get xposition and yposition from data_bank using the index
                dispense_volume = row[column] * 10 # * 10 is for integra pipette format. it is uL * 10, so 1000 is 100 uL for integra
                if pd.notna(dispense_volume) and dispense_volume > 0.00001:
                    dispense_volume = int(dispense_volume)
                    if not tip_TF:
                        tip_pickup()
                        tip_TF = True
                    '''
                    if tip == 'Yes':
                        print('Change tip')
                        tip_discard()
                        tip_pickup()
                    '''
                    # Move to reservoir coordinate position
                    cmd = f'G1 X{x_reservoir_coord} Y{y_reservoir_coord} Z0' + ' F18000'
                    duet_command(cmd)

                    cmd = 'G1 Z80' + ' F7000'
                    duet_command(cmd)
                    
                    # Aspirate Volume from reagent reservoir
                    time.sleep(2)
                    print('Aspirate ' + str(dispense_volume) + ' ul from ' + column)
                    # Aspirate: PipetteAPI('Aspirate', speed, volume, mixcycles)
                    pipette.PipetteAPI('Aspirate', str(speed), str(dispense_volume), str(mixcycles))
                    time.sleep(5)

                    cmd = 'G1 Z0' + ' F7000'
                    duet_command(cmd)

                    cmd = f'G1 X {xposition} Y{yposition} Z0' + ' F18000'
                    duet_command(cmd)

                    cmd = 'G1 Z104' + ' F7000'
                    duet_command(cmd)

                    print('Dispense ' + str(dispense_volume) + ' ul to ' + column)
                    pipette.PipetteAPI('Dispense_NoBo', str(speed), str(dispense_volume), str(mixcycles))
                    time.sleep(5)
                    
                    '''
                    if mix != 0:
                        pipette.PipetteAPI('BlowIn', str(speed), str(volume), str(mixcycles))
                        time.sleep(1)
                        print('Mix ' + str(mix) + ' times')
                        pipette.PipetteAPI('Mix', str(speed), str(volume), str(mix))
                        time.sleep(mix * 3)
                    '''
                    cmd = 'G1 Z0' + ' F7000'
                    duet_command(cmd)
                    
                    pipette.PipetteAPI('BlowIn', str(speed), str(dispense_volume), str(mixcycles))

                    # After finishing all steps, discard tip if currently in use
                    if tip_TF:
                        tip_discard()
                        tip_TF = False




