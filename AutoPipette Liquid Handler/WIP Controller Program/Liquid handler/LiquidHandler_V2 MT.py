import os, time, sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import pyvisa   # PyVISA allows control of instruments (e.g., Chemyx pump, Duet) over serial
import serial   # The serial module, typically for COM port communications
import clr      # pythonnet: used to load .NET assemblies (the RemoteControl DLL)

# ------------------------------------------
# Load the RemoteControl DLL for the Integra VIAFLO pipette
# ------------------------------------------
filepath = r"D:\Plexymer\Liquid Handling Robots\Liquid Handling Bot\Liquid handler\Liquid handler\RemoteControl.dll"
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

def pipette_connect():
    """
    Connects to the VIAFLO pipette on the specified COM port.
    Sets the pipette_TF flag to True upon successful connection.
    """
    global pipette_TF 
    pipette_TF = True
    pipette.Connect('COM4')  # Update 'COM6' to the appropriate port for your system

def chemyx_connect():
    """
    Establishes a serial connection to the Chemyx syringe pump using pyvisa,
    configures pump parameters (units, diameter, flow rate), and sets 
    the chemyx_TF flag to True.
    """
    global chemyx
    global rx
    global chemyx_TF
    chemyx_TF = True

    rx = pyvisa.ResourceManager()
    # The line below opens the Chemyx pump resource over an ASRL (COM) port.
    # Update 'ASRL4::INSTR' to the correct COM port number for your system.
    chemyx = rx.open_resource('ASRL8::INSTR', timeout=25000, write_termination='\r\n', 
                              read_termination='\r\n', baud_rate=38400, data_bits=8)
    
    # Set up initial pump parameters
    chemyx_command('set units mL/min')
    chemyx_command('set diameter 8.66')  # Syringe diameter in mm
    chemyx_command('set rate 8')        # Flow rate in mL/min

def chemyx_command(command):
    """
    Sends a command to the Chemyx pump and waits for the pump
    to finish its current movement (polling 'pump status').
    """
    global chemyx
    chemyx.query(command)   # Send command to the pump
    response = chemyx.read()
    print(response)

    # Continuously check if the pump is busy (status '1')
    chemyx.query('pump status')
    response = chemyx.read()
    while response == '1':
        time.sleep(0.5)
        chemyx.query('pump status')
        response = chemyx.read()

def chemyx_dispense(volume):
    """
    Sets the pump to dispense the specified volume (in mL) and starts it.
    """
    command = 'set volume ' + str(volume)
    chemyx_command(command)
    chemyx_command('start')

def chemyx_aspirate(volume):
    """
    Sets the pump to aspirate the specified volume (in mL) by sending
    a negative volume, then starts it.
    """
    asp_volume = volume * -1
    command = 'set volume ' + str(asp_volume)
    chemyx_command(command)
    chemyx_command('start')

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
    duet = rm.open_resource('ASRL7::INSTR', timeout=60000, write_termination='\r\n', 
                            baud_rate=9600, data_bits=8)

    # M564: Ensure no movement constraints, G1 H2 Z5: raise Z slightly, G28: home all axes
    duet_command("M564")
    duet_command("G1 H2 Z5")
    duet_command("G28")

    # change the idle timeout to 6000
    duet_command('M84 S6000')

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
    global root
    # Move to X0, Y0, Z0 first, then move up to a safe Z=44
    duet_command("G1 X0 Y0 Z0")
    duet_command("G1 X0 Y0 Z44")
    duet_command("M18")  # Disable stepper motors
    root.quit()


def run_pipette():
    """
    Reads steps from 'pipette_method.csv' (Volume, Mix, Source, Destination),
    performs aspirate/dispense actions using the VIAFLO pipette, and 
    optionally changes tips between steps.
    """
    global tip_TF

    # Ensure pipette is connected
    if not pipette_TF:
        pipette_connect()

    # Default pipette speed and mix cycles
    speed = 8
    mixcycles = 3

    # Load the pipette method from CSV
    method = pd.read_csv('pipette_method.csv', index_col='Step')
    print('List of steps')
    print(method)
    print('\nRunning Method')

    for ind in method.index:
        volume = method['Volume ul'][ind] * 10  # The code uses *10—check that this matches your pipette's internal logic
        mix = method['Mix'][ind]
        tip = method['New Tip'][ind]

        # If we don't currently have a tip on, or if the method says "Yes" to pick a new tip, pick one up
        if not tip_TF:
            tip_pickup()
            tip_TF = True

        if tip == 'Yes':
            print('Change tip')
            tip_discard()
            tip_pickup()

        # Move to the source location
        name = method['Source'][ind]
        x = str(positions._get_value(name, 'x'))
        y = str(positions._get_value(name, 'y'))
        z = str(positions._get_value(name, 'z'))
        cmd = 'G1 X' + x + ' Y' + y + ' Z0'
        duet_command(cmd)

        # Adding a pre-blowout step to ensure no liquid is left in the tip
        #pipette.PipetteAPI('BlowOut', str(speed), str(10), str(mixcycles))

        cmd = 'G1 X' + x + ' Y' + y + ' Z' + z
        duet_command(cmd)

        time.sleep(2)
        print('Aspirate ' + str(volume) + ' ul from ' + name)
        # Aspirate: PipetteAPI('Aspirate', speed, volume, mixcycles)
        pipette.PipetteAPI('Aspirate', str(speed), str(volume), str(mixcycles))
        time.sleep(5)

        cmd = 'G1 X' + x + ' Y' + y + ' Z0'
        duet_command(cmd)

        # Move to the destination location
        name = method['Destination'][ind]
        x = str(positions._get_value(name, 'x'))
        y = str(positions._get_value(name, 'y'))
        z = str(positions._get_value(name, 'z'))
        cmd = 'G1 X' + x + ' Y' + y + ' Z0'
        duet_command(cmd)
        cmd = 'G1 X' + x + ' Y' + y + ' Z' + z
        duet_command(cmd)
        
        time.sleep(2)
        print('Dispense ' + str(volume) + ' ul to ' + name)
        pipette.PipetteAPI('Dispense_NoBo', str(speed), str(volume), str(mixcycles))
        time.sleep(5)

        # Optional mix steps
        if mix != 0:
            pipette.PipetteAPI('BlowIn', str(speed), str(volume), str(mixcycles))
            time.sleep(1)
            print('Mix ' + str(mix) + ' times')
            pipette.PipetteAPI('Mix', str(speed), str(volume), str(mix))
            time.sleep(mix * 3)
            
        cmd = 'G1 X' + x + ' Y' + y + ' Z0'
        duet_command(cmd)
        pipette.PipetteAPI('BlowIn', str(speed), str(volume), str(mixcycles))

    # After finishing all steps, discard tip if currently in use
    if tip_TF:
        tip_discard()
        tip_TF = False

    duet_command("G1 X0 Y0 Z0")
    print('Method complete')

def run_pump():
    """
    Reads steps from 'pump_method.csv' (Volume, Mix, Source, Destination),
    performs aspirate/dispense actions using the Chemyx syringe pump (needle),
    and optionally performs mixing by aspirating/dispensing partial volumes.
    """
    # Ensure the Chemyx pump is connected
    chemyx_connect()

    method = pd.read_csv('pump_method.csv', index_col='Step')
    print('List of steps')
    print(method)
    print('\nRunning Method')

    for ind in method.index:
        volume = method['Volume ml'][ind]
        mix = method['Mix'][ind]

        # Move to the source location
        name = method['Source'][ind]
        x = str(positions._get_value(name, 'x'))
        y = str(positions._get_value(name, 'y'))
        z = str(positions._get_value(name, 'z'))
        cmd = 'G1 X' + x + ' Y' + y + ' Z0'
        duet_command(cmd)
        cmd = 'G1 X' + x + ' Y' + y + ' Z' + z
        duet_command(cmd)

        time.sleep(2)
        print('Aspirate ' + str(volume) + ' ml from ' + name)
        chemyx_aspirate(volume)
        cmd = 'G1 X' + x + ' Y' + y + ' Z0'
        duet_command(cmd)

        # Move to the destination location
        name = method['Destination'][ind]
        x = str(positions._get_value(name, 'x'))
        y = str(positions._get_value(name, 'y'))
        z = str(positions._get_value(name, 'z'))
        cmd = 'G1 X' + x + ' Y' + y + ' Z0'
        duet_command(cmd)
        cmd = 'G1 X' + x + ' Y' + y + ' Z' + z
        duet_command(cmd)
        
        time.sleep(2)
        print('Dispense ' + str(volume) + ' ml to ' + name)
        chemyx_dispense(volume)

        # Optional mixing by repeatedly aspirating/dispensing partial volume
        if mix != 0:
            print('Mix ' + str(mix) + ' times')
            for i in range(mix):
                chemyx_aspirate(0.5)
                chemyx_dispense(0.5)
            
        cmd = 'G1 X' + x + ' Y' + y + ' Z0'
        duet_command(cmd)

    duet_command("G1 X0 Y0 Z0")
    print('Method complete')

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
    x = x_A1 + 9 * int((tip_count - tip_count % 8) / 8)
    y = y_A1 - 9 * (tip_count % 8)

    # Move above the tip, go down to Z10 and then further to Z44 (grab tip), then back to Z10
    cmd = 'G1 X' + str(x) + ' Y' + str(y) + ' Z10'
    duet_command(cmd)
    cmd = 'G1 X' + str(x) + ' Y' + str(y) + ' Z44'
    duet_command(cmd)
    cmd = 'G1 X' + str(x) + ' Y' + str(y) + ' Z10'
    duet_command(cmd)

    tip_count += 1

def tip_discard():
    """
    Moves to a designated discard location, 
    performs a motion to drop the tip into a waste container,
    and then retracts back to a safe Z height.
    """
    # Move from the current position to the disposal area
    cmd = 'G1 X265 Y25 Z0'
    duet_command(cmd)
    cmd = 'G1 X265 Y25 Z36'
    duet_command(cmd)

    # Additional small motions to shake off or drop the tip
    cmd = 'G1 X265 Y0 Z36'
    duet_command(cmd)
    cmd = 'G1 X265 Y0 Z31'
    duet_command(cmd)
    cmd = 'G1 X265 Y0 Z36'
    duet_command(cmd)
    cmd = 'G1 X265 Y25 Z36'
    duet_command(cmd)
    cmd = 'G1 X265 Y25 Z0'
    duet_command(cmd)

def main():
    """
    The main Tkinter GUI function:
    - Creates a small GUI window for moving the robot to X, Y, Z positions
    - Teaching new positions (saves to positions.csv)
    - Running either the pipette or pump scripts
    - Quitting the program
    """
    global root
    
    def moveto():
        """
        Reads X, Y, Z from the GUI and sends a G-code move to the Duet.
        """
        x = xEntry.get()
        y = yEntry.get()
        z = zEntry.get()
        cmd = 'G1 X' + x + ' Y' + y + ' Z' + z
        print('move to ' + cmd)
        duet_command(cmd)

    def teach():
        """
        Captures the current entries (x, y, z) and names it as 'name' in 
        teach_positions (copy of the main positions DataFrame), then saves to CSV.
        """
        name = nameEntry.get()
        x = xEntry.get()
        y = yEntry.get()
        z = zEntry.get()
        teach_positions.loc[name] = [x, y, z]
        teach_positions.to_csv('positions.csv')
        print("taught " + name + " at x = " + x + " y = " + y + " z = " + z)
        print(positions)

    def goto():
        """
        Goes to a previously taught position (name) by retrieving it from
        the positions DataFrame, and sends the coordinates to the Duet.
        """
        name = nameEntry.get()
        print('goto ' + name)
        x = str(positions._get_value(name, 'x'))
        y = str(positions._get_value(name, 'y'))
        z = str(positions._get_value(name, 'z'))
        cmd = 'G1 X' + x + ' Y' + y + ' Z' + z
        duet_command(cmd)
        print('move successful')

    # Build Tkinter window
    root = tk.Tk()
    root.title('Plexymer Liquid Handler')
    root.geometry('500x200')

    # X axis setup
    xLabel = tk.Label(root, text='X')
    xEntry = tk.Entry(root)
    xEntry.insert(0, '100')  # default example
    xLabelComment = tk.Label(root, text='0..335 mm')

    # Y axis setup
    yLabel = tk.Label(root, text='Y')
    yEntry = tk.Entry(root)
    yEntry.insert(0, '100')  # default example
    yLabelComment = tk.Label(root, text='0..250 mm')

    # Z axis setup
    zLabel = tk.Label(root, text='Z')
    zEntry = tk.Entry(root)
    zEntry.insert(0, '0')    # default example
    zLabelComment = tk.Label(root, text='0..45 mm')

    # Name for teaching/goto
    nameLabel = tk.Label(root, text='name')
    nameEntry = tk.Entry(root)
    nameLabelComment = tk.Label(root, text='ex. pos1')

    # Buttons for control
    sendCommandButton = tk.Button(root, text='Move', command=moveto)
    sendCommandButton.grid(row=4, column=0)

    TeachButton = tk.Button(root, text='Teach', command=teach)
    TeachButton.grid(row=4, column=1)

    GoToButton = tk.Button(root, text='GoTo', command=goto)
    GoToButton.grid(row=4, column=2)

    RunPipButton = tk.Button(root, text='Run Pipette', command=run_pipette)
    RunPipButton.grid(row=4, column=3)

    RunPumpButton = tk.Button(root, text='Run Pump', command=run_pump)
    RunPumpButton.grid(row=4, column=4)

    QuitButton = tk.Button(root, text='Quit', command=quit_program)
    QuitButton.grid(row=4, column=5)

    # Layout
    xLabel.grid(row=0, column=0)
    xEntry.grid(row=0, column=1)
    xLabelComment.grid(row=0, column=2)

    yLabel.grid(row=1, column=0)
    yEntry.grid(row=1, column=1)
    yLabelComment.grid(row=1, column=2)

    zLabel.grid(row=2, column=0)
    zEntry.grid(row=2, column=1)
    zLabelComment.grid(row=2, column=2)

    nameLabel.grid(row=3, column=0)
    nameEntry.grid(row=3, column=1)
    nameLabelComment.grid(row=3, column=2)

    root.mainloop()

# ------------------------------------------
# Main script logic begins here
# ------------------------------------------

# NOTE:
# The positions.csv file must at least contain:
#   'A1'      -> A1 position of tip rack
#   'P_rack'  -> Pipette reference position on rack
#   'N_rack'  -> Needle reference position on rack
tip_positions = pd.read_csv('Tip Positions.csv', index_col='name')
positions = pd.read_csv('Computed Positions.csv', index_col='name')


# Connect to the Duet motion controller and home the axes
duet_connect()

# Start the Tkinter main loop to provide a GUI for user interaction
main()

# Close the VISA resource manager
rm.close()

# If Chemyx pump was used, close that connection
if chemyx_TF == True:
    rx.close()
    print('close chemyx')

# If pipette was used, home pipette and then power it off
if pipette_TF == True:
    pipette.PipetteAPI('HomePipette', str(8), str(1000), str(3))
    time.sleep(5)
    pipette.PowerOff()
    print('poweroff pipette')

print('Program finished')
