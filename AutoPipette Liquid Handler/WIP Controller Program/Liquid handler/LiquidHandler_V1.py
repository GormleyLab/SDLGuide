import os, time, sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import pyvisa
import serial
import clr

filepath = r"C:\Users\adamj\Box\_Programming\IntegraAPI\RemoteControl\RemoteControl\bin\Debug\RemoteControl.dll"

clr.AddReference(filepath)

from RemoteControl import PythonAPI

pipette = PythonAPI();

global chemyx_TF
global pipette_TF
chemyx_TF = False
pipette_TF = False
tip_TF = False
tip_count = 0

def pipette_connect():
    global pipette_TF 
    pipette_TF = True
    pipette.Connect('COM6')

def chemyx_connect():
    global chemyx
    global rx
    global chemyx_TF
    chemyx_TF = True

    rx = pyvisa.ResourceManager()
    chemyx = rx.open_resource('ASRL4::INSTR', timeout = 25000, write_termination = '\r\n', 
        read_termination = '\r\n', baud_rate = 38400, data_bits = 8)
    #ASRL# where the number should be the com port #
    chemyx_command('set units mL/min')
    chemyx_command('set diameter 8.66')
    chemyx_command('set rate 8')


def chemyx_command(command):
    global chemyx
    chemyx.query(command)
    response = chemyx.read()
    print(response)
    chemyx.query('pump status')
    response = chemyx.read()
    while response == '1':
        time.sleep(0.5)
        chemyx.query('pump status')
        response = chemyx.read()

def chemyx_dispense(volume):
    command = 'set volume ' + str(volume)
    chemyx_command(command)
    chemyx_command('start')

def chemyx_aspirate(volume):
    asp_volume = volume * -1
    command = 'set volume ' + str(asp_volume)
    chemyx_command(command)
    chemyx_command('start')

def duet_connect():

    global duet
    global rm
    rm = pyvisa.ResourceManager()
    duet = rm.open_resource('ASRL3::INSTR', timeout = 60000, write_termination = '\r\n', 
    baud_rate = 9600, data_bits = 8)

    duet_command("M564")
    duet_command("G1 H2 Z5")
    duet_command("G28")

def duet_command(code):
    global duet
    duet.query(code)
    duet.query('M400')
    print('move successful ' + code)

def quit_program():
    global root
    duet_command("G1 X0 Y0 Z0")
    duet_command("G1 X0 Y0 Z44")
    duet_command("M18")
    root.quit()

def rack_positions():
    global positions

    #Needle teaching coordinates
    #locating origin to top left corner of rack 
    #which is 4mm in x and y from teach position
    Nx = positions._get_value('N_rack', 'x')-4
    Ny = positions._get_value('N_rack', 'y')+4

    #Pipette teaching coordinates
    #locating origin to top left corner of rack 
    #which is 4mm in x and y from teach position
    Px = positions._get_value('P_rack', 'x')-4
    Py = positions._get_value('P_rack', 'y')+4

    #Large Vials Needle
    for i in range(7):
        name_row1 = 'N_Lgvial' + str(i)
        name_row2 = 'N_Lgvial' + str(i+8)
        x = Nx + 6 + 24.3*i
        y_row1 = Ny - 6
        y_row2 = Ny - 31
        z = 44
        positions.loc[name_row1] = [x,y_row1,z]
        positions.loc[name_row2] = [x,y_row2,z]

    #HPLC Vials and small tubes Needle
    for i in range(10):
        name_row1 = 'N_HPLCvial' + str(i)
        name_row2 = 'N_HPLCvial' + str(i+12)
        name_row3 = 'N_Smtubel' + str(i)
        x = Nx + 4 + 16.4*i
        y_row1 = Ny - 54
        y_row2 = Ny - 69
        y_row3 = Ny - 84.5
        z = 44
        positions.loc[name_row1] = [x,y_row1,z]
        positions.loc[name_row2] = [x,y_row2,z]
        positions.loc[name_row3] = [x,y_row3,z]

    #Large Vials Pipette
    for i in range(7):
        name_row1 = 'P_Lgvial' + str(i)
        name_row2 = 'P_Lgvial' + str(i+8)
        x = Px + 6 + 24.3*i
        y_row1 = Py - 6
        y_row2 = Py - 31
        z = 37
        positions.loc[name_row1] = [x,y_row1,z]
        positions.loc[name_row2] = [x,y_row2,z]

    #HPLC Vials and small tubes Pipette
    for i in range(10):
        name_row1 = 'P_HPLCvial' + str(i)
        name_row2 = 'P_HPLCvial' + str(i+12)
        name_row3 = 'P_Smtubel' + str(i)
        x = Px + 4 + 16.4*i
        y_row1 = Py - 54
        y_row2 = Py - 69
        y_row3 = Py - 84.5
        z_HPLC = 28
        z_SmTube = 35
        positions.loc[name_row1] = [x,y_row1,z_HPLC]
        positions.loc[name_row2] = [x,y_row2,z_HPLC]
        positions.loc[name_row3] = [x,y_row3,z_SmTube]

    print('List of positions')
    print(positions)

def run_pipette():
    global tip_TF

    if pipette_TF == False:
        pipette_connect()

    speed = 8
    mixcycles = 3

    method = pd.read_csv('pipette_method.csv', index_col='Step')
    print('List of steps')
    print(method)
    print('\nRunning Method')

    for ind in method.index:
        volume = method['Volume ul'][ind]*10
        mix = method['Mix'][ind]
        tip = method['New Tip'][ind]

        if tip_TF == False:
            tip_pickup()
            tip_TF = True

        if tip == 'Yes':
            print('Change tip')
            tip_discard()
            tip_pickup()

        name = method['Source'][ind]
        x = str(positions._get_value(name, 'x'))
        y = str(positions._get_value(name, 'y'))
        z = str(positions._get_value(name, 'z'))
        cmd = 'G1 X' + x + ' Y' + y + ' Z0'
        duet_command(cmd)
        cmd = 'G1 X' + x + ' Y' + y + ' Z' + z
        duet_command(cmd)

        time.sleep(2)
        print('Aspirate ' + str(volume) + ' ul from ' + name)
        pipette.PipetteAPI('Aspirate', str(speed), str(volume), str(mixcycles))
        time.sleep(2)
        cmd = 'G1 X' + x + ' Y' + y + ' Z0'
        duet_command(cmd)

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
        pipette.PipetteAPI('Dispense', str(speed), str(volume), str(mixcycles))
        time.sleep(2)

        if mix!=0:
            pipette.PipetteAPI('BlowIn', str(speed), str(volume), str(mixcycles))
            time.sleep(1)
            print('Mix ' + str(mix) + ' times')
            pipette.PipetteAPI('Mix', str(speed), str(volume), str(mix))
            time.sleep(mix*3)
            
        cmd = 'G1 X' + x + ' Y' + y + ' Z0'
        duet_command(cmd)
        pipette.PipetteAPI('BlowIn', str(speed), str(volume), str(mixcycles))

    if tip_TF == True:
        tip_discard()
        tip_TF = False

    duet_command("G1 X0 Y0 Z0")

    print('Method complete')

def run_pump():

    chemyx_connect()

    method = pd.read_csv('pump_method.csv', index_col='Step')
    print('List of steps')
    print(method)
    print('\nRunning Method')

    for ind in method.index:
        volume = method['Volume ml'][ind]
        mix = method['Mix'][ind]

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

        if mix!=0:
            print('Mix ' + str(mix) + ' times')
            for i in range(mix):
                chemyx_aspirate(0.5)
                chemyx_dispense(0.5)
            
        cmd = 'G1 X' + x + ' Y' + y + ' Z0'
        duet_command(cmd)

    duet_command("G1 X0 Y0 Z0")

    print('Method complete')

def tip_pickup():
    global tip_count

    x_A1 = positions._get_value('A1', 'x')
    y_A1 = positions._get_value('A1', 'y')

    #increments tip pickup position (9mm spacing)
    x = x_A1 + 9*int((tip_count - tip_count % 8)/8)
    y = y_A1 - 9*(tip_count % 8)

    cmd = 'G1 X' + str(x) + ' Y' + str(y) + ' Z10'
    duet_command(cmd)
    cmd = 'G1 X' + str(x) + ' Y' + str(y) + ' Z44'
    duet_command(cmd)
    cmd = 'G1 X' + str(x) + ' Y' + str(y) + ' Z10'
    duet_command(cmd)
    tip_count += 1

def tip_discard():
    cmd = 'G1 X265 Y25 Z0'
    duet_command(cmd)
    cmd = 'G1 X265 Y25 Z36'
    duet_command(cmd)
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

    global root
    
    def moveto():
        x = xEntry.get()
        y = yEntry.get()
        z = zEntry.get()
        cmd = 'G1 X' + x + ' Y' + y + ' Z' + z
        print('move to ' + cmd)

        duet_command(cmd)

    def teach():
        name = nameEntry.get()
        x = xEntry.get()
        y = yEntry.get()
        z = zEntry.get()
        teach_positions.loc[name] = [x,y,z]
        teach_positions.to_csv('positions.csv')
        print("taught " + name + " at x = " + x + " y = " + y + " z = " + z)
        print(positions)

    def goto():
        name = nameEntry.get()
        print('goto ' + name)
        x = str(positions._get_value(name, 'x'))
        y = str(positions._get_value(name, 'y'))
        z = str(positions._get_value(name, 'z'))
        cmd = 'G1 X' + x + ' Y' + y + ' Z' + z
        duet_command(cmd)

        print('move successful')

    root = tk.Tk()
    root.title('Plexymer Liquid Handler')
    root.geometry('500x200')
    #root.minsize(400, 200)
    #root.maxsize(400, 200)
    xLabel = tk.Label(root, text='X')
    xEntry = tk.Entry(root)
    xEntry.insert(0, '100')
    xLabelComment = tk.Label(root, text='0..335 mm')
    yLabel = tk.Label(root, text='Y')
    yEntry = tk.Entry(root)
    yEntry.insert(0, '100')
    yLabelComment = tk.Label(root, text='0..250 mm')
    zLabel = tk.Label(root, text='Z')
    zEntry = tk.Entry(root)
    zEntry.insert(0, '0')
    zLabelComment = tk.Label(root, text='0..45 mm')
    nameLabel = tk.Label(root, text='name')
    nameEntry = tk.Entry(root)
    nameLabelComment = tk.Label(root, text='ex. pos1')
    sendCommandButton = tk.Button(root, text='Move',
                                  command=moveto)
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

#NOTE: There should be at least three coordinates in the file
#'A1' corresponding to A1 position of tip rack for pipette
#'P_rack' corresponding to teach position in rack using the pipette
#'N_rack' corresponding to teach position in rack using the needle
positions = pd.read_csv('positions.csv', index_col='name')
teach_positions = positions #included so that teaching new positions doesn't populate the csv with all the calculated rack positions
rack_positions()

duet_connect()
main()

rm.close()

if chemyx_TF == True:
    rx.close()
    print('close chemyx')

if pipette_TF == True:
    pipette.PipetteAPI('HomePipette', str(8), str(1000), str(3))
    time.sleep(5)
    pipette.PowerOff()
    print('poweroff pipette')

print('Program finished')