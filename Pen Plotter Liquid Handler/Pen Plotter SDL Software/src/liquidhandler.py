import time
from tkinter import messagebox
import numpy as np
import pandas as pd
import pyvisa
from pyaxidraw import axidraw

import src.sdlvariables as var

# Liquid Handler Specific Variables
calibrationfactor = 0.95
speed_pendown = 0
speed_penup = 110
penposup = 100
penposdown = 21
x = 0
y = 0

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


ad = axidraw.AxiDraw()
ad.interactive()
ad.options.speed_pendown = speed_pendown
ad.options.speed_penup = speed_penup
ad.options.pen_pos_up = penposup
ad.options.pen_pos_down = penposdown
ad.options.units = 2

def axidraw_connect():
    if not ad.connect():
        messagebox.showerror("Connection Error", "Connect the AxiDraw")
    else:
        ad.update()
        print("AxiDraw Connected")

axidraw_connect()
initialize_chemyx()

#----------------------------- Manual Tab Movement Control -------------------------------
def ad_left(step):
    global x
    if x - step >= 0:
        ad.move(-step,0)
        x = x - step

def ad_right(step):
    global x
    if x + step <= 190:
        ad.move(step,0)
        x = x + step

def ad_up(step):
    global y
    if y - step >= 0:
        ad.move(0,-step)
        y = y - step

def ad_down(step):
    global y
    if y + step <= 140:
        ad.move(0,1)
        y = y + 1

def penup():
    ad.penup()

def pendown():
    ad.pendown()

def ad_move(x_val,y_val):
    ad.move(x_val,y_val)

def filter_next_mixtures(data_bank):
    dispense_mixtures = data_bank[(data_bank['Current'] == True) & (data_bank['Status'] == False)]
    
    return dispense_mixtures

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
# Calculate and draw the required volume of reagent. Checks to see whether there are any NaN values
# to filter out so we can sum the draw volume in this iteration
def draw_volume(drawvolume,x_coord,y_coord):
    # Remove NaN values
    drawvolume = [vol for vol in drawvolume if pd.notna(vol)]

    if sum(drawvolume) > 0:
        skip = False
        ad.moveto(104,0)
        ad.pendown()
        time.sleep(1)
        chemyx_command('set rate 1.8')
        time.sleep(0.5)
        chemyx_command('set volume 0.05')
        time.sleep(0.5)
        chemyx_command('start')
        time.sleep(6)
        chemyx_command('stop')
        time.sleep(0.5)
        # Move needle up and down to try and remove any air bubbles stuck at the tip of the needle
        ad.options.pen_pos_down = penposdown + 8
        ad.update()
        ad.options.pen_pos_down = penposdown
        ad.update()
        ad.options.pen_pos_down = penposdown + 8
        ad.update()
        ad.options.pen_pos_down = penposdown
        ad.update()
        ad.options.pen_pos_down = penposdown + 8
        ad.update()
        ad.options.pen_pos_down = penposdown
        ad.update()
        ad.options.pen_pos_down = penposdown + 8
        ad.update()
        ad.options.pen_pos_down = penposdown
        ad.update()
        chemyx_command('set volume -0.15')
        time.sleep(0.5)
        chemyx_command('start')
        time.sleep(0.1*32 + 3)
        chemyx_command('stop')
        time.sleep(0.5)

        draw_volume = round((sum(drawvolume)/ 2) / calibrationfactor + 0.1,3)
        print(draw_volume)
        ad.moveto(x_coord,y_coord)
        ad.options.pen_pos_down = 0
        ad.update()
        time.sleep(0.5)
        chemyx_command('set rate 0.5')
        time.sleep(0.5)
        chemyx_command('set volume -0.0125')
        time.sleep(0.5)
        chemyx_command('start')
        time.sleep(4.5)
        chemyx_command('stop')
        time.sleep(1)
        chemyx_command('set rate 1.8')
        time.sleep(0.5)
        ad.pendown()
        print(f'set volume -{draw_volume}')
        chemyx_command(f'set volume -{draw_volume}') #ml
        time.sleep(0.5)
        chemyx_command('start')
        time.sleep(draw_volume*33 + 3)
        chemyx_command('stop')
        time.sleep(0.5)
        chemyx_command('set rate 1.8')
        time.sleep(0.5)
        chemyx_command('set volume 0.05') # was halved
        time.sleep(0.5)
        chemyx_command('start')
        time.sleep(0.05*33 + 2)
        chemyx_command('stop')
        time.sleep(0.5)
        ad.options.pen_pos_down = penposdown
        ad.update()
    else:
        skip = True
    return skip

def perform_dispensing(dispense_mixtures, reservoir_coords):
    # Identify the columns between "Status" and "WV 1"
    chemyx_command('set units 0') #0 = ml/min
    time.sleep(0.5)
    chemyx_command('set diameter 4.78') #mm diameter
    time.sleep(0.5)
    start_idx = list(dispense_mixtures.columns).index("Status") + 1
    end_idx = list(dispense_mixtures.columns).index("Current")
    dynamic_columns = dispense_mixtures.columns[start_idx:end_idx]

    # Iterate over each column (component/ingredient)
    for column in dynamic_columns:
        reservoir = column  # Assume the column name matches the reservoir name
        if reservoir == 'Current':
            break
        x_reservoir_coord, y_reservoir_coord = reservoir_coords.loc[reservoir, ['X', 'Y']]
        print(reservoir)
        # Draw volume for the current component
        column_volumes = dispense_mixtures[column].values  # Get the entire column as an array
        skip = draw_volume(column_volumes, x_reservoir_coord, y_reservoir_coord)
        prev_dispense_volume = 0
        
        if not skip:
            # Iterate over each well to dispense the current component
            for index, row in dispense_mixtures.iterrows():
                xposition, yposition = dispense_mixtures.loc[index, ['X', 'Y']]  # Get xposition and yposition from data_bank using the index
                dispense_volume = row[column] / 2  # Get dispense_volume from dispense_mixtures
                if pd.notna(dispense_volume) and dispense_volume > 0.00001:
                    if dispense_volume != prev_dispense_volume:
                        chemyx_command(f'set volume {dispense_volume / calibrationfactor}')
                        time.sleep(0.1)

                    ad.moveto(xposition, yposition)
                    ad.pendown()
                    chemyx_command('start')
                    time.sleep(dispense_volume * 34 + 1.6)
                    ad.options.pen_pos_up = 50
                    ad.update()
                    ad.penup()  # jiggle up
                    ad.pendown()  # jiggle down
                    ad.penup()  # jiggle up
                    ad.pendown()  # jiggle down
                    chemyx_command('stop')
                    ad.options.pen_pos_up = 100
                    ad.update()
                    ad.delay(100)
                    ad.penup()
                    prev_dispense_volume = dispense_volume
            # Move to garbage position and reset after dispensing for the current component
            ad.moveto(104,27)
            ad.options.pen_pos_down = 16
            ad.update()
            chemyx_command('set volume 0.1625')
            time.sleep(0.5)
            ad.pendown()
            chemyx_command('start')
            time.sleep(0.1625 * 35 + 3)
            chemyx_command('stop')
            ad.penup()
            time.sleep(0.5)
            ad.moveto(104,70)
            ad.pendown()
            ad.penup()
            ad.pendown()
            ad.penup()
    ad.moveto(0,0)