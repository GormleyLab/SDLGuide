#Python code for remote control of Chemyx syringe pump. See manual for list of all commands
import pyvisa	#must pip install pyvisa-py
import serial	#must pip install pyserial
import os, time, sys



def initialize_chemyx():
    global chemyx
    global rm
    rm = pyvisa.ResourceManager()
    chemyx = rm.open_resource('ASRL4::INSTR', timeout = 25000, write_termination = '\r\n', 
        read_termination = '\r\n', baud_rate = 38400, data_bits = 8)
    #ASRL# where the number should be the com port #


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

initialize_chemyx()

chemyx_command('set units mL/min')
chemyx_command('set diameter 8.66')
chemyx_command('set rate 8')
#chemyx_command('set mode 0')
chemyx_command('set volume -0.5')

chemyx_command('start')

print('done')

rm.close()