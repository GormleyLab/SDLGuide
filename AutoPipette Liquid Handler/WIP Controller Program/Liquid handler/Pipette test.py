import clr
import sys, os, time

filepath = r"C:\Users\adamj\Box\_Programming\IntegraAPI\RemoteControl\RemoteControl\bin\Debug\RemoteControl.dll"

clr.AddReference(filepath)

from RemoteControl import PythonAPI

pipette = PythonAPI();

speed = 8
volume = 1000
mixcycles = 3
comport = 'COM6'

pipette.Connect(comport)
time.sleep(2)
pipette.PipetteAPI('Aspirate', str(speed), str(volume), str(mixcycles))
time.sleep(2)
pipette.PipetteAPI('Dispense', str(speed), str(volume), str(mixcycles))
time.sleep(2)
pipette.PipetteAPI('BlowIn', str(speed), str(volume), str(mixcycles))
time.sleep(2)
pipette.PipetteAPI('Mix', str(speed), str(volume), str(mixcycles))
time.sleep(10)
pipette.PipetteAPI('BlowIn', str(speed), str(volume), str(mixcycles))
time.sleep(2)
pipette.PipetteAPI('Aspirate', str(speed), str(volume), str(mixcycles))
time.sleep(2)
pipette.PipetteAPI('Dispense', str(speed), str(volume), str(mixcycles))
time.sleep(2)
pipette.PipetteAPI('HomePipette', str(speed), str(volume), str(mixcycles))
time.sleep(5)
pipette.PowerOff()


'''List of actions and descriptions:
pipette.Connect() - Connect to waiting pipette
pipette.Disconnect() - Disconnect connection but keep the pipette on
pipette.PowerOff() - Use this rather than Disconnect as the pipette returns to connecting mode when restarted
pipette.PipetteAPI('Aspirate', str(speed), str(volume), str(mixcycles)) - Aspirate
pipette.PipetteAPI('Dispense', str(speed), str(volume), str(mixcycles)) - Dispense. But, this results in a blow out which means that a blow in is required after to reset
pipette.PipetteAPI('Dispense_NoBO', str(speed), str(volume), str(mixcycles)) - Use most of the time but it doesn't blow out so there might be a bit left in the tip
pipette.PipetteAPI('Mix', str(speed), str(volume), str(mixcycles)) - Mix but this results in a a blow out which means that a blow in is required after to reset
pipette.PipetteAPI('Mix_NoBo', str(speed), str(volume), str(mixcycles)) - Use most of the time but it doesn't blow out so there might be a bit left in the tip
pipette.PipetteAPI('RelMixAspirate', str(speed), str(volume), str(mixcycles)) - Mixes then aspirates
pipette.PipetteAPI('RelMixDispense', str(speed), str(volume), str(mixcycles)) - Mixes during dispense. Must be preceeded by aspirate. Does not blow out so does not need resetting
pipette.PipetteAPI('Purge', str(speed), str(volume), str(mixcycles)) - Purges tips
pipette.PipetteAPI('BlowOut', str(speed), str(volume), str(mixcycles)) - Pushes a bit out of the tip that is remaining
pipette.PipetteAPI('BlowIn', str(speed), str(volume), str(mixcycles)) - resets pipette to zero position after a blow out. Make sure tip is not in liquid otherwise it will aspitate a bit
pipette.PipetteAPI('HomePipette', str(speed), str(volume), str(mixcycles)) - resets the whole pipette back to zero position. Should be done prior to turning off.
'''