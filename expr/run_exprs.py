import sys
import os
import time

while True:
    with open(sys.argv[1], 'r') as f:
        cmds = f.readlines()
    
    if len(cmds) == 0:
        break
    
    with open(sys.argv[1], 'w') as f:
        f.write(''.join(cmds[1:]))
    
    print('>>>>> start time:', time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    print('>>>>> command:', cmds[0])
    os.system(cmds[0])
    print('<<<<< end time:', time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
