import sys
from time import sleep
total = 1000
point = total / 100
increment = total / 20
for i in range(total):
    sys.stdout.write("\r"+str(i) + "%")
    sys.stdout.flush()
    sleep(0.25)
