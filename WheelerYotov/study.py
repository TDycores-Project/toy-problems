import sys,subprocess,threading
import numpy as np
import sys

P,X = (float(sys.argv[1]),int(sys.argv[2])) if len(sys.argv) == 3 else (0,0)
N = np.asarray([8,16,32,64,128,256])
h = 1./N
E = np.zeros(N.shape)

print "1/h  Cartesian" if P < 1e-6 else "1/h  Perturbed %.2f" % P
for i,n in enumerate(N):
    process = subprocess.Popen("./WheelerYotov -pc_type lu -N %d -P %f -E %d" % (n,P,X),
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    output,errors = process.communicate()
    E[i] = float(output)
    print "%3d  %.2e" % (n,E[i])
print "rate = %.2f" % (np.polyfit(np.log10(h),np.log10(E),1)[0])
