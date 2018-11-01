import sys,subprocess,threading
import pylab as plt
import numpy as np
import sys

def _buildTriangle(rate,offset,h,E):
    dh = np.log10(h[0])-np.log10(h[-1])
    x  =    [10**(np.log10(h[-1])+ offset     *dh)]
    x.append(10**(np.log10(h[-1])+(offset+0.15)*dh))
    x.append(x[-1])
    x.append(x[ 0])
    y = [E[-1],E[-1]]
    y.append( 10**(np.log10(E[-1]) + rate*(np.log10(x[1])-np.log10(x[0]))))
    y.append(E[-1])
    return x,y
    
N = np.asarray([8,16,32,64,128,256,512]) #[:-2]
h = 1./N
E = np.zeros((N.size,3))
for i,n in enumerate(N):
    ref = "" if i == 0 else " -dm_refine %d" % i
    process = subprocess.Popen("./WheelerYotov -pc_type lu -N %d -P 1 -E 2 %s" % (N[0],ref),
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    output,errors = process.communicate()
    E[i] = [float(v) for v in output.split()]
    print "%3d  %.2e %.2e %.2e" % (n,E[i,0],E[i,1],E[i,2])
s = 2
print "rate = %.2f  %.2f  %.2f" % (np.polyfit(np.log10(h[s:]),np.log10(E[s:,0]),1)[0],
                                   np.polyfit(np.log10(h[s:]),np.log10(E[s:,1]),1)[0],
                                   np.polyfit(np.log10(h[s:]),np.log10(E[s:,2]),1)[0])

pad = 0.05
lbl = 0.19
dh  = np.log10(h[0])-np.log10(h[-1])
dh *= (1. + 2*pad + lbl)
hL  = 10**(np.log10(h[-1]) -  pad*dh)
hR  = 10**(np.log10(h[ 0]) + (pad+lbl)*dh)
h0  = 10**(np.log10(h[ 0]) + 0.02*dh)

fig,ax = plt.subplots(tight_layout=True)
ax.loglog(h,E[:,0],'-o',ms=4)
ax.loglog(h,E[:,1],'-^',ms=4)
ax.loglog(h,E[:,2],'-s',ms=4)
tx,ty = _buildTriangle(2.0,0.05,h,E[:,0])
ax.loglog(tx,ty,'-k')
ax.text(tx[1],ty[1],"2 ",ha="right",va="bottom")
ax.text(h0,E[0,0],r"$|||p-p_h|||$")
ax.text(h0,E[0,1],r"$|||\mathbf{u}-\mathbf{u}_h|||$")
ax.text(h0,E[0,2],r"$|||\nabla\cdot\left(\mathbf{u}-\mathbf{u}_h\right)|||$")
ax.set_xlim(hL,hR)
ax.set_xlabel("Mesh size, $h$")
ax.set_ylabel("Norm of Error")
fig.savefig("out.pdf")
fig.savefig("out.png")
