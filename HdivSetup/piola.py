import matplotlib.pyplot as plt
import numpy as np
    
def LeftClick(event):
    global data_index
    if event.button != 1   : return
    if event.inaxes is None: return # only worry about stuff in the axis
    d = np.linalg.norm(X-np.asarray([event.xdata,event.ydata]),axis=1)
    i = np.argmin(d)
    if d[i] < 0.1: data_index = i

def LeftRelease(event):
    global data_index
    if event.button != 1: return
    data_index = None
    
def LeftDrag(event):
    global data_index
    if data_index is None: return
    X[data_index] = [event.xdata,event.ydata]
    x,n = PiolaTransform()
    edge_plot.set_xdata(X[[0,1,2,0],0])
    edge_plot.set_ydata(X[[0,1,2,0],1])
    vert_plot.set_xdata(X[:,0])
    vert_plot.set_ydata(X[:,1])
    ax.patches.pop(0)
    norm_plot = plt.Arrow(x[0],x[1],n[0],n[1],width=0.2,lw=0.1,color='r')
    ax.add_patch(norm_plot)
    fig.canvas.draw_idle()

def PiolaTransform():
    x0 = np.asarray([0.,0.]) # map [0,0] to element
    x  = X[0]*  0.5*(x0[0]+1)
    x += X[1]*  0.5*(x0[1]+1)
    x += X[2]*(-0.5*(x0[0]+x0[1]))
    dx = 0.5*(X[0]-X[2])    # compute Jacobian
    dy = 0.5*(X[1]-X[2])
    DF = np.asarray([dx,dy]).T
    J  = np.linalg.det(DF)
    n  = DF @ N0 / J        # Piola transform
    return x,n

# initialize data
N0 = np.asarray([0.5*np.sqrt(2),0.5*np.sqrt(2)])
X  = np.asarray([[0.8,2.1],
                 [2.0,1.0],
                 [2.9,2.6]])
x,n = PiolaTransform()
data_index = None

# setup plots
fig,ax = plt.subplots(tight_layout=True)
ax.axis('equal')
Xm = X.mean(axis=0)
dx = 5.
ax.set_xlim(Xm[0]-dx,Xm[0]+dx)
ax.set_ylim(Xm[1]-dx,Xm[1]+dx)

# plot element, vertices, and mapped unit normal
edge_plot, = ax.plot(X[[0,1,2,0],0],X[[0,1,2,0],1],'-k')
vert_plot, = ax.plot(X[:,0],X[:,1],'ok')
norm_plot  = plt.Arrow(x[0],x[1],n[0],n[1],width=0.2,lw=0.1,color='r')
ax.add_patch(norm_plot)

# setup figure callbacks
fig.canvas.mpl_connect('button_press_event'  ,LeftClick)
fig.canvas.mpl_connect('button_release_event',LeftRelease)
fig.canvas.mpl_connect('motion_notify_event' ,LeftDrag)

plt.show()
