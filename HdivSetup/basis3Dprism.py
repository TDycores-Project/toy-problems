from sympy import *
import numpy as np

petsc = True
"""
    Local numbering of Prism
    
        4--------6           z
      / |                    |
    5   |                    |
    |   |                    /----y
    |   |                   /
    |   1 -------3         x
    | /
    2
nodes:             normals at node:
nd1 = (-1,-1,-1)   nl = [0,-1,0],  nbk = [-1,0,0],               nbt = [0,0,-1]
nd2 = (1 ,-1,-1)   nl = [0,-1,0],  nd = [sqrt(2)/2,sqrt(2)/2,0], nbt = [0,0,-1]
nd3 = (-1,1 ,-1)   nbk = [-1,0,0], nd = [sqrt(2)/2,sqrt(2)/2,0], nbt = [0,0,-1]
nd4 = (-1,-1, 1)   nl = [0,-1,0],  nbk = [-1,0,0],               nt = [0,0,1]
nd5 = (1 ,-1, 1)   nl = [0,-1,0],  nd = [sqrt(2)/2,sqrt(2)/2,0], nt = [0,0,1]
nd6 = (-1,1 , 1)   nbk = [-1,0,0], nd = [sqrt(2)/2,sqrt(2)/2,0], nt = [0,0,1]
"""
             
x,y,z = symbols('x y z')
a0,b0,c0,d0,a1,b1,c1,d1,a2,b2,c2,d2 = symbols("a0 b0 c0 d0 a1 b1 c1 d1 a2 b2 c2 d2")
r0,r1,s0,s1,t0,t1 = symbols("r0 r1 s0 s1 t0 t1")

vx = a0 + b0*x + c0*y + d0*z + 2*r0*y*z + 2*s1*x*z + t0*x**2 + t1*x*y
vy = a1 + b1*x + c1*y + d1*z + 2*r1*x*z + 2*s0*y*z + t0*x*y  + t1*y**2
vz = a2 + b2*x + c2*y + d2*z - s0*z**2  - s1*z**2  - 3*t0*x*z  - 3*t1*y*z

nl = [0,-1,0]
nbk = [-1,0,0]
nbt = [0,0,-1]
nt = [0,0,1]
cf = 0.7071067811865476
nd = [cf,cf,0]
for i in range(6): # for each vertex
    for j in range(3): # for each direction
        k = 3*i+j
        eqs = [ np.dot([vx,vy,vz],nl).subs( {x:-1,y:-1,z:-1}),   # node1
                np.dot([vx,vy,vz],nbk).subs({x:-1,y:-1,z:-1}),   # node1
                np.dot([vx,vy,vz],nbt).subs({x:-1,y:-1,z:-1}),   # node1
                np.dot([vx,vy,vz],nl).subs( {x: 1,y:-1,z:-1}),   # node2
                np.dot([vx,vy,vz],nd).subs( {x: 1,y:-1,z:-1}),   # node2
                np.dot([vx,vy,vz],nbt).subs({x: 1,y:-1,z:-1}),   # node2
                np.dot([vx,vy,vz],nbk).subs({x:-1,y: 1,z:-1}),   # node3
                np.dot([vx,vy,vz],nd).subs( {x:-1,y: 1,z:-1}),   # node3
                np.dot([vx,vy,vz],nbt).subs({x:-1,y: 1,z:-1}),   # node3
                np.dot([vx,vy,vz],nl).subs( {x:-1,y:-1,z: 1}),   # node4
                np.dot([vx,vy,vz],nbk).subs({x:-1,y:-1,z: 1}),   # node4
                np.dot([vx,vy,vz],nt).subs( {x:-1,y:-1,z: 1}),   # node4
                np.dot([vx,vy,vz],nl).subs( {x: 1,y:-1,z: 1}),   # node5
                np.dot([vx,vy,vz],nd).subs( {x: 1,y:-1,z: 1}),   # node5
                np.dot([vx,vy,vz],nt).subs( {x: 1,y:-1,z: 1}),   # node5
                np.dot([vx,vy,vz],nbk).subs({x:-1,y: 1,z: 1}),   # node6
                np.dot([vx,vy,vz],nd).subs( {x:-1,y: 1,z: 1}),   # node6
                np.dot([vx,vy,vz],nt).subs( {x:-1,y: 1,z: 1}),  ]# node6
        
        eqs[k] -= 1 
        sol = solve(eqs)
        ux = vx.subs(sol)
        uy = vy.subs(sol)
        uz = vz.subs(sol)
        
        if petsc:
            def _f(fcn):
                fcn = fcn.replace("x**2","x*x")
                fcn = fcn.replace("y**2","y*y")
                fcn = fcn.replace("z**2","z*z")
                fcn = fcn.replace("x","x[0]")
                fcn = fcn.replace("y","x[1]")
                fcn = fcn.replace("z","x[2]")
                #if "/8" in fcn: fcn = "(%s)*0.125;" % (fcn.replace("/8",""))
                #if "/16" in fcn: fcn = "(%s)*0.0625;" % (fcn.replace("/16",""))
                return fcn
            print("B[%2d] = " % (3*k)   + _f("%s" % (ux)))
            print("B[%2d] = " % (3*k+1) + _f("%s" % (uy)))
            print("B[%2d] = " % (3*k+2) + _f("%s" % (uz)))
