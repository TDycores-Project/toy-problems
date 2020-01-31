from sympy import *
import numpy as np

petsc = True

def div(v):
    d  = diff(v[0],x)
    d += diff(v[1],y)
    d += diff(v[2],z)
    return d
    
def curl(v):
    c = []
    c.append(  diff(v[2],y)-diff(v[1],z) )
    c.append(-(diff(v[2],x)-diff(v[0],z)))
    c.append(  diff(v[1],x)-diff(v[0],y) )
    return np.asarray(c)
             
x,y,z = symbols('x y z')
a0,b0,c0,d0,a1,b1,c1,d1,a2,b2,c2,d2 = symbols("a0 b0 c0 d0 a1 b1 c1 d1 a2 b2 c2 d2")
r0,r1,r2,r3,s0,s1,s2,s3,t0,t1,t2,t3 = symbols("r0 r1 r2 r3 s0 s1 s2 s3 t0 t1 t2 t3")

BDDF1 = np.asarray([a0+b0*x+c0*y+d0*z,
                    a1+b1*x+c1*y+d1*z,
                    a2+b2*x+c2*y+d2*z])
BDDF1 += r0*curl([0       ,0       ,x*y*z   ])
BDDF1 += r1*curl([0       ,0       ,x*y**2  ])
BDDF1 += r2*curl([0       ,0       ,x**2*z  ])
BDDF1 += r3*curl([0       ,0       ,x**2*y*z])
BDDF1 += s0*curl([x*y*z   ,0       ,0       ])
BDDF1 += s1*curl([y*z**2  ,0       ,0       ])
BDDF1 += s2*curl([x*y**2  ,0       ,0       ])
BDDF1 += s3*curl([x*y**2*z,0       ,0       ])
BDDF1 += t0*curl([0       ,x*y*z   ,0       ])
BDDF1 += t1*curl([0       ,x**2*z  ,0       ])
BDDF1 += t2*curl([0       ,y*z**2  ,0       ])
BDDF1 += t3*curl([0       ,x*y*z**2,0       ])

for i in range(8): # for each vertex
    for j in range(3): # for each direction
        k = 3*i+j

        eqs = []
        for v in [{x:-1,y:-1,z:-1},
                  {x:+1,y:-1,z:-1},
                  {x:-1,y:+1,z:-1},
                  {x:+1,y:+1,z:-1},
                  {x:-1,y:-1,z:+1},
                  {x:+1,y:-1,z:+1},
                  {x:-1,y:+1,z:+1},
                  {x:+1,y:+1,z:+1}]:
            for l,d in enumerate([x,y,z]): # the outward normal at evaluated at each vertex/direction
                n = np.zeros(3,dtype=int)
                n[l] = v[d]
                eqs.append(np.dot(BDDF1,n).subs(v))
        
        eqs[k] -= 1 # the k^th functions should be a 1, rest are 0
        sol = solve(eqs)
        ux = BDDF1[0].subs(sol)
        uy = BDDF1[1].subs(sol)
        uz = BDDF1[2].subs(sol)
        
        if petsc:
            def _f(fcn):
                fcn = fcn.replace("x**2","x*x")
                fcn = fcn.replace("y**2","y*y")
                fcn = fcn.replace("z**2","z*z")
                fcn = fcn.replace("x","x[0]")
                fcn = fcn.replace("y","x[1]")
                fcn = fcn.replace("z","x[2]")
                if "/8" in fcn: fcn = "(%s)*0.125;" % (fcn.replace("/8",""))
                if "/16" in fcn: fcn = "(%s)*0.0625;" % (fcn.replace("/16",""))
                return fcn
            print("B[%2d] = " % (3*k)   + _f("%s" % (ux)))
            print("B[%2d] = " % (3*k+1) + _f("%s" % (uy)))
            print("B[%2d] = " % (3*k+2) + _f("%s" % (uz)))
