from sympy import *
import numpy as np

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
print(div(BDDF1))

v11,v12,v13,v21,v22,v23,v31,v32,v33,v41,v42,v43 = symbols("v11 v12 v13 v21 v22 v23 v31 v32 v33 v41 v42 v43")
v51,v52,v53,v61,v62,v63,v71,v72,v73,v81,v82,v83 = symbols("v51 v52 v53 v61 v62 v63 v71 v72 v73 v81 v82 v83")

U = [a0,b0,c0,d0,a1,b1,c1,d1,a2,b2,c2,d2,r0,r1,r2,r3,s0,s1,s2,s3,t0,t1,t2,t3]
V = [v11,v12,v13,v21,v22,v23,v31,v32,v33,v41,v42,v43,v51,v52,v53,v61,v62,v63,v71,v72,v73,v81,v82,v83]
        
eqs = (v11 - -BDDF1[0].subs({x:-1,y:-1,z:-1}),
       v12 - -BDDF1[1].subs({x:-1,y:-1,z:-1}),
       v13 - -BDDF1[2].subs({x:-1,y:-1,z:-1}),

       v21 - +BDDF1[0].subs({x:+1,y:-1,z:-1}),
       v22 - -BDDF1[1].subs({x:+1,y:-1,z:-1}),
       v23 - -BDDF1[2].subs({x:+1,y:-1,z:-1}),

       v31 - -BDDF1[0].subs({x:-1,y:+1,z:-1}),
       v32 - +BDDF1[1].subs({x:-1,y:+1,z:-1}),
       v33 - -BDDF1[2].subs({x:-1,y:+1,z:-1}),

       v41 - +BDDF1[0].subs({x:+1,y:+1,z:-1}),
       v42 - +BDDF1[1].subs({x:+1,y:+1,z:-1}),
       v43 - -BDDF1[2].subs({x:+1,y:+1,z:-1}),
       
       v51 - -BDDF1[0].subs({x:-1,y:-1,z:+1}),
       v52 - -BDDF1[1].subs({x:-1,y:-1,z:+1}),
       v53 - +BDDF1[2].subs({x:-1,y:-1,z:+1}),

       v61 - +BDDF1[0].subs({x:+1,y:-1,z:+1}),
       v62 - -BDDF1[1].subs({x:+1,y:-1,z:+1}),
       v63 - +BDDF1[2].subs({x:+1,y:-1,z:+1}),

       v71 - -BDDF1[0].subs({x:-1,y:+1,z:+1}),
       v72 - +BDDF1[1].subs({x:-1,y:+1,z:+1}),
       v73 - +BDDF1[2].subs({x:-1,y:+1,z:+1}),

       v81 - +BDDF1[0].subs({x:+1,y:+1,z:+1}),
       v82 - +BDDF1[1].subs({x:+1,y:+1,z:+1}),
       v83 - +BDDF1[2].subs({x:+1,y:+1,z:+1}))

sol = solve(eqs)    
    
BDDF1[0] = collect(expand(BDDF1[0].subs(sol)),V)
BDDF1[1] = collect(expand(BDDF1[1].subs(sol)),V)
BDDF1[2] = collect(expand(BDDF1[2].subs(sol)),V)

 
def petsc(s):
    return s.replace("x","x[0]").replace("y","x[1]").replace("z","x[2]").replace("= 0","= +0")

if True:
    print(petsc("B[0]  = %s;" % (BDDF1[0].coeff(v11).evalf())))
    print(petsc("B[1]  = %s;" % (BDDF1[1].coeff(v12).evalf())))
    print(petsc("B[2]  = %s;" % (BDDF1[2].coeff(v13).evalf())))

    print(petsc("B[3]  = %s;" % (BDDF1[0].coeff(v21).evalf())))
    print(petsc("B[4]  = %s;" % (BDDF1[1].coeff(v22).evalf())))
    print(petsc("B[5]  = %s;" % (BDDF1[2].coeff(v23).evalf())))

    print(petsc("B[6]  = %s;" % (BDDF1[0].coeff(v31).evalf())))
    print(petsc("B[7]  = %s;" % (BDDF1[1].coeff(v32).evalf())))
    print(petsc("B[8]  = %s;" % (BDDF1[2].coeff(v33).evalf())))

    print(petsc("B[9]  = %s;" % (BDDF1[0].coeff(v41).evalf())))
    print(petsc("B[10] = %s;" % (BDDF1[1].coeff(v42).evalf())))
    print(petsc("B[11] = %s;" % (BDDF1[2].coeff(v43).evalf())))

    print(petsc("B[12] = %s;" % (BDDF1[0].coeff(v51).evalf())))
    print(petsc("B[13] = %s;" % (BDDF1[1].coeff(v52).evalf())))
    print(petsc("B[14] = %s;" % (BDDF1[2].coeff(v53).evalf())))

    print(petsc("B[15] = %s;" % (BDDF1[0].coeff(v61).evalf())))
    print(petsc("B[16] = %s;" % (BDDF1[1].coeff(v62).evalf())))
    print(petsc("B[17] = %s;" % (BDDF1[2].coeff(v63).evalf())))

    print(petsc("B[18] = %s;" % (BDDF1[0].coeff(v71).evalf())))
    print(petsc("B[19] = %s;" % (BDDF1[1].coeff(v72).evalf())))
    print(petsc("B[20] = %s;" % (BDDF1[2].coeff(v73).evalf())))

    print(petsc("B[21] = %s;" % (BDDF1[0].coeff(v81).evalf())))
    print(petsc("B[22] = %s;" % (BDDF1[1].coeff(v82).evalf())))
    print(petsc("B[23] = %s;" % (BDDF1[2].coeff(v83).evalf())))

    

