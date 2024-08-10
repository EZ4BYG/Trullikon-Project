"""
This file contains 3 main functions: GeertsmaSol_JP_py, Coeff_ISO_py, and Coeff_VTI_py.
Relationship: 'GeertsmaSol_JP_py' calls 'Coeff_ISO_py' and 'Coeff_VTI_py'.
"""

import numpy as np
import cmath as cm
import numpy.linalg.linalg as lin
from scipy import special

def GeertsmaSol_JP_py(NL, G, nu, alpha, p, z_top, z_bot, r, R, iL_pp, RHO, VS, VP):
    """
    input:
        NL: number of layers
        G: shear modulus array of 1xNL, used for isotropic model [Pa] ———— Only used in 'Coeff_ISO_py' function!
        nu: Poisson ratio array of 1xNL, used for isotropic model     ———— Only used in 'Coeff_ISO_py' function!
        alpha: Biot coefficient array of 1xNL
        p: pressure array of 1xNL e.g. =[0,+/-1,0,0] of iL_pp=2, unit [Pa]
        z_top: layer top coord. array of 1xNL, actually always 0.0
        z_bot: layer bottom coord. array of 1xNL
        r: observation points' radial coord. array for response points [m]
        R: radius of reservoir
        iL_pp: layer index for pressurized reservoir (a number)
        RHO: density array of 1xNL, used for VTI model [kg/m^3] ———— Only used in 'Coeff_VTI_py' function!
        VS: s-wave velocity array of 2xNL, used for VTI model [m/s],
            row 1=horizontal, row2=vertical                     ———— Only used in 'Coeff_VTI_py' function!
        VP: p-wave velocity array of 2xNL, used for VTI model [m/s],
            row 1=horizontal, row2=vertical                     ———— Only used in 'Coeff_VTI_py' function!
    output:
        ur_r: radial displacements at top surface [m]
        uz_r: vertical displacements at top surface [m]
    Call:
        [C,P,E]=Coeff_ISO_py(z,xi,nu,G,alpha,p,R,iflag_E)
        [C,P,E]=Coeff_VTI_py(z,k,rho,Vs,Vst,Vp,Vpt,alpha,p,R,iflag_E)
    """
    NR = 1      # parameter for pressure distribution of triangular shape
    Nxi = 250   # adhoc: 500 seems good enough
    depth_reservoir = np.sum(z_bot[0:iL_pp])  # compute the depth of the reservoir layer
    # xi_max = 10.0*pi/(R/NR) # adhoc: 10pi/R seems good enough, too :)
    xi_max = 10.0/depth_reservoir
    dxi = xi_max/Nxi
    xi = np.arange(0,Nxi+1)*dxi + dxi*1.0e-2
    
    iflag_E = 1
    Lr = np.size(r)
    Lxi = np.size(xi)
    uz_xi = np.zeros((NL,Lxi), dtype=complex)
    uz_r = np.zeros((NL,Lr), dtype=complex)
    ur_xi = np.zeros((NL,Lxi), dtype=complex)
    ur_r = np.zeros((NL,Lr), dtype=complex)
    
    for ixi in range(0,Lxi):
        
        C_top = np.zeros((NL,4,4), dtype=complex)
        C_bot = np.zeros((NL,4,4), dtype=complex)
        P_top = np.zeros((NL,4,1), dtype=complex)
        P_bot = np.zeros((NL,4,1), dtype=complex)
        E_top = np.zeros((NL,4,4), dtype=complex)
        E_bot = np.zeros((NL,4,4), dtype=complex)
        
        for iL in range(0,NL):
            # Isotropic model: if nu is not empty, then it is always isotropic!!
            # nu, G: √
            # RHO, VS, VP: []
            if np.size(nu) > 0:
                # Return the coefficients for the 'top' of the layer
                [C_top[iL],P_top[iL],E_top[iL]] = Coeff_ISO_py(z=z_top[iL], xi=xi[ixi],
                                                               nu=nu[iL], G=G[iL],
                                                               alpha=alpha[iL], p=p[iL], R=R, iflag_E=iflag_E)
                if iL < NL-1:
                    # Return the coefficients for the 'bottom' of the layer
                    [C_bot[iL],P_bot[iL],E_bot[iL]] = Coeff_ISO_py(z=z_bot[iL], xi=xi[ixi],
                                                                   nu=nu[iL], G=G[iL],
                                                                   alpha=alpha[iL], p=p[iL], R=R, iflag_E=iflag_E)
 
            # VTI model
            # RHO, VS, VP: √
            # nu, G: []
            elif np.size(VS) > 0:
                # Return the coefficients for the 'top' of the layer
                [C_top[iL],P_top[iL],E_top[iL]] = Coeff_VTI_py(z=z_top[iL], k=xi[ixi], rho=RHO[iL],
                                                               Vs=VS[0][iL], Vst=VS[1][iL], Vp=VP[0][iL], Vpt=VP[1][iL],
                                                               alpha=alpha[iL], p=p[iL], R=R, iflag_E=iflag_E)
                if iL < NL-1:
                    # Return the coefficients for the 'bottom' of the layer
                    [C_bot[iL],P_bot[iL],E_bot[iL]] = Coeff_VTI_py(z=z_bot[iL], k=xi[ixi], rho=RHO[iL],
                                                                   Vs=VS[0][iL], Vst=VS[1][iL], Vp=VP[0][iL], Vpt=VP[1][iL],
                                                                   alpha=alpha[iL], p=p[iL], R=R, iflag_E=iflag_E)
 
        Ndof = 4*(NL-1)+2
        C = np.zeros((Ndof,Ndof), dtype=complex)
        P = np.zeros((Ndof,1), dtype=complex)
        
        for iL in range(0,NL):
            if iL == 0:
                row1=+1-1
                row2=+3-1
                col1=+1-1
                C[row1:(row1+1+1),col1:(col1+3+1)]=-C_top[iL][2: ,:]
                C[row2:(row2+3+1),col1:(col1+3+1)]=+C_bot[iL][0: ,:]
                P[row1:(row1+1+1)]=P[row1:(row1+1+1)]-+P_top[iL][2:,:]
                P[row2:(row2+3+1)]=P[row2:(row2+3+1)]--P_bot[iL][0:,:]
                
            elif iL == NL-1:
                row1 = 4*(iL-1+1)-1-1
                col1 = 4*(iL-1+1)+1-1
                C[row1:(row1+3+1),col1:(col1+1+1)]=-C_top[iL][0: ,[1,3]]
                P[row1:(row1+3+1)] = P[row1:(row1+3+1)]-+P_top[iL][0:,:]
                
            else:
                row1 = 4*(iL-1+1)-1-1
                row2 = row1+4-0
                col1 = 4*(iL-1+1)+1-1
                C[row1:(row1+3+1),col1:(col1+3+1)]=-C_top[iL][0: ,:]
                C[row2:(row2+3+1),col1:(col1+3+1)]=+C_bot[iL][0: ,:]
                P[row1:(row1+3+1)]=P[row1:(row1+3+1)]-+P_top[iL][0:,:]
                P[row2:(row2+3+1)]=P[row2:(row2+3+1)]--P_bot[iL][0:,:]

        A = np.dot(lin.inv(C),P)
        # vertical disp at top surface
        uz_xi[0,ixi] = np.dot(C_top[0][1],A[0:4,0])
        # radial disp at top surface
        ur_xi[0,ixi] = np.dot(C_top[0][0],A[0:4,0])
    
    duNR = np.zeros(Lxi)
    for nr in range(1,NR+1,1):
        duNR = duNR + (nr/NR)*R/xi*special.jn(1,(nr/NR)*R*xi)/NR  # R is used!
            
    for ir in range(0,Lr):
        tmp = -uz_xi[0]*xi*special.jn(0,xi*r[ir])*duNR
        uz_r[0,ir] = np.sum(tmp)*dxi
        tmp = ur_xi[0]*xi*special.jn(1,xi*r[ir])*duNR
        ur_r[0,ir] = np.sum(tmp)*dxi

    return [uz_r, ur_r]


# Equation 8: Solution for isotropic medium for the completeness of the code
def Coeff_ISO_py(z, xi, nu, G, alpha, p, R, iflag_E):
    
    C = np.zeros((4,4))
    P = np.zeros((4,1))

    # Equation (8)
    exp1 = np.exp(+xi*z)
    exp2 = np.exp(-xi*z)
    cm = alpha*(1.0-2.0*nu)/2.0/G/(1.0-nu)  # Note: nu = Poisson ratio; G = shear modulus
    a = 1.0/2.0/(1.0-2.0*nu)
    b = nu/(1.0-2.0*nu)
    # ur, uz, srz, szz
    C[0] = np.array([                   a*z*exp1**iflag_E,              -a*z*exp2**iflag_E,     exp1**iflag_E,     exp2**iflag_E])  # terms in U1
    C[1] = np.array([      ((a+1.0)/xi-a*z)*exp1**iflag_E, -((a+1.0)/xi+a*z)*exp2**iflag_E,    -exp1**iflag_E,     exp2**iflag_E])  # terms in U3
    C[2] = 2.0*G*np.array([(a*z*xi-1.0/2.0)*exp1**iflag_E, +(a*z*xi+1.0/2.0)*exp2**iflag_E,  xi*exp1**iflag_E, -xi*exp2**iflag_E])  # terms in Srz
    C[3] = 2.0*G*np.array([  (1.0-a*z*xi+b)*exp1**iflag_E,   +(1.0+a*z*xi+b)*exp2**iflag_E, -xi*exp1**iflag_E, -xi*exp2**iflag_E])  # terms in Szz
    E = np.diag([exp1, exp2, exp1, exp2])
    P[0] = cm/xi*(-p)
    P[3] = -2.0*G*cm*(-p)

    return [C,P,E]

# Solution for anisotropic medium
def Coeff_VTI_py(z, k, rho, Vs, Vst, Vp, Vpt, alpha, p, R, iflag_E):
    C = np.zeros((4,4), dtype=complex)
    P = np.zeros((4,1))
    
    Vp2 = Vp**2.0
    Vs2 = Vs**2.0
    Vpt2 = Vpt**2.0
    Vst2 = Vst**2.0

    # Parameters affiliated with the equation (6)
    a = 1.0
    b = (Vp2 - (Vpt2-2.0*Vst2))/Vst2
    c = Vp2/Vpt2
    # Equation (6)
    k1 = cm.sqrt( (b+cm.sqrt(b**2.0-4.0*a*c))/2.0/a )*k  # 小问题：应该是b/2/a - cm.sqrt(b**2-4*a*c)
    k2 = cm.sqrt( (b-cm.sqrt(b**2.0-4.0*a*c))/2.0/a )*k  # 同理
    ph1 = -(Vpt2-Vst2)*(k1/k) / (-Vst2+Vpt2*(k1/k)**2.0)
    ph2 = -(Vpt2-Vst2)*(k2/k) / (-Vst2+Vpt2*(k2/k)**2.0)

    # Parameters affiliated with the equation (7)
    exp1 = cm.exp(+k1*z)
    exp2 = cm.exp(-k1*z)
    exp3 = cm.exp(+k2*z)
    exp4 = cm.exp(-k2*z)
    a = k1-k*ph1
    b = k2-k*ph2
    c = (Vpt2-2.0*Vst2)*k + Vpt2*k1*ph1
    d = (Vpt2-2.0*Vst2)*k + Vpt2*k2*ph2
    # Equation (7): ur, uz, srz, szz
    C[0] = np.array([            exp1,      exp2,     exp3,      exp4])
    C[1] = np.array([        ph1*exp1, -ph1*exp2, ph2*exp3, -ph2*exp4])
    C[2] = rho*Vst2*np.array([ a*exp1,   -a*exp2,   b*exp3,   -b*exp4])  # terms in Srz
    C[3] = rho*np.array([      c*exp1,    c*exp2,   d*exp3,    d*exp4])  # terms in Szz
    E = np.diag([exp1, exp2, exp3, exp4])
    P[0] = -alpha*p/(k*rho*Vp2)
    P[3] = +alpha*p*(1.0-(Vpt2-2.0*Vst2)/Vp2)

    return [C,P,E]
    