import numpy as np
from scipy import special

eps = 1e-14 #numerical constant

class Penny():
    """
    Class used to represent a penny-shaped crack using the Fialko (2001) model.

    Attributes
    ----------
    parameters : array
        names for the parameters in the model
    """
    def __init__(self):
        self.set_parnames()

    def print_model(self, x):
        """
        The function prints the parameters for the model.

        Parameters:
           x (list): Parameters for the model.
        """
        print("Penny-shaped Crack:")
        print("\tx  = %f (m)" % x[0])
        print("\ty  = %f (m)" % x[1])
        print("\td  = %f (m)" % x[2])
        print("\tP_G= %f (m)" % x[3])
        print("\ta  = %f (m)" % x[4])

    def set_parnames(self):
        """
        Function defining the names for the parameters in the model.
        """
        self.parameters=("xcen","ycen","depth","pressure","radius")

    # =====================
    # Forward Models
    # =====================

    def model(self,x,y,xcen,ycen,depth,dP_mu,radius,nu=0.25):
        """
        Calculates surface deformation based on pressurized penny-shaped crack
        References: Fialko

        Parameters:
            x: x-coordinate grid (m)
            y: y-coordinate grid (m)
            xcen: x-offset of penny-shaped crack (m)
            ycen: y-offset of penny-shaped crack (m)
            depth: depth to penny-shaped crack (m)
            dP_mu: change in pressure (in terms of mu if mu=1 if not unit is Pa)
            radius: radius for penny shaped crack (m)
            mu: shear modulus for medium (Pa) (default 1)
            nu: poisson's ratio for medium

        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
        """

        nans=np.array([x*0+1e6,x*0+1e6,x*0+1e6])
        if np.sum(depth<=0)>0 or radius<=0:
            return nans
        rd=np.copy(radius)

        P_G = dP_mu

        # Center coordinate grid on point source, normalized by radius
        x = (x - xcen) / rd
        y = (y - ycen) / rd
        z = (0 - depth)    / rd

        eps=1e-8

        h  = depth / rd
        r  = np.sqrt(x ** 2 + y ** 2)

        csi1,w1 = self.gauleg(eps,10,41)
        csi2,w2 = self.gauleg(10,60,41)
        csi     = np.concatenate((csi1,csi2))
        wcsi    = np.concatenate((w1,w2))

        if csi.shape[0] == 1:
            csi=csi.T

        phi1,psi1,t,wt=self.psi_phi(h)

        phi=np.matmul(np.sin(np.outer(csi,t)) , phi1*wt)
        psi=np.matmul(np.divide(np.sin(np.outer(csi,t)), np.outer(csi,t)) - np.cos(np.outer(csi,t)),psi1*wt)
        a=csi * h
        A=np.exp((-1)*a)*(a*psi + (1 + a)*phi)
        B=np.exp((-1)*a)*((1-a)*psi - a*phi)
        Uz=np.zeros(r.shape)
        Ur=np.zeros(r.shape)

        for i in range(r.size):
            J0=special.jv(0,r[i] * csi)
            Uzi=J0*(((1-2*nu)*B - csi*(z+h)*A)*np.sinh(csi*(z+h)) +(2*(1-nu)*A - csi*(z+h)*B)*np.cosh(csi*(z+h)))
            Uz[i]=np.dot(wcsi , Uzi)
            J1=special.jv(1,r[i] * csi)
            Uri=J1*(((1-2*nu)*A + csi*(z+h)*B)*np.sinh(csi*(z+h)) + (2*(1-nu)*B + csi*(z+h)*A)*np.cosh(csi*(z+h)))
            Ur[i]=np.dot(wcsi , Uri)

        ux = rd * P_G * Ur*x / r
        uy = rd * P_G * Ur*y / r
        uz = -rd * P_G * Uz

        return np.array([ux,uy,uz])

    def model_depth(self,x,y,z,xcen,ycen,depth,dP_mu,radius,mu=1,nu=0.25):
        """
        Calculates deformation at depth based on pressurized penny-shaped crack
        References: Fialko

        Parameters:
            x: x-coordinate grid (m)
            y: y-coordinate grid (m)
            z: z-coordinate grid (m)
            xcen: x-offset of penny-shaped crack (m)
            ycen: y-offset of penny-shaped crack (m)
            depth: depth to penny-shaped crack (m)
            dP_mu: change in pressure (in terms of mu if mu=1 if not unit is Pa)
            radius: radius for penny shaped crack (m)
            mu: shear modulus (Pa)
            nu: poisson's ratio for medium

        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
        """
        P_G=dP_mu/mu

        #
        rd=np.copy(radius)

        # Center coordinate grid on point source, normalized by radius
        x = (x - xcen) / rd
        y = (y - ycen) / rd
        z = (0 - (depth-z)) / rd

        eps=1e-8

        h  = depth / rd
        r  = np.sqrt(x ** 2 + y ** 2)

        csi1,w1 = self.gauleg(eps,10,41)
        csi2,w2 = self.gauleg(10,60,41)
        csi     = np.concatenate((csi1,csi2))
        wcsi    = np.concatenate((w1,w2))

        if csi.shape[0] == 1:
            csi=csi.T

        phi1,psi1,t,wt=self.psi_phi(h)

        phi=np.matmul(np.sin(np.outer(csi,t)) , phi1*wt)
        psi=np.matmul(np.divide(np.sin(np.outer(csi,t)), np.outer(csi,t)) - np.cos(np.outer(csi,t)),psi1*wt)
        a=csi * h
        A=np.exp((-1)*a)*(a*psi + (1 + a)*phi)
        B=np.exp((-1)*a)*((1-a)*psi - a*phi)
        Uz=np.zeros(r.shape)
        Ur=np.zeros(r.shape)

        for i in range(r.size):
            J0=special.jv(0,r[i] * csi)
            Uzi=J0*(((1-2*nu)*B - csi*(z+h)*A)*np.sinh(csi*(z+h)) +(2*(1-nu)*A - csi*(z+h)*B)*np.cosh(csi*(z+h)))
            Uz[i]=np.dot(wcsi , Uzi)
            J1=special.jv(1,r[i] * csi)
            Uri=J1*(((1-2*nu)*A + csi*(z+h)*B)*np.sinh(csi*(z+h)) + (2*(1-nu)*B + csi*(z+h)*A)*np.cosh(csi*(z+h)))
            Ur[i]=np.dot(wcsi , Uri)

        ux = rd * P_G * Ur*x / r
        uy = rd * P_G * Ur*y / r
        uz = -rd * P_G * Uz

        return np.array([ux,uy,uz])

    def psi_phi(self,h):
        """
        Auxiliary function for the Fialko (2001) model
        """
        t,w=self.gauleg(0,1,41)
        t=np.array(t)
        g=-2.0*t/np.pi
        d=np.concatenate((g,np.zeros(g.size)))
        T1,T2,T3,T4=self.giveT(h,t,t)
        T1p=np.zeros(T1.shape)
        T2p=np.zeros(T1.shape)
        T3p=np.zeros(T1.shape)
        T4p=np.zeros(T1.shape)
        N=t.size
        for j in range(N):
            T1p[:,j]=w[j]*T1[:,j]
            T2p[:,j]=w[j]*T2[:,j]
            T3p[:,j]=w[j]*T3[:,j]
            T4p[:,j]=w[j]*T4[:,j]
        M1=np.concatenate((T1p,T3p),axis=1)
        M2=np.concatenate((T4p,T2p),axis=1)
        Kp=np.concatenate((M1,M2),axis=0)
        y=np.matmul(np.linalg.inv(np.eye(2*N,2*N)-(2/np.pi)*Kp),d)
        phi=y[0:N]
        psi=y[N:2*N]
        return phi,psi,t,w

    def giveP(self,h,x):
        """
        Auxiliary function for the Fialko (2001) model
        """
        P=np.zeros((4,x.size))
        P[0]=(12*np.power(h,2)-np.power(x,2))/np.power((4*np.power(h,2)+np.power(x,2)),3)
        P[1] = np.log(4*np.power(h,2)+np.power(x,2)) + (8*np.power(h,4)+2*np.power(x,2)*np.power(h,2)-np.power(x,4))/np.power(4*np.power(h,2)+np.power(x,2),2)
        P[2] = 2*(8*np.power(h,4)-2*np.power(x,2)*np.power(h,2)+np.power(x,4))/np.power(4*np.power(h,2)+np.power(x,2),3)
        P[3] = (4*np.power(h,2)-np.power(x,2))/np.power((4*np.power(h,2)+np.power(x,2)),2)
        return P

    def giveT(self,h,t,r):
        """
        Auxiliary function for the Fialko (2001) model
        """
        M = t.size
        N = r.size
        T1 = np.zeros((M,N)) 
        T2 = np.zeros((M,N)) 
        T3 = np.zeros((M,N))
        for i in range(M):
            Pm=self.giveP(h,t[i]-r)
            Pp=self.giveP(h,t[i]+r)
            T1[i] = 4*np.power(h,3)*(Pm[0,:]-Pp[0,:])
            T2[i] = (h/(t[i]*r))*(Pm[1,:]-Pp[1,:]) +h*(Pm[2,:]+Pp[2,:])
            T3[i] = (np.power(h,2)/r)*(Pm[3,:]-Pp[3,:]-2*r*((t[i]-r)*Pm[0,:]+(t[i]+r)*Pp[0,:]))
        T4=np.copy(T3.T)
        return T1,T2,T3,T4

    def gauleg(self,a,b,n):
        """
        Auxiliary function for the Fialko (2001) model
        """
        xs, cs = self.gauleg_params1(n)
        coeffp = 0.5*(b+a) 
        coeffm = 0.5*(b-a)
        ts = coeffp - coeffm*xs
        ws=cs*coeffm
        #contribs = cs*f(ts)
        #return coeffm*np.sum(contribs)
        return ts[::-1],ws

    def gauleg_params1(self,n):
        """
        Auxiliary function for the Fialko (2001) model
        """
        xs,cs=np.polynomial.legendre.leggauss(n)
        return xs,cs


class Yang():
    """
    Class that represents a pressurized well following the implementation from Yang et., al. (1988).

    Attributes
    ----------
    parameters : array
        names for the parameters in the model
    """
    def __init__(self):
        self.set_parnames()

    def print_model(self, x):
        """
        The function prints the parameters for the model.

        Parameters:
           x (list): Parameters for the model.
        """
        print("Yang")
        print("\txcen = %f" % x[0])
        print("\tycen = %f" % x[1])
        print("\tz0 = %f" % x[2])
        print("\tP= %f" % x[3])
        print("\ta = %f" % x[4])
        print("\tb = %f" % x[5])
        print("\tphi = %f" % x[6])
        print("\ttheta= %f" % x[7])
        print("\tmu = %f" % x[8])

    def set_parnames(self):
        """
        Function defining the names for the parameters in the model.
        """
        self.parameters=("xcen","ycen","depth","pressure","a","b","az","dip")

    # =====================
    # Forward Models
    # =====================

    def model(self,x,y,xcen=0,ycen=0,z0=5e3,P=1e-3,a=2,b=1,phi=0,theta=0,nu=0.25,mu=1):
        """
        Computes surface deformation due to a prolate spheroid pressurized chamber in elastic half-space
        Yang et al., vol 93, JGR, 4249-4257, 1988)

        Parameters:
            x: x-coordinate (m)
            y: y-coordinate (m)
            xcen: source easting epicenter (m)
            ycen: source northing epicenter (m)
            z0: source depth (m)
            P: excess pressure (Pa)
            a: major axis [km]
            b: minor axis [km]
            phi: strike (degrees clockwise from north)
            theta: dip (degrees from horizontal)
            mu: normalized shear modulus (Pa)
            nu: poisson ratio (unitless)

        Returns:
            ux: deformation in the x-axis (m)
            uy: deformation in the y-axis (m)
            uz: deformation in the z-axis (m)
        """

        d_crita = a * np.sin(np.deg2rad(theta))
        d_critb = b * np.cos(np.deg2rad(theta))

        # TODO suggested fix, check if works
        if False:
            nans=np.array([x*0+1e6,x*0+1e6,x*0+1e6])
        else:
            nans = (np.full_like(x, np.nan),
                    np.full_like(x, np.nan),
                    np.full_like(x, np.nan))

        if d_crita>=z0 or b>=z0 or a<=0 or b<=0 or a<b:
            return nans
        if theta==0:
            theta=0.1
        elif theta==90:
            theta=89.9

        tp = 0 #topography vector?
        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)

        # Store some commonly used parameters (material properties)
        coeffs = np.zeros(3)
        coeffs[0] = 1 / (16 * mu * (1 - nu))
        coeffs[1] = 3 - 4 * nu
        coeffs[2] = 4 * (1 - nu) * (1 - 2 * nu)
        # Elastic constant array
        matrl = np.array([mu,mu,nu])

        # Geometery
        e_theta = np.zeros(2)
        e_theta[0] = np.sin(theta)
        e_theta[1] = np.cos(theta)
        cosp = np.cos(phi)
        sinp = np.sin(phi)
        c = np.sqrt(a ** 2 - b ** 2)

        xn = x - xcen
        yn = y - ycen

        # Call spheroid function to get geometric paramters
        # NOTE: had to add file name
        sph = Yang.spheroid(a,b,c,matrl,phi,theta,P)

        # Rotate points
        xp = xn * cosp - yn * sinp
        yp = yn * cosp + xn * sinp

        # Run forward model ?why called twice?, for each side of model...
        xi = c
        Up1,Up2,Up3 = Yang.yang(sph,xi,z0,xp,yp,0,matrl,e_theta,coeffs,tp)
        xi = -xi
        Um1,Um2,Um3 = Yang.yang(sph,xi,z0,xp,yp,0,matrl,e_theta,coeffs,tp)

        # Sum
        U1r = -Up1 + Um1
        U2r = -Up2 + Um2

        # Rotate horiz. displacements back to the orig. coordinate system
        U1 = U1r * cosp + U2r * sinp
        U2 = -U1r * sinp + U2r * cosp
        U3 = Up3 - Um3

        return U1,U2,U3

    def model_depth(self,x,y,z,xcen=0,ycen=0,z0=5e3,P=1e-3,a=2,b=1,phi=0,theta=0,nu=0.25,mu=1):
        """
        Computes deformation at depth due to a prolate spheroid pressurized chamber in elastic half-space
        Yang et al., vol 93, JGR, 4249-4257, 1988)

        Parameters:
            x: x-coordinate (m)
            y: y-coordinate (m)
            z: depth-coordinate (m)
            xcen: source easting epicenter (m)
            ycen: source northing epicenter (m)
            z0: source depth (m)
            P: excess pressure (Pa)
            a: major axis [km]
            b: minor axis [km]
            phi: strike (degrees clockwise from north)
            theta: dip (degrees from horizontal)
            mu: normalized shear modulus (Pa)
            nu: poisson ratio (unitless)

        Returns:
            ux: deformation in the x-axis (m)
            uy: deformation in the y-axis (m)
            uz: deformation in the z-axis (m)
        """
        if theta==0:
            theta=0.1
        elif theta==90:
            theta=89.9

        tp = 0 #topography vector?
        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)

        # Store some commonly used parameters (material properties)
        coeffs = np.zeros(3)
        coeffs[0] = 1 / (16 * mu * (1 - nu))
        coeffs[1] = 3 - 4 * nu
        coeffs[2] = 4 * (1 - nu) * (1 - 2 * nu)
        # Elastic constant array
        matrl = np.array([mu,mu,nu])

        # Geometery
        e_theta = np.zeros(2)
        e_theta[0] = np.sin(theta)
        e_theta[1] = np.cos(theta)
        cosp = np.cos(phi)
        sinp = np.sin(phi)
        c = np.sqrt(a ** 2 - b ** 2)

        xn = x - xcen
        yn = y - ycen

        # Call spheroid function to get geometric paramters
        # NOTE: had to add file name
        sph = Yang.spheroid(a,b,c,matrl,phi,theta,P)

        # Rotate points
        xp = xn * cosp - yn * sinp
        yp = yn * cosp + xn * sinp

        # Run forward model ?why called twice?, for each side of model...
        xi = c
        Up1,Up2,Up3 = Yang.yang(sph,xi,z0,xp,yp,z,matrl,e_theta,coeffs,tp)
        xi = -xi
        Um1,Um2,Um3 = Yang.yang(sph,xi,z0,xp,yp,z,matrl,e_theta,coeffs,tp)

        # Sum
        U1r = -Up1 + Um1
        U2r = -Up2 + Um2

        # Rotate horiz. displacements back to the orig. coordinate system
        U1 = U1r * cosp + U2r * sinp
        U2 = -U1r * sinp + U2r * cosp
        U3 = Up3 - Um3

        return U1,U2,U3

    def spheroid(a,b,c,matrl,phi,theta,P):
        '''
        Geometry used in yang pressure source computation
        '''
        pi = np.pi
        lamda = matrl[0]
        mu = matrl[1]
        nu = matrl[2]

        ac = (a - c) / (a + c)
        L1 = np.log(ac)
        iia = 2 / a / c ** 2 + L1 / c ** 3
        iiaa = 2 / 3 / a ** 3 / c ** 2 + 2 / a / c ** 4 + L1 / c ** 5
        coef1 = -2 * pi * a * b ** 2
        Ia = coef1 * iia
        Iaa = coef1 * iiaa

        u = 8 * pi * (1 - nu)
        Q = 3 / u
        R = (1 - 2 * nu) / u

        a11 = 2 * R * (Ia - 4 * pi)
        a12 = -2 * R * (Ia + 4 * pi)
        a21 = Q * a ** 2 * Iaa + R * Ia - 1
        a22 = -(Q * a ** 2 * Iaa + Ia * (2 * R - Q))

        coef2 = 3 * lamda + 2 * mu
        w = 1 / (a11 * a22 - a12 * a21)
        e11 = (3 * a22 - a12) * P * w / coef2
        e22 = (a11 - 3 * a21) * P * w / coef2

        Pdila = 2 * mu * (e11 - e22)
        Pstar = lamda * e11 + 2 * (lamda + mu) * e22
        a1 = -2 * b ** 2 * Pdila
        b1 = 3 * b ** 2 * Pdila / c ** 2 + 2 * (1 - 2 * nu) * Pstar

        sph = np.zeros(10)
        sph[0] = a
        sph[1] = b
        sph[2] = c
        sph[3] = phi
        sph[4] = theta
        sph[5] = Pstar
        sph[6] = Pdila
        sph[7] = a1
        sph[8] = b1
        sph[9] = P
        return sph

    def yang(sph,xi,z0,x,y,z,matrl,e_theta,coeffs,tp):
        """
        Auxiliary function for spheroidal model
        """
        pi = np.pi

        # Load required spheroid parameters
        a = sph[0]
        b = sph[1]
        c = sph[2]
        #phi=sph[3]
        #theta=sph[4]
        #Pstar=sph[5]
        Pdila = sph[6]
        a1 = sph[7]
        b1 = sph[8]
        #P=sph[9]

        sinth = e_theta[0]
        costh = e_theta[1]

        # Poisson's ratio, Young's modulus, and the Lame coeffiecents mu and lamda
        nu = matrl[2]
        nu4 = coeffs[1]
        #nu2=1 - 2 * nu
        nu1 = 1 - nu
        coeff = a * b ** 2 / c ** 3 * coeffs[0]

        # Introduce new coordinates and parameters (Yang et al., 1988, page 4251):
        xi2 = xi * costh
        xi3 = xi * sinth
        y0 = 0
        z00 = tp + z0
        x1 = x
        x2 = y - y0
        x3 = z - z00
        xbar3 = z + z00
        y1 = x1
        y2 = x2 - xi2
        y3 = x3 - xi3
        ybar3 = xbar3 + xi3
        r2 = x2 * sinth - x3 * costh
        q2 = x2 * sinth + xbar3 * costh
        r3 = x2 * costh + x3 * sinth
        q3 = -x2 * costh + xbar3 * sinth
        rbar3 = r3 - xi
        qbar3 = q3 + xi
        R1 = (y1 ** 2 + y2 ** 2 + y3 ** 2) ** (0.5)
        R2 = (y1 ** 2 + y2 ** 2 + ybar3 ** 2) ** (0.5)

        #C0 = y0 * costh + z00 * sinth # check this?
        C0=z0/sinth

        betatop = (costh * q2 + (1 + sinth) * (R2 + qbar3))
        betabottom = costh * y1
        # Strange replacement for matlab 'find'
        #nz=np.flatnonzero(np.abs(betabottom) != 0) #1D index
        nz  = (np.abs(betabottom) != 0) # 2D index
        atnbeta = pi / 2 * np.sign(betatop)
        # change atan --> arctan, and -1 not needed for 2D boolean index
        #atnbeta[(nz-1)]=np.atan(betatop[(nz-1)] / betabottom[(nz-1)])
        atnbeta[nz] = np.arctan(betatop[nz] / betabottom[nz])

        # Set up other parameters for dipping spheroid (Yang et al., 1988, page 4252):
        # precalculate some repeatedly used natural logs:
        Rr = R1 + rbar3
        Rq = R2 + qbar3
        Ry = R2 + ybar3
        lRr = np.log(Rr)
        lRq = np.log(Rq)
        lRy = np.log(Ry)

        # Note: dot products should in fact be element-wise multiplication
        #A1star=a1 / (R1.dot(Rr)) + b1 * (lRr + (r3 + xi) / Rr)
        #Abar1star=- a1 / (R2.dot(Rq)) - b1 * (lRq + (q3 - xi) / Rq)
        A1star =  a1 / (R1*Rr) + b1*(lRr + (r3 + xi)/Rr)
        Abar1star = -a1 / (R2*Rq) - b1*(lRq + (q3 - xi)/Rq)
        A1 = xi / R1 + lRr
        Abar1 = xi / R2 - lRq
        #A2=R1 - r3.dot(lRr)
        #Abar2=R2 - q3.dot(lRq)
        A2 = R1 - r3*lRr
        Abar2 = R2 - q3*lRq
        A3 = xi * rbar3 / R1 + R1
        Abar3 = xi * qbar3 / R2 - R2

        #B=xi * (xi + C0) / R2 - Abar2 - C0.dot(lRq)
        B = xi * (xi + C0)/R2 - Abar2 - C0*lRq
        Bstar = a1 / R1 + 2 * b1 * A2 + coeffs[1] * (a1 / R2 + 2 * b1 * Abar2)
        F1 = 0
        F1star = 0
        F2 = 0
        F2star = 0


        if z != 0:
            F1 = (-2*sinth*z* (xi*(xi+C0)/R2**3 +
                              (R2+xi+C0)/(R2*(Rq)) +
                              4*(1-nu)*(R2+xi)/(R2*(Rq))
                              )
                 )

            F1star = (2*z*(costh*q2*(a1*(2*Rq)/(R2**3*(Rq)**2) - b1*(R2 + 2*xi)/(R2*(Rq)**2)) +
                           sinth*(a1/R2**3 -2*b1*(R2 + xi)/(R2* (Rq)))
                           )
                     )

            F2 = -2*sinth*z*(xi*(xi+C0)*qbar3/R2**3 + C0/R2 + (5-4*nu)*Abar1)

            F2star = 2*z*(a1*ybar3/R2**3 - 2*b1*(sinth*Abar1 + costh*q2*(R2+xi)/(R2*Rq)))


        # Calculate little f's
        ff1 = (xi*y1/Ry +
               3/(costh)**2*(y1*lRy*sinth -y1*lRq + 2*q2*atnbeta) +
               2*y1*lRq -
               4*xbar3*atnbeta/costh
              )

        ff2 = (xi*y2/Ry +
               3/(costh)**2*(q2*lRq*sinth - q2*lRy + 2*y1*atnbeta*sinth + costh*(R2-ybar3)) -
               2*costh*Abar2 +
               2/costh*(xbar3*lRy - q3*lRq)
              )

        ff3 = ((q2*lRq - q2*lRy*sinth + 2*y1*atnbeta)/costh +
                2*sinth*Abar2 + q3*lRy - xi
              )


        # Assemble into x, y, z displacements (1,2,3):
        u1 = coeff*(A1star + nu4*Abar1star + F1star)*y1

        #u2 = coeff*(sinth*(A1star*r2+(nu4*Abar1star+F1star)*q2) +
        #            costh*(Bstar-F2star) + 2*sinth*costh*z*Abar1star)
        u2 = coeff*(sinth*(A1star*r2+(nu4*Abar1star+F1star)*q2) +
                    costh*(Bstar-F2star))
        #u3 = coeff*(-costh*(Abar1star*r2+(nu4*Abar1star-F1star)*q2) +
        #            sinth*(Bstar+F2star) + 2*(costh)**2*z*Abar1star)
        u3 = coeff*(-costh*(A1star*r2+(nu4*Abar1star-F1star)*q2) +
                    sinth*(Bstar+F2star))


        u1 = u1 + 2*coeff*Pdila*((A1 + nu4*Abar1 + F1)*y1 - coeffs[2]*ff1)

        u2 = u2 + 2*coeff*Pdila*(sinth*(A1*r2+(nu4*Abar1+F1)*q2) -
                                 coeffs[2]*ff2 + 4*nu1*costh*(A2+Abar2) +
                                 costh*(A3 - nu4*Abar3 - F2))

        u3 = u3 + 2*coeff*Pdila*(costh*(-A1*r2 + (nu4*Abar1 + F1)*q2) + coeffs[2]*ff3 +
                                 4*nu1*sinth*(A2+Abar2) +
                                 sinth*(A3 + nu4*Abar3 + F2 - 2*nu4*B))

        return u1,u2,u3


class Okada():
    """
    A class used to represent a tensile or slip dislocation with Okada (1985) model.

    Attributes
    ----------
    type: str
        describes if dislocation is tensile (open) or slip (slip)

    parameters : array
        names for the parameters in the model
    """
    def __init__(self):
        self.set_parnames()

    def set_type(self,typ):
        """
        Defines the type of dislocation.

        Parameters:
            str: 'slip' to represent faults or 'open' to represent sill/dikes.
        """
        self.type=typ

    def get_source_id(self):
        """
        The function defining the name for the model.

        Returns:
            str: Name of the model.
        """
        return "Okada"

    def print_model(self, x):
        """
        The function prints the parameters for the model.

        Parameters:
           x (list): Parameters for the model.
        """
        print("Okada")
        if self.type=='slip':
            print("\txcen = %f" % x[0])
            print("\tycen = %f" % x[1])
            print("\tdepth = %f" % x[2])
            print("\tlength= %f" % x[3])
            print("\twidth = %f" % x[4])
            print("\tslip = %f" % x[5])
            print("\tstrike= %f" % x[6])
            print("\tdip = %f" % x[7])
            print("\trake = %f" % x[8])
        elif self.type=='open':
            print("\txcen = %f" % x[0])
            print("\tycen = %f" % x[1])
            print("\tdepth = %f" % x[2])
            print("\tlength= %f" % x[3])
            print("\twidth = %f" % x[4])
            print("\topening = %f" % x[5])
            print("\tstrike= %f" % x[6])
            print("\tdip = %f" % x[7])

    def set_parnames(self):
        """
        Function defining the names for the parameters in the model.
        """
        if self.type=='slip':
            self.parameters=("xcen","ycen","depth","length","width","slip","strike","dip","rake")
        elif self.type=='open':
            self.parameters=("xcen","ycen","depth","length","width","opening","strike","dip")

    def get_args(self,args, tilt):
        """
        Function that arranges the parameters for the dislocation model depending on the type.

        Parameters:
            args (list) : parameters given by the user
            tilt (boolean) : compute tilt displacements (True) or 3d displacements (False)

        Returns:
            rargs (list) : parameters for the Okada model.
        """
        nu=0.25
        if self.type=='slip':
            xcen,ycen,depth,length,width,slip,strike,dip,rake=args
            opening=0.0
        else:
            xcen,ycen,depth,length,width,opening,strike,dip=args
            slip=0.0
            rake=0.0
        rargs=[xcen, ycen,depth, length, width,slip, opening,strike, dip, rake,nu, tilt]
        return rargs

    # =====================
    # Forward Models
    # =====================
    def model(self,x,y,*args):
        """
        3d displacement field on surface for dislocation (Okada, 1985)

        Parameters:
            args (list) : parameters given by the user

        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
        """
        rargs=self.get_args(args,tilt=False)
        return self.model_gen(x,y, *rargs)

    def model_tilt(self,x,y,*args):
        """
        Tilt displacement field on surface for dislocation (Okada, 1985)

        Parameters:
            args (list) : parameters given by the user

        Returns:
            dx (array) : tilt displacements in the x-axis (radians).
            dy (array) : tilt displacements in the y-axis (radians).
        """
        rargs=self.get_args(args,tilt=True)
        return self.model_gen(x,y, *rargs)

    def model_gen(self,x,y, xcen=0, ycen=0,
                        depth=5e3, length=1e3, width=1e3,
                        slip=0.0, opening=10.0,
                        strike=0.0, dip=0.0, rake=0.0,
                        nu=0.25,tilt=False):
        """
        Computes tilt or 3d displacement field on surface for dislocation (Okada, 1985)

        Parameters:
            x: x-coordinate for displacement (m)
            y: y-coordinate for displacement (m)
            xcen: x-offset of dislocation center (m)
            ycen: y-offset of dislocation center (m)
            depth: depth to dislocation center (m)
            length: length of dislocation path (m)
            width: width of dislocation path (m)
            slip: fault movement (m)
            opening: amount of closing or opening of sill/dike (m)
            strike: horizontal clock wise orientation from north of dislocation (degrees)
            dip: dipping angle of dislocation (degrees)
            rake: fault's angle of rupture where 0 represents a strike-slip fault and 90 represents a normal fault (degrees)
            nu: Poisson's ratio
            tilt: boolean to indicate calculation of tilt displacements (True) or 3d displacements (False) (default False)

        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
        """
        e = x - xcen
        n = y - ycen

        # A few basic parameter checks
        if not (0.0 <= strike <= 360.0) or not (0 <= dip <= 90):
            print('Strike',strike)
            print('Dip',dip)
            print('Please use 0<strike<360 clockwise from North')
            print('And 0<dip<90 East of strike convention')
            raise ValueError

        # Don't allow faults that prech the surface
        d_crit = width/2 * np.sin(np.deg2rad(dip))

        if tilt:
            nans=np.array([e*0+1e6,e*0+1e6])
        else:
            nans=np.array([e*0+1e6,e*0+1e6,e*0+1e6])

        if np.sum(depth<=0)>0:
            return nans
        if np.sum(depth<d_crit)>0:
            return nans
        elif length<0:
            return nans
        elif width<0:
            return nans
        elif rake>180:
            return nans
        elif not -1.0 <= nu <= 0.5:
            return nans

        #assert depth >= d_crit, 'depth must be greater than {}'.format(d_crit)
        #assert length >=0, 'fault length must be positive'
        #assert width >=0, 'fault length must be positive'
        #assert rake <= 180, 'rake should be:  rake <= 180'
        #assert -1.0 <= nu <= 0.5, 'Poisson ratio should be: -1 <= nu <= 0.5'

        strike = np.deg2rad(strike) #transformations accounted for below
        dip    = np.deg2rad(dip)
        rake   = np.deg2rad(rake)

        L = length
        W = width

        U1 = np.cos(rake) * slip
        U2 = np.sin(rake) * slip
        U3 = opening

        d = depth + np.sin(dip) * W / 2 #fault top edge
        ec = e + np.cos(strike) * np.cos(dip) * W / 2
        nc = n - np.sin(strike) * np.cos(dip) * W / 2
        x = np.cos(strike) * nc + np.sin(strike) * ec + L / 2
        y = np.sin(strike) * nc - np.cos(strike) * ec + np.cos(dip) * W
        p = y * np.cos(dip) + d * np.sin(dip)
        q = y * np.sin(dip) - d * np.cos(dip)

        if tilt:
            ssx=Okada.dx_ss
            dsx=Okada.dx_ds
            tfx=Okada.dx_tf
            
            ssy=Okada.dy_ss
            dsy=Okada.dy_ds
            tfy=Okada.dy_tf
        else:
            ssx=Okada.ux_ss
            dsx=Okada.ux_ds
            tfx=Okada.ux_tf
            
            ssy=Okada.uy_ss
            dsy=Okada.uy_ds
            tfy=Okada.uy_tf
            
            uz = - U1 / (2 * np.pi) * Okada.chinnery(Okada.uz_ss, x, p, L, W, q, dip, nu) - \
                   U2 / (2 * np.pi) * Okada.chinnery(Okada.uz_ds, x, p, L, W, q, dip, nu) + \
                   U3 / (2 * np.pi) * Okada.chinnery(Okada.uz_tf, x, p, L, W, q, dip, nu)
            
        ux = - U1 / (2 * np.pi) * Okada.chinnery(ssx, x, p, L, W, q, dip, nu) - \
               U2 / (2 * np.pi) * Okada.chinnery(dsx, x, p, L, W, q, dip, nu) + \
               U3 / (2 * np.pi) * Okada.chinnery(tfx, x, p, L, W, q, dip, nu)

        uy = - U1 / (2 * np.pi) * Okada.chinnery(ssy, x, p, L, W, q, dip, nu) - \
               U2 / (2 * np.pi) * Okada.chinnery(dsy, x, p, L, W, q, dip, nu) + \
               U3 / (2 * np.pi) * Okada.chinnery(tfy, x, p, L, W, q, dip, nu)


        ue = np.sin(strike) * ux - np.cos(strike) * uy
        un = np.cos(strike) * ux + np.sin(strike) * uy
        
        if tilt:
            return ue,un
        else:
            return ue,un,uz

    def chinnery(f, x, p, L, W, q, dip, nu):
        """
        Chinnery's notation [equation (24) p. 1143]
        """
        u =  (f(x, p, q, dip, nu) -
              f(x, p - W, q, dip, nu) -
              f(x - L, p, q, dip, nu) +
              f(x - L, p - W, q, dip, nu))

        return u


    def ux_ss(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = xi * q / (R * (R + eta)) + \
            Okada.I1(xi, eta, q, dip, nu, R) * np.sin(dip)
        k = (q != 0)
        #u[k] = u[k] + np.arctan2( xi[k] * (eta[k]) , (q[k] * (R[k])))
        u[k] = u[k] + np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]) )
        return u


    def uy_ss(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + eta)) + \
            q * np.cos(dip) / (R + eta) + \
            Okada.I2(eta, q, dip, nu, R) * np.sin(dip)
        return u


    def uz_ss(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        u = db * q / (R * (R + eta)) + \
            q * np.sin(dip) / (R + eta) + \
            Okada.I4(db, eta, q, dip, nu, R) * np.sin(dip)
        return u

    def dx_ss(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = -xi * q**2 * Okada.A(eta,R) * np.cos(dip) + \
            (xi * q / R**3 - Okada.K1(xi, eta, q, dip, nu, R)) * np.sin(dip)
        return u

    def dy_ss(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        yb = eta * np.cos(dip) + q * np.sin(dip)

        u = db * q / R**3 * np.cos(dip) + \
            (xi**2 * q * Okada.A(eta,R) * np.cos(dip) - np.sin(dip) / R + yb * q / R**3 - Okada.K2(xi, eta, q, dip, nu, R)) * np.sin(dip)
        return u

    def ux_ds(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = q / R - \
            Okada.I3(eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip)
        return u

    def uy_ds(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = ( (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + xi)) -
               Okada.I1(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip) )
        k = (q != 0)
        u[k] = u[k] + np.cos(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]))
        return u

    def uz_ds(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        u = ( db * q / (R * (R + xi)) -
              Okada.I5(xi, eta, q, dip, nu, R, db) * np.sin(dip) * np.cos(dip) )
        k = (q != 0)
        #u[k] = u[k] + np.sin(dip) * np.arctan2(xi[k] * eta[k] , q[k] * R[k])
        u[k] = u[k] + np.sin(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]))
        return u

    def dx_ds(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(dip) - q * np.cos(dip)

        u = db * q / R**3 + \
            q * np.sin(dip) / (R * (R + eta)) + \
            Okada.K3(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip)

        return u

    def dy_ds(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        yb = eta * np.cos(dip) + q * np.sin(dip)

        u = yb * db * q * Okada.A(xi,R) - \
            (2 * db / (R * (R + xi)) + xi * np.sin(dip) / (R * (R + eta))) * np.sin(dip) + \
            Okada.K1(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip)

        return u

    def ux_tf(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi**2 + eta**2 + q**2)
        u = q**2 / (R * (R + eta)) - \
            (Okada.I3(eta, q, dip, nu, R) * np.sin(dip)**2)
        return u

    def uy_tf(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi**2 + eta**2 + q**2)
        u = - (eta * np.sin(dip) - q * np.cos(dip)) * q / (R * (R + xi)) - \
            (np.sin(dip) * xi * q / (R * (R + eta))) - \
            (Okada.I1(xi, eta, q, dip, nu, R) * np.sin(dip) ** 2)
        k = (q != 0)
        #u[k] = u[k] + np.sin(dip) * np.arctan2(xi[k] * eta[k] , q[k] * R[k])
        u[k] = u[k] + np.sin(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]) )
        return u

    def uz_tf(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi**2 + eta**2 + q**2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        u = (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + xi)) + \
             np.cos(dip) * xi * q / (R * (R + eta)) - \
             Okada.I5(xi, eta, q, dip, nu, R, db) * np.sin(dip)**2
        k = (q != 0)
        u[k] = u[k] - np.cos(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]) )
        return u

    def dx_tf(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi**2 + eta**2 + q**2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        yb = eta * np.cos(dip) + q * np.sin(dip)

        u = q**2 * np.sin(dip) / R**3 - q**3 * Okada.A(eta,R) * np.cos(dip) + Okada.K3(xi, eta, q, dip, nu, R) * (np.sin(dip))**2

        return u

    def dy_tf(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi**2 + eta**2 + q**2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        yb = eta * np.cos(dip) + q * np.sin(dip)

        u = (yb * np.sin(dip) + db * np.cos(dip)) * q**2 * Okada.A(xi,R) + \
            xi * q**2 * Okada.A(eta,R) * np.sin(dip) * np.cos(dip) - \
            (2 * q/(R * (R + xi)) - Okada.K1(xi, eta, q, dip, nu, R)) * (np.sin(dip))**2

        return u

    def I1(xi, eta, q, dip, nu, R):
        """
        Auxiliary function for Okada (1985) model.
        """
        db = eta * np.sin(dip) - q * np.cos(dip)
        if np.cos(dip) > eps:
            I = (1 - 2 * nu) * (- xi / (np.cos(dip) * (R + db))) - \
                np.sin(dip) / np.cos(dip) * Okada.I5(xi, eta, q, dip, nu, R, db)
        else:
            I = -(1 - 2 * nu)/2 * xi * q / (R + db)**2
        return I

    def I2(eta, q, dip, nu, R):
        """
        Auxiliary function for Okada (1985) model.
        """
        I = (1 - 2 * nu) * (-np.log(R + eta)) - \
            Okada.I3(eta, q, dip, nu, R)
        return I

    def I3(eta, q, dip, nu, R):
        """
        Auxiliary function for Okada (1985) model.
        """
        yb = eta * np.cos(dip) + q * np.sin(dip)
        db = eta * np.sin(dip) - q * np.cos(dip)
        if np.cos(dip) > eps:
            I = (1 - 2 * nu) * (yb / (np.cos(dip) * (R + db)) - np.log(R + eta)) + \
                np.sin(dip) / np.cos(dip) * Okada.I4(db, eta, q, dip, nu, R)
        else:
            I = (1 - 2 * nu) / 2 * (eta / (R + db) + yb * q / (R + db) ** 2 - np.log(R + eta))
        return I

    def I4(db, eta, q, dip, nu, R):
        """
        Auxiliary function for Okada (1985) model.
        """
        if np.cos(dip) > eps:
            I = (1 - 2 * nu) * 1.0 / np.cos(dip) * \
                (np.log(R + db) - np.sin(dip) * np.log(R + eta))
        else:
            I = - (1 - 2 * nu) * q / (R + db)
        return I

    def I5(xi, eta, q, dip, nu, R, db):
        """
        Auxiliary function for Okada (1985) model.
        """
        X = np.sqrt(xi**2 + q**2)
        if np.cos(dip) > eps:
            I = (1 - 2 * nu) * 2 / np.cos(dip) * \
                 np.arctan( (eta * (X + q*np.cos(dip)) + X*(R + X) * np.sin(dip)) /
                            (xi*(R + X) * np.cos(dip)) )
            I[xi == 0] = 0
        else:
            I = -(1 - 2 * nu) * xi * np.sin(dip) / (R + db)
        return I

    def K1(xi, eta, q, dip, nu, R):
        """
        Auxiliary function for Okada (1985) model.
        """
        db = eta * np.sin(dip) - q * np.cos(dip)
        if np.cos(dip) > eps:
            K = (1 - 2 * nu) * xi / np.cos(dip) * (1/(R * (R + db)) - np.sin(dip) / (R * (R + eta)))
        else:
            K = (1 - 2 * nu) * xi * q / (R * (R + db)**2)
        return K

    def K2(xi, eta, q, dip, nu, R):
        """
        Auxiliary function for Okada (1985) model.
        """
        K3 = Okada.K3(xi, eta, q, dip, nu, R)

        K = (1 - 2 * nu) * (-np.sin(dip) / R + q * np.cos(dip) / (R * (R + eta))) - K3

        return K

    def K3(xi, eta, q, dip, nu, R):
        """
        Auxiliary function for Okada (1985) model.
        """
        yb = eta * np.cos(dip) + q * np.sin(dip)
        db = eta * np.sin(dip) - q * np.cos(dip)
        if np.cos(dip) > eps:
            K = (1 - 2 * nu) / np.cos(dip) * ((q / R) * (1 / (R + eta)) - yb / (R * (R + db)))
        else:
            K = (1 - 2 * nu) * np.sin(dip) / (R + db) * (xi**2 / (R * (R + db)) - 1)
        return K

    def A(xieta, R):
        """
        Auxiliary function for Okada (1985) model.
        """
        A = (2 * R + xieta) / (R**3 * (R + xieta)**2)

        return A


class Mogi():
    """
    A class used to represent a point source using the Mogi (1958) model.

    Attributes
    ----------
    parameters : array
        names for the parameters in the model
    """
    def __init__(self):
        self.set_parnames()

    def get_source_id(self):
        """
        The function defining the name for the model.

        Returns:
            str: Name of the model.
        """
        return "Mogi"

    def print_model(self, x):
        """
        The function prints the parameters for the model.

        Parameters:
           x (list): Parameters for the model.
        """
        print("Mogi")
        print("\tx = %f" % x[0])
        print("\ty = %f" % x[1])
        print("\td = %f" % x[2])
        print("\tdV= %f" % x[3])

    def set_parnames(self):
        """
        Function defining the names for the parameters in the model.
        """
        self.parameters=("xcen","ycen","depth","dV")

    # =====================
    # Forward Models
    # =====================

    def model(self,x,y, xcen, ycen, depth, dVol, nu=0.25):
        """
        3d displacement field on surface from point source (Mogi, 1958)

        Parameters:
            x: x-coordinate for displacement (m)
            y: y-coordinate for displacement (m)
            xcen: x-offset of point source epicenter (m)
            ycen: y-offset of point source epicenter (m)
            d: depth to point (m)
            rad: chamber radius (m)
            dV: change in volume (m^3)
            dP: change in pressure (Pa)
            nu: poisson's ratio for medium (default 0.25)
            mu: shear modulus for medium (Pa) (default 4e9)

        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
        """
        nans=np.array([x*0+1e6,x*0+1e6,x*0+1e6])
        if depth<=0:
            return nans
        # Center coordinate grid on point source
        x = x - xcen
        y = y - ycen

        # Convert to surface cylindrical coordinates
        th, rho = cart2pol(x,y) # surface angle and radial distance
        R = np.sqrt(depth**2+rho**2)     # radial distance from source

        # Mogi displacement calculation
        C = ((1-nu) / np.pi) * dVol
        ur = C * rho / R**3    # horizontal displacement, m
        uz = C * depth / R**3      # vertical displacement, m

        ux, uy = pol2cart(th, ur)

        return ux, uy, uz #returns tuple
        #return np.array([ux,uy,uz])

    def model_tilt(self, x, y, xcen, ycen, depth, dVol, nu=0.25):
        """
        Tilt displacement field from point source (Mogi, 1958)

        Parameters:
            x: x-coordinate for displacement (m)
            y: y-coordinate for displacement (m)
            xcen: y-offset of point source epicenter (m)
            ycen: y-offset of point source epicenter (m)
            depth: depth to point (m)
            rad: chamber radius (m)
            dVol: change in volume (m^3)
            dP: change in pressure (Pa)
            nu: poisson's ratio for medium
            mu: shear modulus for medium (Pa)
            order: highest order term to include (up to 2)
            output: 'cart' (cartesian), 'cyl' (cylindrical)

        Returns:
            dx (array) : inclination in the x-axis in radians.
            dy (array) : inclination in the y-axis in radians.
        """
        nans=np.array([x*0+1e6,x*0+1e6,x*0+1e6])
        if depth<=0:
            return nans
        # Center coordinate grid on point source
        x = x - xcen
        y = y - ycen

        # Convert to surface cylindrical coordinates
        th, rho = cart2pol(x,y) # surface angle and radial distance
        R = np.sqrt(depth**2+rho**2)     # radial distance from source

        # Mogi displacement calculation
        C = ((1-nu) / np.pi) * dVol

        dx=3*C*depth*x/R**5
        dy=3*C*depth*y/R**5

        #print(dx)

        return dx, dy

    def model_depth(self, x, y, z, xcen, ycen, depth, dVol, nu=0.25, mu=1):
        """
        Internal displacements for a point source (Mindlin, 1936)

        Parameters:
            x: x-coordinate for displacement (m)
            y: y-coordinate for displacement (m)
            z: z-coordinate for displacement (m)
            xcen: y-offset of point source epicenter (m)
            ycen: y-offset of point source epicenter (m)
            depth: depth to point (m)
            rad: chamber radius (m)
            dVol: change in volume (m^3)
            dP: change in pressure (Pa)
            nu: poisson's ratio for medium
            mu: shear modulus for medium (Pa)
            order: highest order term to include (up to 2)
            output: 'cart' (cartesian), 'cyl' (cylindrical)

        Returns:
            dx (array) : inclination in the x-axis in radians.
            dy (array) : inclination in the y-axis in radians.
        """
        nans=np.array([x*0+1e6,x*0+1e6,x*0+1e6])
        xc = x - xcen
        yc = y - ycen

        z=-z

        lamb=2*mu*nu/(1-2*nu)
        alpha=(lamb+mu)/(lamb+2*mu)
        dp=depth-z
        dn=depth+z

        # Convert to surface cylindrical coordinates
        #th, rho = util.cart2pol(x,y) # surface angle and radial distance
        R = lambda dt: np.sqrt(dt**2+xc**2+yc**2)     # radial distance from source

        uah= lambda h,dt:-((1-alpha)/2)*(h/R(dt)**3)
        ubh= lambda h,dt: ((1-alpha)/alpha)*(h/R(dt)**3)
        uch= lambda h,dt: (1-alpha)*(3*h*dt)/(R(dt)**5)

        ucz= lambda dt: (1-alpha)*(1-3*(dt/R(dt))**2)/R(dt)**3

        C=(dVol/(2*np.pi))

        ux=C*(uah(xc,dp)-uah(xc,dn)+ubh(xc,dp)+z*uch(xc,dp))
        uy=C*(uah(yc,dp)-uah(yc,dn)+ubh(yc,dp)+z*uch(yc,dp))
        uz=C*(uah(dp,dp)-uah(dn,dn)+ubh(dp,dp)+z*ucz(dp))

        return ux, uy, uz

    def calc_genmax(self,t,xcen=0,ycen=0,depth=4e3,dP=100e6,a=700,nu=0.25,G=30e9,
                    mu1=0.5,eta=2e16,**kwargs):
        """ Solution for spherical source in a generalized maxwell viscoelastic
        halfspace based on Del Negro et al 2009.

        Required arguments:
        ------------------
        x: x-coordinate grid (m)
        y: y-coordinate grid (m)
        t: time (s)

        Keyword arguments:
        -----------------
        xcen: y-offset of point source epicenter (m)
        ycen: y-offset of point source epicenter (m)
        depth: depth to point (m)
        dV: change in volume (m^3)
        K: bulk modulus (constant b/c incompressible)
        E: Young's moduls
        G: total shear modulus (Gpa)
        mu0: fractional shear modulus (spring part)
        mu1: fractional shear modulus (dashpot part)
        eta: viscosity (Pa s)
        output: 'cart' (cartesian), 'cyl' (cylindrical)

        """
        #WARNING: mu0 != 0
        # center coordinate grid on point source
        x = self.get_xs() - xcen
        y = self.get_ys() - ycen

        # convert to surface cylindrical coordinates
        #th, r = cart2pol(x,y)
        r = np.hypot(x,y) #surface radial distance
        R = np.hypot(depth,r) #radial distance from source center

        # Calculate displacements
        #E = 2.0 * G * (1+nu)
        #K = E / (3.0* (1 - 2*nu)) #bulk modulus = (2/3)*E if poisson solid
        K = (2.0*G*(1+nu)) / (3*(1-(2*nu)))
        mu0 = 1.0 - mu1
        alpha = (3.0*K) + G #recurring terms
        beta = (3.0*K) + (G*mu0)

        # Maxwell times
        try:
            tau0 = eta / (G*mu1)
        except:
            tau0 = np.inf
        tau1 = (alpha / beta) * tau0
        tau2 = tau0 / mu0

        #print('relaxation times:\nT0={}\nT1={}\nT2={}'.format(tau0,tau1,tau2))

        term1 = ((3.0*K + 4*G*mu0) / (mu0*beta))
        term2 = ((3.0 * G**2 * np.exp(-t/tau1))*(1-mu0)) / (beta*alpha)
        term3 = ((1.0/mu0) - 1) * np.exp(-t/tau2)

        A = (1.0/(2*G)) * (term1 - term2 - term3)
        C = (dP * a**3) / R**3
        ur = C * A * r
        uz = C * A * depth

        return ur, uz


def cart2pol(x1,x2):
    """
    Converts cartesian coordinates to polar coordinates

    Parameters:
        x1 (float): x-coordinate (m)
        x2 (float): y-coordinate (m)

    Returns:
        theta (float): polar angle (radians)
        r (float): polar radius (m)
    """
    #theta = np.arctan(x2/x1)
    theta = np.arctan2(x2,x1) #sign matters -SH
    r = np.sqrt(x1**2 + x2**2)
    return theta, r


def pol2cart(theta,r):
    """
    Converts polar coordinates to cartesian coordinates

    Parameters:
        theta (float): polar angle (radians)
        r (float): polar radius (m)

    Returns:
        x1 (float): x-coordinate (m)
        x2 (float): y-coordinate (m)
    """
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    return x1,x2