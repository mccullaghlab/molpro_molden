import numpy as np
import time
from tqdm import tqdm, tqdm_notebook
from tqdm import trange
from numba import jit

angstromToBohr = 1.889726124565062
bohrToAngstrom = 1.0/angstromToBohr

class molpro:

    def __init__(self,moldenFileName):
        f = open(moldenFileName)
        for line in f:
            if "_ENERGY_BASIS" in line:
                data = line.split("=")
                self.basisSet = data[1].strip()
            if "[Atoms]" in line:
                self.nAtoms = 0
                self.atomPos = []
                self.atomicNum = []
                self.atomName = []
                self.atomicNumber = []
                self.nuclearCharge = []
                for line in f:
                    if "[GTO]" in line:
                        break
                    data = line.split()
                    self.atomName.append(data[0])
                    if data[0] == "H":
                        self.atomicNumber.append(1)
                        self.nuclearCharge.append(1.0)
                    elif data[0] == "He":
                        self.atomicNumber.append(2)
                        self.nuclearCharge.append(2.0)
                    elif data[0] == "Li":
                        self.atomicNumber.append(3)
                        self.nuclearCharge.append(3.0)
                    elif data[0] == "Be":
                        self.atomicNumber.append(4)
                        self.nuclearCharge.append(4.0)
                    elif data[0] == "B":
                        self.atomicNumber.append(5)
                        self.nuclearCharge.append(5.0)
                    elif data[0] == "C":
                        self.atomicNumber.append(6)
                        self.nuclearCharge.append(6.0)
                    elif data[0] == "N":
                        self.atomicNumber.append(7)
                        self.nuclearCharge.append(7.0)
                    elif data[0] == "O":
                        self.atomicNumber.append(8)
                        self.nuclearCharge.append(8.0)
                    elif data[0] == "F":
                        self.atomicNumber.append(9)
                        self.nuclearCharge.append(9.0)
                    elif data[0] == "Ne":
                        self.atomicNumber.append(10)
                        self.nuclearCharge.append(10.0)
                    self.atomicNum.append(int(data[2]))
                    self.atomPos.append(float(data[3])*angstromToBohr)
                    self.atomPos.append(float(data[4])*angstromToBohr)
                    self.atomPos.append(float(data[5])*angstromToBohr)
                    self.nAtoms += 1 
                self.atomPos = np.array(self.atomPos).reshape((self.nAtoms,3))
            if "[GTO]" in line:
                self.nBasisFunctions = 0
                self.primitivesPerShell = []
                self.contractionCoefficients = []
                self.primitiveExponents = []
                self.shellType = []
                self.shellPos = []
                self.shellNorm = []
                self.totalNorm = []
                self.shellBasisFunctionCount = []
                shell = 0
                for i in range(self.nAtoms):
                    f.readline()  # atom number
                    for line in f:
                        if not line.strip():
                            break
                        data = line.split()
                        self.shellType.append(data[0])
                        self.shellPos.append(self.atomPos[i,:])
                        self.primitivesPerShell.append(int(data[1]))
                        self.primitiveExponents.append([])
                        self.contractionCoefficients.append([])
                        self.shellBasisFunctionCount.append([])
                        for j in range(int(data[1])):
                            line = f.readline()
                            data2 = line.split()
                            # change Ds to Es in numbers
                            self.primitiveExponents[shell].append(float(data2[0].replace('D','E')))
                            self.contractionCoefficients[shell].append(float(data2[1].replace('D','E')))
                        if self.shellType[shell] == "s":
                            self.shellBasisFunctionCount[shell].append([self.nBasisFunctions+1])
                            self.nBasisFunctions += 1
                            # contraction coefficients might not be normalized
                            self.contractionCoefficients[shell] *= (2/np.pi)**0.75*np.power(self.primitiveExponents[shell],0.75)
                            # normalize entire contraction
                            self.shellNorm.append(self.s_norm(self.primitivesPerShell[shell],self.contractionCoefficients[shell],self.primitiveExponents[shell]))
                        elif self.shellType[shell] == "p":
                            self.shellBasisFunctionCount[shell].append([self.nBasisFunctions+1,self.nBasisFunctions+2,self.nBasisFunctions+3])
                            self.nBasisFunctions += 3
                            # contraction coefficients might not be normalized
                            self.contractionCoefficients[shell] *= 2*(2/np.pi)**0.75*np.power(self.primitiveExponents[shell],1.25)
                            # normalize entire contraction
                            self.shellNorm.append(self.p_norm(self.primitivesPerShell[shell],self.contractionCoefficients[shell],self.primitiveExponents[shell]))
                        elif self.shellType[shell] == "d":
                            self.shellBasisFunctionCount[shell].append([self.nBasisFunctions+1,self.nBasisFunctions+2,self.nBasisFunctions+3,self.nBasisFunctions+4,self.nBasisFunctions+5,self.nBasisFunctions+6])
                            self.nBasisFunctions += 6
                            self.shellNorm.append(self.d6_norm(self.contractionCoefficients[shell],self.primitiveExponents[shell]))
                        elif self.shellType[shell] == "f":
                            self.shellBasisFunctionCount[shell].append([self.nBasisFunctions+1,self.nBasisFunctions+2,self.nBasisFunctions+3,self.nBasisFunctions+4,self.nBasisFunctions+5,self.nBasisFunctions+6,self.nBasisFunctions+7,self.nBasisFunctions+8,self.nBasisFunctions+9,self.nBasisFunctions+10])
                            self.nBasisFunctions += 10 
                            self.shellNorm.append(self.f10_norm(self.contractionCoefficients[shell],self.primitiveExponents[shell]))
                        for elem in self.shellNorm[shell]:
                            self.totalNorm.append(elem)
                        shell += 1
            if "[MO]" in line:
                MOCoeff = []
                self.occupancy = []
                self.orbitalEnergy = []
                self.nMOs = 0

                for i in range(self.nBasisFunctions):
                    if "" == f.readline():
                        break
                    #f.readline() # symmetry
                    line = f.readline() # energy
                    self.orbitalEnergy.append(float(line.split()[1]))
                    f.readline() # spin
                    line = f.readline() # occupancy
                    self.occupancy.append(float(line.split()[1]))
                    MOCoeff.append([])
                    for j in range(self.nBasisFunctions):
                        line = f.readline()
                        MOCoeff[i].append(float(line.split()[1]))
                    self.nMOs += 1
                self.MOCoeff = np.array(MOCoeff).reshape((self.nMOs,self.nBasisFunctions))
                self.normMOCoeff = np.copy(self.MOCoeff)
                self.totalNorm = np.array(self.totalNorm)
                for i in range(self.nMOs):
                    #self.normMOCoeff[i,:] /= np.linalg.norm(self.normMOCoeff[i,:])
                    self.normMOCoeff[i,:] *= self.totalNorm
        f.close()


    def mo_values(self,r,moCoeff):

        # determine value of each basis function at r
        basisRValue = []
        for shell, shellType in enumerate(self.shellType):
            # determine number of basis functions per shell type
            if shellType == "s":
                functionsPerShell = 1 # s-type
                tempVal = self.s_val(self.primitivesPerShell[shell],self.contractionCoefficients[shell],self.primitiveExponents[shell],self.shellPos[shell],r)
                basisRValue.append(tempVal)
            elif shellType == "p":
                functionsPerShell = 3 # p-type
                ptemp  = self.p_val(self.primitivesPerShell[shell],self.contractionCoefficients[shell],self.primitiveExponents[shell],self.shellPos[shell],r)
                for val in ptemp:
                    basisRValue.append(val)
            elif shellType == "d":
                functionsPerShell = 6 # 6d-type
                dtemp  = self.d6_val(self.primitivesPerShell[shell],self.contractionCoefficients[shell],self.primitiveExponents[shell],self.shellPos[shell],r)
                for val in dtemp:
                    basisRValue.append(val)
            elif shellType == "f":
                functionsPerShell = 10 # 10f-type
                dtemp  = self.f10_val(self.primitivesPerShell[shell],self.contractionCoefficients[shell],self.primitiveExponents[shell],self.shellPos[shell],r)
                for val in dtemp:
                    basisRValue.append(val)
        # convert to numpy array
        basisRValue = np.array(basisRValue)
        # return value
        return np.dot(basisRValue,moCoeff)

    ######################################################################
    ###################    Normalization constants #######################
    ######################################################################

    # compute value of s-type gaussian basis functions at position r
    def s_norm(self,n,c,zeta):
        # n is number of primitives in CGTO
        # c is array of contraction coefficients (size n)
        # zeta is array of exponents (size n)
        # compute normalization constant
        norm = 0.0
        for i in range(n):
            for j in range(n):
                norm += c[i]*c[j]/(zeta[i]+zeta[j])**1.5
        norm = np.pi**(-0.75) * norm**(-0.5)
        return np.array([norm])

    # compute value of p-type gaussian basis functions at position r
    def p_norm(self,n,c,zeta):
        # n is number of primitives in CGTO
        # c is array of contraction coefficients (size n)
        # zeta is array of exponents (size n)
        # compute normalization constant
        norm = 0.0
        for i in range(n):
            for j in range(n):
                norm += c[i]*c[j]/(zeta[i]+zeta[j])**2.5
        norm = np.sqrt(2.0)*np.pi**(-0.75) * norm**(-0.5)
        return np.array([norm,norm,norm])
    
    # compute value of 5d-type gaussian basis functions at position r
    def d5_norm(self,c,zeta):
        # n is number of primitives in CGTO
        # c is array of contraction coefficients (size n)
        # zeta is array of exponents (size n)
        # compute normalization constant
        norm = np.empty(5,dtype=float)
        norm[0] = 2/np.sqrt(3.0)*(2.0/np.pi)**0.75*zeta[0]**1.75
        norm[1] = 2*(2.0/np.pi)**0.75*zeta[0]**1.75
        norm[2] = norm[3] = norm[4] = 4*(2.0/np.pi)**0.75*zeta[0]**1.75
        return norm

    # compute value of 6d-type gaussian basis functions at position r
    def d6_norm(self,c,zeta):
        # n is number of primitives in CGTO
        # c is array of contraction coefficients (size n)
        # zeta is array of exponents (size n)
        # compute normalization constant
        temp = 4*(2.0/np.pi)**0.75*zeta[0]**1.75
        norm1 = temp/np.sqrt(1.0*3.0*1.0*1.0)
        norm2 = temp/np.sqrt(1.0*1.0*1.0)
        norm = np.array( [norm1,norm1,norm1,norm2,norm2,norm2],dtype=float )
        return norm

    # compute value of 10f-type gaussian basis functions at position r
    def f10_norm(self,c,zeta):
        # n is number of primitives in CGTO
        # c is array of contraction coefficients (size n)
        # zeta is array of exponents (size n)
        # compute normalization constant
        temp = 2**3*(2.0/np.pi)**0.75*zeta[0]**(9.0/4.0)
        norm1 = temp/np.sqrt(1.0*3.0*5.0*1.0*1.0)
        norm2 = temp/np.sqrt(1.0*3.0*1.0*1.0)
        norm3 = temp/np.sqrt(1.0*1.0*1.0)
        norm = np.array( [norm1,norm1,norm1,norm2,norm2,norm2,norm2,norm2,norm2,norm3] )
        return norm

    #####################################################################
    ####################   Evaluate basis functions #####################
    #####################################################################

    # compute value of s-type gaussian basis functions at position r
    def s_val(self,n,c,zeta,r0,r):
        # n is number of primitives in CGTO
        # c is array of contraction coefficients (size n)
        # zeta is array of exponents (size n)
        # r0 is position (x,y,z) of basis function
        # r is position (x,y,z) to evaluate basis function
        # get displacement vector
        diff = r-r0
        r2 = np.dot(diff,diff)
        # sum over primitives 
        psiR = 0.0
        for i in range(n):
            psiR += c[i] * np.exp(-zeta[i]*r2)
        return psiR

    # compute value of p-type gaussian basis functions at position r
    def p_val(self,n,c,zeta,r0,r):
        # n is number of primitives in CGTO
        # c is array of contraction coefficients (size n)
        # zeta is array of exponents (size n)
        # r0 is position (x,y,z) of basis function
        # r is position (x,y,z) to evaluate basis function
        # get displacement vector
        diff = r-r0
        r2 = np.dot(diff,diff)
        # compute value of p-type at this position
        psiTemp = 0.0
        for i in range(n):
            psiTemp += c[i] * np.exp(-zeta[i]*r2)
        # multiply by x, y, or z to get p-type function
        psiR = psiTemp * diff
        # return array of psi values
        return psiR
    
    # compute value of 5d-type gaussian basis functions at position r
    def d5_val(self,n,c,zeta,r0,r):
        # n is number of primitives in CGTO
        # c is array of contraction coefficients (size n)
        # zeta is array of exponents (size n)
        # r0 is position (x,y,z) of basis function
        # r is position (x,y,z) to evaluate basis function
        # get displacement vector
        diff = r-r0
        r2 = np.dot(diff,diff)
        # compute value of d-type at this position
        psiTemp = c[0] * np.exp(-zeta[0]*r2)
        # values of d-type multipliers
        d5 = np.array([2*diff[2]**2-diff[0]**2-diff[1]**2,diff[0]**2-diff[1]**2,diff[0]*diff[1],diff[0]*diff[2],diff[1]*diff[2]])
        psiR = d5 * psiTemp 
        return psiR 

    # compute value of 6d-type gaussian basis functions at position r
    def d6_val(self,n,c,zeta,r0,r):
        # n is number of primitives in CGTO
        # c is array of contraction coefficients (size n)
        # zeta is array of exponents (size n)
        # r0 is position (x,y,z) of basis function
        # r is position (x,y,z) to evaluate basis function
        # get displacement vector
        diff = r-r0
        r2 = np.dot(diff,diff)
        # compute value of d-type at this position
        psiTemp = c[0] * np.exp(-zeta[0]*r2)
        # values of d-type multipliers
        d6 = np.array([diff[0]**2,diff[1]**2,diff[2]**2,diff[0]*diff[1],diff[0]*diff[2],diff[1]*diff[2]])
        psiR = d6 * psiTemp
        return psiR 

    # compute value of 10f-type gaussian basis functions at position r
    def f10_val(self,n,c,zeta,r0,r):
        # n is number of primitives in CGTO
        # c is array of contraction coefficients (size n)
        # zeta is array of exponents (size n)
        # r0 is position (x,y,z) of basis function
        # r is position (x,y,z) to evaluate basis function
        # get displacement vector
        diff = r-r0
        r2 = np.dot(diff,diff)
        # compute value of f-type at this position
        psiTemp = c[0] * np.exp(-zeta[0]*r2)
        # values of f-type multipliers
        f10 = np.array([diff[0]**3,diff[1]**3,diff[2]**3,diff[0]**2*diff[1],diff[0]**2*diff[2],diff[0]*diff[1]**2,diff[1]**2*diff[2],diff[0]*diff[2]**2,diff[1]*diff[2]**2,diff[0]*diff[1]*diff[2]])
        psiR = f10 * psiTemp 
        return psiR 
    
    # subroutine to write MO values out in a gaussian cube file - NOTE: origin and cubeSize should be in units of Angstroms
    def write_mo_cube(self,cubeFileName,mo,origin=np.array([0.0,0.0,0.0]),cubeSize=np.array([[0.1,0,0],[0,0.1,0],[0,0,0.1]]),nCubes=[10,10,10]):
      
        # get total number of points to evalues MO at
        totalCubes = nCubes[0]*nCubes[1]*nCubes[2]
        # open cube file for writing
        f = open(cubeFileName,'w')
        # write two title cards
        f.write("%s\n" % ('Cube File Generated by Code Written by Martin McCullagh 6/11/19'))
        f.write("%s %d\n" % ('Orbital',mo))
        # write number of atoms, origin of grid and total number of grid points
        f.write("%5d%12.6f%12.6f%12.6f%5d\n" % (-self.nAtoms,origin[0],origin[1],origin[2],1))
        # write cube size and spacing information
        f.write("%5d%12.6f%12.6f%12.6f\n" % (nCubes[0], cubeSize[0,0], cubeSize[0,1], cubeSize[0,2]))
        f.write("%5d%12.6f%12.6f%12.6f\n" % (nCubes[1], cubeSize[1,0], cubeSize[1,1], cubeSize[1,2]))
        f.write("%5d%12.6f%12.6f%12.6f\n" % (nCubes[2], cubeSize[2,0], cubeSize[2,1], cubeSize[2,2]))
        # write atom position and charges (charges are not in molden file so are made up)
        for i in range(self.nAtoms):
            f.write("%5d%12.6f%12.6f%12.6f%12.6f\n" % (self.atomicNumber[i],self.nuclearCharge[i],self.atomPos[i,0],self.atomPos[i,1],self.atomPos[i,2]))
        # write MO metadata
        f.write("%5d%5d\n" % (1, mo))
        pbar = tqdm(total=totalCubes)
        # write MO value at grid positions
        for i in range(nCubes[0]):
            x = i * cubeSize[0,:]
            for j in range(nCubes[1]):
                y = j * cubeSize[1,:]
                count = 0
                for k in range(nCubes[2]):
                    z =  k * cubeSize[2,:]
                    r = origin + x + y + z
                    #r = np.array([x,y,z],dtype=float)
                    f.write("%13.5e" % (self.mo_values(r,self.normMOCoeff.T[:,mo-1]))) 
                    pbar.update()
                    count += 1
                    # make new line if six values have been printed
                    if count%6==0:
                        f.write("\n")
                # make new line for next set of z values
                if count%6!=0:
                    f.write("\n")
        pbar.close()
        # close file
        f.close()

    # subroutine to write linear combination of MO values to a cube
    def write_linear_comb_mo_cube(self,cubeFileName,mo1,coeff1,mo2,coeff2,origin=np.array([0.0,0.0,0.0]),cubeSize=np.array([[0.1,0,0],[0,0.1,0],[0,0,0.1]]),nCubes=[10,10,10]):
     

        moCoeff = coeff1*self.normMOCoeff.T[:,mo1-1] + coeff2*self.normMOCoeff.T[:,mo2-1]

        # get total number of points to evalues MO at
        totalCubes = nCubes[0]*nCubes[1]*nCubes[2]
        # open cube file for writing
        f = open(cubeFileName,'w')
        # write two title cards
        f.write("%s\n" % ('Cube File Generated by Code Written by Martin McCullagh 6/11/19'))
        f.write("%s %d %s %d\n" % ('Orbital',mo1,'and orbital', mo2))
        # write number of atoms, origin of grid and total number of grid points
        f.write("%5d%12.6f%12.6f%12.6f%5d\n" % (-self.nAtoms,origin[0],origin[1],origin[2],1))
        # write cube size and spacing information
        f.write("%5d%12.6f%12.6f%12.6f\n" % (nCubes[0], cubeSize[0,0], cubeSize[0,1], cubeSize[0,2]))
        f.write("%5d%12.6f%12.6f%12.6f\n" % (nCubes[1], cubeSize[1,0], cubeSize[1,1], cubeSize[1,2]))
        f.write("%5d%12.6f%12.6f%12.6f\n" % (nCubes[2], cubeSize[2,0], cubeSize[2,1], cubeSize[2,2]))
        # write atom position and charges (charges are not in molden file so are made up)
        for i in range(self.nAtoms):
            f.write("%5d%12.6f%12.6f%12.6f%12.6f\n" % (self.atomicNumber[i],self.nuclearCharge[i],self.atomPos[i,0],self.atomPos[i,1],self.atomPos[i,2]))
        # write MO metadata
        f.write("%5d%5d\n" % (1, mo1))
        pbar = tqdm(total=totalCubes)
        # write MO value at grid positions
        for i in range(nCubes[0]):
            x = i * cubeSize[0,:]
            for j in range(nCubes[1]):
                y = j * cubeSize[1,:]
                count = 0
                for k in range(nCubes[2]):
                    z =  k * cubeSize[2,:]
                    r = origin + x + y + z
                    #r = np.array([x,y,z],dtype=float)
                    f.write("%13.5e" % (self.mo_values(r,moCoeff)))
                    pbar.update()
                    count += 1
                    # make new line if six values have been printed
                    if count%6==0:
                        f.write("\n")
                # make new line for next set of z values
                if count%6!=0:
                    f.write("\n")
        pbar.close()
        # close file
        f.close()

