import molpro_molden as mol
import numpy as np

# read in molden file into object 
ch2 = mol.molpro("ch2_c2v.molden")

# set origin and cube size arrays
origin = np.array([-5.952637, -5.952637, -6.142723])
cubeSize = np.array([ [0.150700,0.000000,0.0000000], [0.000000,0.150700,0.000000], [0.00000, 0.000000,0.150700] ])

# pick two example MOs to work with - note that the write_mo_cube routines start at mo=1
mo1 = 7 
mo2 = 8

# write each mo out to cube file
fileName = "ch2_c2v_mo" + str(mo1) + ".cub"
print("Writing file:", fileName)
ch2.write_mo_cube(fileName,mo1,origin=origin,nCubes=[80,80,80],cubeSize=cubeSize)
fileName = "ch2_c2v_mo" + str(mo2) + ".cub"
print("Writing file:", fileName)
ch2.write_mo_cube(fileName,mo2,origin=origin,nCubes=[80,80,80],cubeSize=cubeSize)

# write linear combinations to cube file
coeff1 = np.sqrt(2.0)/2.0  # linear coefficient for mo1
coeff2 = np.sqrt(2.0)/2.0  # linear coefficient for mo2
fileName = "ch2_c2v_mo" + str(mo1) + "_+_mo" + str(mo2) + ".cub"
print("Writing file:", fileName)
ch2.write_linear_comb_mo_cube(fileName,mo1,coeff1,mo2,coeff2,origin=origin,nCubes=[80,80,80],cubeSize=cubeSize)
fileName = "ch2_c2v_mo" + str(mo1) + "_-_mo" + str(mo2) + ".cub"
print("Writing file:", fileName)
ch2.write_linear_comb_mo_cube(fileName,mo1,coeff1,mo2,-coeff2,origin=origin,nCubes=[80,80,80],cubeSize=cubeSize)

