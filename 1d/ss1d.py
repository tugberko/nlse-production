#!/usr/bin/python

#import matplotlib.pyplot as plt
import scipy.linalg as la
from math import ceil
import datetime
from scipy.fftpack import fft, ifft, dct, idct, dst, idst, fftshift, fftfreq
from numpy import linspace, zeros, zeros_like, array, pi, sin, cos, exp, arange, matmul, abs, conj, real, convolve, ones, diag

start  = datetime.datetime.now()

#plt.rcParams['figure.figsize'] = 16, 9

# Simulation Parameters

dt = 0.005   # Temporal seperation

N = 2**9   # Number of position grid points

L = 3 * pi  # Box width

filename = 'ss'+str(N) + '+dt' + str(dt) + '.dat' # Output file

# Derived parameters

halfdt = dt*0.5 # Name says it all

dx = L/N    # Spatial separation

x = arange(-L/2,L/2,dx) # Position space grid

# This function generates the momentum space grid.
def GenerateMomentumSpace():
    dk = (2*pi/L)
    k = zeros(N)
    if ((N%2)==0):
        #-even number
        for i in range(1,N//2):
            k[i]=i
            k[N-i]=-i
    else:
        #-odd number
        for i in range(1,(N-1)//2):
            k[i]=i
            k[N-i]=-i

    return dk * k

k = GenerateMomentumSpace() # momentum space grid

ksq = k**2  # k-square



# This function produces a discrete 1D Laplacian matrix compatible with wavefunction.
def laplacian1D():
    result = zeros((N, N))
    for i in range(N):
        result[i][i]=-2
    for i in range(N-1):
        result[i][i+1]=1
        result[i+1][i]=1
    return 1/(dx**2) * result

# This function outputs a matrix representing the kinetic
# energy part of the Hamiltonian.
def TMatrix():
    return -0.5 * laplacian1D()

# This function generates an array of potential energies corresponding to the
# position space.
def Potential():
    pot = zeros_like(x)
    for i in range(N):
        pot[i] = 0.5*(x[i]**2)
    return pot

# This function outputs a matrix representing the potential
# energy part of the Hamiltonian.
def VMatrix():
    return diag(Potential())

# This function creates a discrete 1D kinetic energy matrix.
def TMatrix():
    return -0.5 * laplacian1D()

V = VMatrix()

# This function performs split step fourier time evolution.
def EvolveSplitStep(some_psi):
    # Refer to: https://www.overleaf.com/7461894969pxkgqkzvmdws

    # First, neglect kinetic for half step, U1
    psi = matmul( la.expm( -1j * halfdt * V) , some_psi  )

    # Second, neglect potential for whole step, U2-hat in momentum space
    psi_hat = fft(psi)
    psi_hat = exp( -1j * ksq * dt / 2) * psi_hat
    psi = ifft(psi_hat)

    # Third, neglect kinetic for half more step, U3
    psi = matmul( la.expm( -1j * halfdt * V) , psi  )

    return psi

# Exact solution of the coherent state
def CoherentStateExact(t):
    k=1
    return ((1/pi)**(0.25)) * exp(   -0.5 * (x-k*sin(t))**2   ) * exp(1j*k*x*cos(t))

# This function calculates the overlap between two wavefunctions.
def Overlap(psi1, psi2):
    overlap = 0
    for i in range(x.size):
        overlap += psi1[i] * conj(psi2[i]) * dx
    return abs(overlap)



def RunWithoutVisuals():
    number_of_oscillations = 10
    terminateAt = number_of_oscillations * 2 * pi
    timesteps = ceil(terminateAt / dt)

    with open(filename, 'w') as f:
        f.write('# 1D Schrodinger with Split Step Fourier Method\n#\n')
        f.write('# Simulation started at ' + str(start) + '\n')
        f.write('#\n')
        f.write('# Simulation details: \n')
        f.write('# Box width = ' + str(L) + ' \n')
        f.write('# Grid size : ' + str(N) +' \n')
        f.write('# dt = ' + str(dt) + ' s \n')
        f.write('# dx = ' + str(dx) + ' \n')
        f.write('#\n')
        f.write('# time\terror\n')

    psi_num = CoherentStateExact(0)

    for i in range(timesteps):
        currentTime = i*dt

        psi_exact = CoherentStateExact(currentTime)

        current_time_as_string = "{:.5f}".format(currentTime)
        print('Current time: ' + current_time_as_string)

        error = abs(1 - Overlap(psi_exact, psi_num))
        error_as_string = "{:.4e}".format(error)

        with open(filename, 'a') as f:
            f.write(current_time_as_string+'\t'+error_as_string+'\n')

        psi_num = EvolveSplitStep(psi_num)

RunWithoutVisuals()

finish  = datetime.datetime.now()
with open(filename, 'a') as f:
    f.write('# Simulation took ' + str(finish-start))
