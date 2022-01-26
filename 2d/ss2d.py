#!/usr/bin/python

import matplotlib.pyplot as plt
import scipy.linalg as la
from math import ceil
import datetime
from scipy.fftpack import fft, fft2, ifft, ifft2, dct, idct, dst, idst, fftshift, fftfreq
from numpy import reshape, linspace, zeros, zeros_like, array, pi, sin, cos, exp, arange, matmul, abs, conj, real, convolve, ones, diag, inner, meshgrid, outer
import numpy as np

start  = datetime.datetime.now()

#plt.rcParams['figure.figsize'] = 16, 9




# Simulation Parameters

number_of_oscillations = 10

dt = 0.1   # Temporal seperation

Nx = 2**4  # Number of position grid points along x-direction
Lx = 3 * pi  # Box width along x-direction

Ny = 2**4   # Number of position grid points along y-direction
Ly = 3 * pi  # Box width along y-direction

kappa = 1 # Anisotropy const.

# Initial conditions for test wf
x0 = 2
ywavenumber = 1


# Output file
filename = 'ss' + str(Nx) + 'x' + str(Ny) + '+dt' + str(dt) + '.dat' # Output file




# Derived parameters

halfdt = dt*0.5 # Name says it all

dx = Lx/Nx    # Spatial separation along x-axis
dy = Ly/Ny    # Spatial separation along y-axis

x = arange(-Lx/2,Lx/2,dx) # Position space grid
y = arange(-Ly/2,Ly/2,dy) # Position space grid

# This function generates the momentum space grid.
def GenerateMomentumSpace(L, N):
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

kx = GenerateMomentumSpace(Lx , Nx) # momentum space grid
ky = GenerateMomentumSpace(Ly , Ny) # momentum space grid










# This function calculates the anisotropic potential energy at specified position.
# kappa is defined in such a way that force constant omega_y^2 = kappa * omega_x^2   .
def Potential(posx, posy):
    return 0.5*(posx**2) + 0.5*(kappa * (posy**2))


# This function creates a discrete 2D potential energy matrix. It's compatible
# with flattening with row priority.
def VMatrix():
    result = zeros(( x.size*y.size , x.size*y.size ))
    cursor = 0
    for i in range(y.size):
        for j in range(x.size):
            result[cursor][cursor] = Potential(x[j] , y[i])
            cursor += 1

    return result

# Initialize once, use as needed
V = VMatrix()

# This function performs split step fourier time evolution.
def EvolveSplitStep(some_psi):
    # Refer to: https://www.overleaf.com/7461894969pxkgqkzvmdws

    # First, neglect kinetic for half step, U1 acts
    psi = np.matmul( la.expm( -1j * halfdt * V), np.ndarray.flatten(some_psi) ).reshape(Ny,Nx)

    # Second neglect potential for whole step, U2 acts, in momentum space
    psi_x = ifft( exp( -1j * (kx**2/2) * dt) * fft(psi, axis = 0), axis = 0)
    psi_y = ifft( exp( -1j * (ky**2/2) * dt) * fft(psi, axis = 1), axis = 1)

    psi = psi_x + psi_y

    # Third, neglect kinetic for half more step, U3 acts
    psi = np.matmul(la.expm( -1j * halfdt * V), np.ndarray.flatten(psi)).reshape(Ny,Nx)

    return psi




# Implementation of coherent state specified in the
#
# https://files.slack.com/files-pri/T5MM8M0CR-F02LPDYN02X/download/2021-11-09-2d_coherent_state.pdf
#
# with initial x and y displacements
def CoherentStateExact2D(xinitial , ywavenumber , t):
    result = np.zeros((Ny, Nx), dtype=complex)
    for i in range(Ny):
        for j in range(Nx):
            currentx = x[j]
            currenty = y[i]

            A = (kappa**0.125)/(pi**0.5)

            X = np.exp(   -0.5 * ( currentx - xinitial * cos(t) )**2   - 1j * xinitial * currentx * sin(t)    )

            Y = np.exp(   -0.5 * (kappa**0.5) * ( currenty - (ywavenumber/(kappa**0.5)) * sin(t*(kappa**0.5))   )**2     + 1j*ywavenumber*currenty*cos(t*(kappa**0.5))                       )

            result[i][j] = A*X*Y

    return result



# This function calculates the overlap between two 2D wavefunctions
def Overlap2D(psi1, psi2):
    overlap = 0
    for i in range(x.size):
        for j in range(y.size):
            overlap += psi1[i][j] * np.conj(psi2[i][j]) * dx * dy
    return np.abs(overlap)




# This function will generate tons of plots. Use ffmpeg to convert them to a video:
#
#
# ffmpeg -r 60 -f image2 -s 1920x1080 -i visual%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p fft.mp4
def Run():
    terminateAt = number_of_oscillations * 2 * np.pi
    timesteps = ceil(terminateAt / dt)

    currentDate  = datetime.datetime.now()

    with open(filename, 'a') as f:
        f.write('# Simulation started at ' + str(currentDate) + '\n')
        f.write('#\n')
        f.write('# Simulation details: \n')

        f.write('# x-initial = ' + str(x0) + ' \n')
        f.write('# k_y = ' + str(ky) + ' \n')

        f.write('# Lx = ' + str(Lx) + ' \n')


        f.write('# Ly = ' + str(Ly) + ' \n')


        f.write('# dt = ' + str(dt) + ' s \n')
        f.write('# dx = ' + str(dx) + ' \n')
        f.write('# dy = ' + str(dy) + ' \n')
        f.write('# Spatial grid points : ' + str(Nx) + ' x ' + str(Ny) + ' \n')
        f.write('#\n')
        f.write('# time\terror\n')

    psi_num = CoherentStateExact2D(x0, ywavenumber, 0)

    for i in range(timesteps):
        currentTime = i*dt

        psi_exact = CoherentStateExact2D(x0, ywavenumber, currentTime)
        error = np.abs(1 - Overlap2D(psi_num, psi_exact))

        current_time_as_string = "{:.5f}".format(currentTime)
        print('Current time: ' + current_time_as_string)
        error_as_string = "{:.5e}".format(error)



        with open(filename, 'a') as f:
            f.write(current_time_as_string+'\t'+error_as_string+'\n')


        plt.suptitle('Time: ' + current_time_as_string + '\n ky = '+str(ywavenumber)+' , x_initial = '+str(x0)+' \nError: ' + error_as_string)
        plt.subplot(1, 2, 1)
        plt.contourf(x, y, np.abs(psi_num)**2, 256, cmap='RdGy')
        plt.title('Numerical solution')
        plt.xlim([-Lx/2, Lx/2])
        plt.ylim([-Ly/2, Ly/2])
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.contourf(x, y, np.abs(psi_exact)**2, 256, cmap='RdGy')
        plt.title('Exact solution')
        plt.xlim([-Lx/2, Lx/2])
        plt.ylim([-Ly/2, Ly/2])
        plt.grid()

        plt.tight_layout()
        plt.savefig('visual'+str(i)+'.png')
        plt.clf()



        psi_num = EvolveSplitStep(psi_num)


    end = time.time()
    difference = int(end - start)
    print( str(difference) + 's')
    with open(filename, 'a') as f:
        f.write('# Simulation finished at ' + str(currentDate) + '\n')
        f.write('# Simulation took ' + str(difference) + ' seconds\n')


Run()
