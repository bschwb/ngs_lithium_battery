"""Simulation of Lithium-Ion Battery Microstructural Model

Length scale in micrometers (µm).
"""
from time import sleep

from scipy.constants import physical_constants

import ngsolve as ngs
from ngsolve import grad
from ngsolve.internal import visoptions


## Physical Constants
F = physical_constants['Faraday constant'][0]  # C mol^-1

## Geometry Parameters
thickness_cathode = 174  # µm
thickness_separator = 50  # µm
height = thickness_separator + thickness_cathode  # µm
width = 200  # µm

## Model Parameters
battery_capacity =  2 * 1e-3 * 3600 * 1e-8  # A s µm^-2
discharge_current_density = 1 * battery_capacity  # A µm^-2

## Initial values
initial_concentration = 1e3 * 1e-18  # mol µm^-3

## Material Properties
diffusivity = 2.66e-5 * 1e8  # µm^2 s^-1


with ngs.TaskManager():
    mesh = ngs.Mesh('mesh.vol')

    V = ngs.H1(mesh, order=1)
    print(V.ndof)

    u = V.TrialFunction()
    v = V.TestFunction()

    mass = ngs.BilinearForm(V)
    mass += ngs.SymbolicBFI(u * v)
    mass.Assemble()

    a = ngs.BilinearForm(V)
    a += ngs.SymbolicBFI(diffusivity * grad(u) * grad(v))
    a.Assemble()

    f = ngs.LinearForm(V)
    f += ngs.SymbolicLFI(discharge_current_density / F * v.Trace(), ngs.BND,
                         definedon=mesh.Boundaries('anode'))
    f.Assemble()

    # Initial conditions
    gfu = ngs.GridFunction(V)
    gfu.Set(ngs.CoefficientFunction(initial_concentration))

    # Visualization
    ngs.Draw(gfu)
    print('0s')
    input()

    # Time stepping
    timestep = 1
    t = 0

    res = gfu.vec.CreateVector()
    update = gfu.vec.CreateVector()

    timestep_mat = mass.mat.CreateMatrix()
    timestep_mat.AsVector().data = mass.mat.AsVector() + timestep/2 * a.mat.AsVector()
    timestep_inv = timestep_mat.Inverse(V.FreeDofs())

    while t < 1000:
        t += timestep
        print(t, 's', sep='')
        sleep(0.2)

        res.data = f.vec - a.mat * gfu.vec

        update.data = timestep_inv * res

        gfu.vec.data += timestep * update
        ngs.Redraw()
