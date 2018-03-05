"""Simulation of Lithium-Ion Battery Microstructural Model

Length scale in micrometers (µm).

TODO:
* Add boundary conditions
* Add time stepping with non-linear solver
* Add activity coefficient
"""

from scipy.constants import physical_constants
import ngsolve as ngs
from ngsolve import grad, y

# Physical Constants
R = physical_constants['molar gas constant'][0]  # J mol^-1 K^-1
F = physical_constants['Faraday constant'][0]  # C mol^-1

# Geometry Parameters
thickness_cathode = 174  # µm
thickness_separator = 50  # µm
height = thickness_separator + thickness_cathode  # µm

# Model Parameters
T = 298.15  # K (25deg Celsius)
discharge_rate =  2e-3  # A cm^-2
area = 24  # cm^2

# Initial values
n0 = {'electrolyte': 1.2, 'particle': 0.72}
phi0 = 4.2

# Material Properties
diffusivity = {'electrolyte': 2.66e-25, 'particle': 1e-29}  # µm^2 s^-1
conductivity = {'electrolyte': 2.53e8, 'particle': 3.8e8}  # S µm^-1
valence = {'electrolyte': 1, 'particle': 0}

with ngs.TaskManager():
    mesh = ngs.Mesh('mesh.vol')

    n_lithium_space = ngs.H1(mesh, order=2, dirichlet='wall|cathode')
    potential_space = ngs.H1(mesh, order=2, dirichlet='wall')
    V = ngs.FESpace([n_lithium_space, potential_space])

    u, p = V.TrialFunction()
    v, q = V.TestFunction()

    # Coefficient functions
    cf_diffusivity = ngs.CoefficientFunction([diffusivity[mat] for mat in mesh.GetMaterials()])
    cf_conductivity = ngs.CoefficientFunction([conductivity[mat] for mat in mesh.GetMaterials()])
    cf_valence = ngs.CoefficientFunction([valence[mat] for mat in mesh.GetMaterials()])

    a = ngs.BilinearForm(V)
    a += ngs.SymbolicBFI(cf_diffusivity * grad(u) * grad(v) + cf_diffusivity * cf_valence * F / R / T * u * grad(p) * grad(v))
    a += ngs.SymbolicBFI(cf_conductivity * grad(p) * grad(q) + cf_diffusivity * cf_valence * F / R / T * u * grad(u) * grad(q))
    a.Assemble()

    # Initial conditions
    gfu = ngs.GridFunction(V)
    cf_n0 = ngs.CoefficientFunction([n0[mat] for mat in mesh.GetMaterials()])
    gfu.components[0].Set(cf_n0)
    gfu.components[1].Set(phi0*y/height)

    ngs.Draw(mesh)
