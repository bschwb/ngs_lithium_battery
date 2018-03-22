"""Simulation of Lithium-Ion Battery Microstructural Model

Length scale in micrometers (µm).

TODO:
* Add activity coefficient
"""

import math

from scipy.constants import physical_constants, epsilon_0
import ngsolve as ngs
from ngsolve import grad, sqrt, exp, log


## Physical Constants
R = physical_constants['molar gas constant'][0]  # J mol^-1 K^-1
F = physical_constants['Faraday constant'][0]  # C mol^-1
vacuum_permittivity = 1e-6 * epsilon_0  # F µm^-1

## Geometry Parameters
particle_radius = 8.5  # µm
thickness_anode = 250  # µm
thickness_cathode = 174  # µm
thickness_separator = 50  # µm
height = thickness_separator + thickness_cathode  # µm

## Model Parameters
temperature = 298.15  # K (25deg Celsius)
norm_discharge_rate =  2 * 1e-3 * 1e-8  # A µm^-2
discharge_rate = 1 * norm_discharge_rate  # A µm^-2
area = 24 * 1e8  # µm^2
discharge_current = discharge_rate * area  # A
width = 200  # µm
thick = area / width

## Initial values
normalization_concentration_3d = 22.86 * 1e-15  # mol µm^-3
normalization_concentration = normalization_concentration_3d / thick  # mol µm^-2
solubility_limit_cathode = normalization_concentration
solubility_limit_anode = 0.7 * normalization_concentration
init_concentr = {'electrolyte': 1.2 * normalization_concentration,
                 'particle': 0.18 * normalization_concentration}
cathode_init_pot = 4.2  # Volt

## Material Properties
diffusivity = {'electrolyte': 2.66e-5 * 1e8, 'particle': 1e-9 * 1e8}  # µm^2 s^-1
conductivity = {'electrolyte': 2.53e-2 * 1e-4, 'particle': 3.8e-2 * 1e-4}  # S µm^-1
valence = {'electrolyte': 1, 'particle': 0}

## Further constants
alpha_a = 0.5
alpha_c = 0.5
reaction_rate = 5e-5 * 1e4  # µm s^-1
electrode_contact_resistance = 2e-13  # Ohm
ohmic_contact_pot = electrode_contact_resistance * discharge_current  # V


def tanh(x):
    """tangens hyperbolicus for CoefficientFunctions"""
    return (1 - exp(-2*x)) / (1 + exp(-2*x))


def Pow(a, b):
    """Power function for CoefficientFunctions"""
    return exp(log(a)*b)


## equations
def open_circuit_manganese(concentration):
    """Return open-circuit potential for Li_yMn2O4

    param: concentration - Lithium concentration

    normalized concentration range: [0, 1.0]
    """
    concentr = concentration / normalization_concentration
    a = 4.19829
    b = 0.0565661 * tanh(-14.5546*concentr + 8.60942)
    c = 0.0275479 * (1/Pow(0.998432-concentr, 0.492465) - 1.90111)
    d = 0.157123 * exp(-0.04738 * concentr**8)
    e = 0.810239 * exp(-40*(concentr-0.133875))
    return a + b - c - d + e


def open_circuit_carbon(concentration):
    """Return open-circuit potential for Li_xC6

    param: concentration - Lithium concentration

    normalized concentration range: [0, 0.7]
    """
    concentr = concentration / normalization_concentration
    return -0.16 + 1.32 * exp(-3 * concentr)


def charge_flux_prefactor_cathode(concentration):
    """Return prefactor for Butler-Volmer relation in cathode

    params: concentration - Lithium concentration
    """
    # TODO: use power empirical constants here instead of sqrt
    solubility_difference = ngs.IfPos(solubility_limit_cathode - concentration,
                                      solubility_limit_cathode - concentration, 0)
    li_factor = sqrt(solubility_difference) * sqrt(concentration)
    return F * reaction_rate * li_factor


def charge_flux_prefactor_anode(concentration):
    """Return prefactor for Butler-Volmer relation in anode

    params: concentration - Lithium concentration
    """
    # TODO: use power empirical constants here instead of sqrt
    solubility_difference = ngs.IfPos(solubility_limit_anode - concentration,
                                      solubility_limit_anode - concentration, 0)
    li_factor = sqrt(solubility_difference) * sqrt(concentration)
    return F * reaction_rate * li_factor


mesh = ngs.Mesh('mesh.vol')

n_lithium_space = ngs.H1(mesh, order=2)
potential_space = ngs.H1(mesh, order=2)
V = ngs.FESpace([n_lithium_space, potential_space])
print(V.ndof)

u, p = V.TrialFunction()
v, q = V.TestFunction()

# Coefficient functions
cf_diffusivity = ngs.CoefficientFunction([diffusivity[mat] for mat in mesh.GetMaterials()])
cf_conductivity = ngs.CoefficientFunction([conductivity[mat] for mat in mesh.GetMaterials()])
cf_valence = ngs.CoefficientFunction([valence[mat] for mat in mesh.GetMaterials()])

n = ngs.specialcf.normal(mesh.dim)

def material_overpotential_cathode(concentr, pot):
    """Return material overpotential for cathode Li_yMn2O4 particles"""
    interface_work = -cf_conductivity * grad(pot) * n  # V
    return interface_work - open_circuit_manganese(concentr) + ohmic_contact_pot  # V


def material_overpotential_anode(concentr, pot):
    """Return material overpotential for Li_xC6 anode"""
    # interface_work = -thickness_anode * cf_conductivity * grad(pot) * n  # V
    interface_work = -cf_conductivity * grad(pot) * n  # V
    return interface_work - open_circuit_carbon(concentr) + ohmic_contact_pot  # V


mass = ngs.BilinearForm(V)
mass += ngs.SymbolicBFI(u * v)

a = ngs.BilinearForm(V)
a += ngs.SymbolicBFI(-cf_diffusivity * grad(u) * grad(v))
a += ngs.SymbolicBFI(cf_diffusivity * discharge_rate / F * v,
                     ngs.BND, definedon=mesh.Boundaries('anode'))

a += ngs.SymbolicBFI(-cf_diffusivity * cf_valence * F / R / temperature * u * grad(p) * grad(v))
a += ngs.SymbolicBFI(-charge_flux_prefactor_cathode(u) * (alpha_a + alpha_c)
                     * material_overpotential_cathode(u, p) * cf_diffusivity
                     * cf_valence * F**2 / R**2 / temperature**2 / conductivity['particle'] * u * v,
                     ngs.BND, definedon=mesh.Boundaries('particle'))
a += ngs.SymbolicBFI(-charge_flux_prefactor_anode(u) * (alpha_a + alpha_c)
                     * material_overpotential_anode(u, p) * cf_diffusivity
                     * cf_valence * F**2 / R**2 / temperature**2 / cf_conductivity * u * v,
                     ngs.BND, definedon=mesh.Boundaries('anode'))
a += ngs.SymbolicBFI(cf_diffusivity * F / R / temperature / conductivity['electrolyte']
                     * discharge_rate * u * v,
                     ngs.BND, definedon=mesh.Boundaries('cathode'))

a += ngs.SymbolicBFI(-cf_conductivity * grad(p) * grad(q))
a += ngs.SymbolicBFI(-charge_flux_prefactor_anode(u) * (alpha_a + alpha_c) * F / R / temperature
                     * material_overpotential_anode(u, p) * q,
                     ngs.BND, definedon=mesh.Boundaries('anode'))
a += ngs.SymbolicBFI(-charge_flux_prefactor_cathode(u) * (alpha_a + alpha_c) * F / R / temperature
                     * material_overpotential_cathode(u, p) * q,
                     ngs.BND, definedon=mesh.Boundaries('particle'))
a += ngs.SymbolicBFI(discharge_rate * q, ngs.BND, definedon=mesh.Boundaries('cathode'))

a += ngs.SymbolicBFI(-cf_diffusivity * cf_valence * F / R / temperature * u * grad(u) * grad(q))
a += ngs.SymbolicBFI(cf_diffusivity * cf_valence * F / R / temperature * discharge_rate * u * q,
                     ngs.BND, definedon=mesh.Boundaries('anode'))

gfu = ngs.GridFunction(V)

# test if matrix contains a nan entry
ones = gfu.vec.CreateVector()
result = gfu.vec.CreateVector()
ones[:] = 1
a.Apply(ones, result)
assert (not math.isnan(ngs.Norm(result))), 'System matrix a contains a nan entry!'

with ngs.TaskManager():
    mass.Assemble()

    # Initial conditions
    cf_n0 = ngs.CoefficientFunction([init_concentr[mat] for mat in mesh.GetMaterials()])
    gfu.components[0].Set(cf_n0)

    ## Poisson's equation for initial potential
    # An extra space is needed, due to different dirichlet conditions for the initial potential
    initial_potential_space = ngs.H1(mesh, order=2, dirichlet='cathode|anode')
    phi = initial_potential_space.TrialFunction()
    psi = initial_potential_space.TestFunction()

    ### Stiffness matrix with small regularization term
    a_pot = ngs.BilinearForm(initial_potential_space)
    a_pot += ngs.SymbolicBFI(grad(phi) * grad(psi))
    a_pot.Assemble()

    f_pot = ngs.LinearForm(initial_potential_space)
    f_pot += ngs.SymbolicLFI(cf_valence * cf_n0 * F * psi)
    f_pot.Assemble()

    gf_phi = ngs.GridFunction(initial_potential_space)
    gf_phi.Set(ngs.CoefficientFunction(cathode_init_pot), definedon=mesh.Boundaries('cathode'))
    ngs.Draw(gf_phi)
    res = f_pot.vec.CreateVector()
    res.data = f_pot.vec - a_pot.mat * gf_phi.vec
    gf_phi.vec.data += a_pot.mat.Inverse(initial_potential_space.FreeDofs()) * res

    ngs.Redraw()
    gfu.components[1].vec.data = gf_phi.vec

    # Time stepping
    ngs.Draw(gfu.components[1])
    ngs.Draw(gfu.components[0])
    timestep = 1
    t = 0

    w = gfu.vec.CreateVector()
    w2 = gfu.vec.CreateVector()
    b = gfu.vec.CreateVector()
    b2 = gfu.vec.CreateVector()
    mid = gfu.vec.CreateVector()
    d = gfu.vec.CreateVector()
    z = gfu.vec.CreateVector()
    mat = mass.mat.CreateMatrix()

    curr = gfu.vec.CreateVector()

    du = gfu.vec.CreateVector()
    while t < 1000:
        t += timestep
        print(t)

        a.Apply(gfu.vec, b)
        mass.Apply(gfu.vec, b2)
        curr.data = gfu.vec
        for i in range(2):
            a.Apply(curr, w)
            w2.data = mass.mat * curr
            mid.data = timestep/2 * (w + b) - w2 + b2

            a.AssembleLinearization(curr)
            mat.AsVector().data = timestep/2 * a.mat.AsVector() - mass.mat.AsVector()
            inv = mat.Inverse(V.FreeDofs())
            du.data = inv * mid
            curr.data -= du

        gfu.vec.data = curr.data
        ngs.Redraw()
