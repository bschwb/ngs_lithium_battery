"""Simulation of Lithium-Ion Battery Microstructural Model

Length scale in micrometers (µm).

TODO:
* Add boundary conditions
* Add time stepping with non-linear solver
* Add activity coefficient
"""

from scipy.constants import physical_constants
import ngsolve as ngs
from ngsolve import grad, y, sqrt, exp


## Physical Constants
R = physical_constants['molar gas constant'][0]  # J mol^-1 K^-1
F = physical_constants['Faraday constant'][0]  # C mol^-1

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

## Initial values
norm_concentr = 22.86 * 1e-15  # mol µm^-3
solubility_limit = 1.2  # TODO: Still unsure about that
norm_init_concentr = {'electrolyte': solubility_limit, 'particle': 0.72}
anode_init_pot = 4.2  # Volt

## Material Properties
diffusivity = {'electrolyte': 2.66e-5 * 1e8, 'particle': 1e-9 * 1e8}  # µm^2 s^-1
conductivity = {'electrolyte': 2.53e-2 * 1e-4, 'particle': 3.8e-2 * 1e-4}  # S µm^-1
valence = {'electrolyte': 1, 'particle': 0}

## Further constants
alpha_a = 0.5
alpha_c = 0.5
reaction_rate = 5e-5 * 1e4  # µm s^-1
electrode_contact_resistance = 2e-13  # Ohm


def tanh(x):
    """tangens hyperbolicus for CoefficientFunctions"""
    return (1 - exp(-2*x)) / (1 + exp(-2*x))


## equations
def open_circuit_manganese(concentr):
    """Return open-circuit potential for Li_yMn2O4

    param: concentr - relative lithium concentration

    concentration range: [0, 1.0]
    """
    a = 4.19829
    b = 0.0565661 * tanh(-14.5546*concentr + 8.60942 )
    c = 0.0275479 * (1/((0.998432-concentr) * 0.492465 - 1.90111))
    d = 0.157123 * exp(-0.04738 * concentr**8)
    e = 0.810239 * exp(-40*(concentr-0.133875))
    return a + b - c - d + e


def open_circuit_carbon(concentr):
    """Return open-circuit potential for Li_xC6

    param: concentr - relative lithium concentration

    concentration range: [0, 0.7]
    """
    return -0.16 + 1.32 * exp(-3 * concentr)


def charge_flux_prefactor(concentr):
    """Return prefactor for Butler-Volmer relation

    params: concentr - relative lithium concentration
    """
    li_factor = norm_concentr * sqrt(solubility_limit - concentr) * sqrt(concentr)
    return F * reaction_rate * li_factor



mesh = ngs.Mesh('mesh.vol')
n = ngs.specialcf.normal(mesh.dim)


def material_overpotential_cathode(concentr, pot):
    """Return material overpotential for cathode Li_yMn2O4 particles

    TODO: Fix interface work dimension problem
    """
    ohmic_contact_pot = electrode_contact_resistance * discharge_current  # V
    interface_work = -particle_radius * grad(pot) * n  # V
    # return interface_work - open_circuit_manganese(concentr) + ohmic_contact_pot  # V
    return open_circuit_manganese(concentr) + ohmic_contact_pot  # V


def material_overpotential_anode(concentr, pot):
    """Return material overpotential for Li_xC6 anode

    TODO: Fix interface work dimension problem
    """
    ohmic_contact_pot = electrode_contact_resistance * discharge_current  # V
    interface_work = -thickness_anode * grad(pot) * thickness_anode  # V
    # return interface_work - open_circuit_carbon(concentr) + ohmic_contact_pot  # V
    return open_circuit_carbon(concentr) + ohmic_contact_pot  # V


n_lithium_space = ngs.H1(mesh, order=2, dirichlet='wall|cathode')
potential_space = ngs.H1(mesh, order=2, dirichlet='wall')
V = ngs.FESpace([n_lithium_space, potential_space])

u, p = V.TrialFunction()
v, q = V.TestFunction()

# Coefficient functions
cf_diffusivity = ngs.CoefficientFunction([diffusivity[mat] for mat in mesh.GetMaterials()])
cf_conductivity = ngs.CoefficientFunction([conductivity[mat] for mat in mesh.GetMaterials()])
cf_valence = ngs.CoefficientFunction([valence[mat] for mat in mesh.GetMaterials()])

f = ngs.BilinearForm(V)
f += ngs.SymbolicBFI(cf_diffusivity * grad(u) * grad(v))
f += ngs.SymbolicBFI(cf_diffusivity * discharge_rate / F / solubility_limit * v,
                     ngs.BND, definedon=mesh.Boundaries('anode'))

g = ngs.BilinearForm(V)
g += ngs.SymbolicBFI(cf_diffusivity * cf_valence * F / R / temperature * u * grad(p) * grad(v))
g += ngs.SymbolicBFI(charge_flux_prefactor(u) * (alpha_a + alpha_c) * material_overpotential_anode(u, p)
                     * cf_diffusivity * cf_valence * F**2 / R**2 / temperature**2 / cf_conductivity * u * v,
                     ngs.BND, definedon=mesh.Boundaries('particle'))
g += ngs.SymbolicBFI(charge_flux_prefactor(u) * (alpha_a + alpha_c) * material_overpotential_cathode(u, p)
                     * cf_diffusivity * cf_valence * F**2 / R**2 / temperature**2 / cf_conductivity * u * v,
                     ngs.BND, definedon=mesh.Boundaries('cathode'))
g += ngs.SymbolicBFI(cf_diffusivity * cf_valence * F / R / temperature / cf_conductivity
                     * discharge_rate * u * v,
                     ngs.BND, definedon=mesh.Boundaries('anode'))

A = ngs.BilinearForm(V)
A += ngs.SymbolicBFI(cf_conductivity * grad(p) * grad(q))


Aboundary = ngs.BilinearForm(V)
Aboundary += ngs.SymbolicBFI(charge_flux_prefactor(u) * (alpha_a + alpha_c) * F / R / temperature
                             * material_overpotential_anode(u, p) * q,
                             ngs.BND, definedon=mesh.Boundaries('particle'))
Aboundary += ngs.SymbolicBFI(charge_flux_prefactor(u) * (alpha_a + alpha_c) * F / R / temperature
                             * material_overpotential_cathode(u, p) * q,
                             ngs.BND, definedon=mesh.Boundaries('cathode'))
Aboundary += ngs.SymbolicBFI(discharge_rate / cf_conductivity * q,
                             ngs.BND, definedon=mesh.Boundaries('anode'))

h = ngs.BilinearForm(V)
h = ngs.SymbolicBFI(cf_diffusivity * cf_valence * F / R / temperature * u * grad(u) * grad(q))
h = ngs.SymbolicBFI(cf_diffusivity * cf_valence * F / R / temperature * u * grad(u) * grad(q))

with ngs.TaskManager():
    A.Assemble()

    # Initial conditions
    gfu = ngs.GridFunction(V)
    cf_n0 = ngs.CoefficientFunction([norm_init_concentr[mat] for mat in mesh.GetMaterials()])
    gfu.components[0].Set(cf_n0)
    gfu.components[1].Set(anode_init_pot*y/height)

    ngs.Draw(mesh)
