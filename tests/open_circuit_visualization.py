import matplotlib.pyplot as plt
import numpy as np
from numpy import tanh, exp


def open_circuit_manganese(concentration):
    """Return open-circuit potential for Li_yMn2O4

    param: concentration - Lithium concentration

    normalized concentration range: [0, 1.0]
    """
    concentr = concentration
    a = 4.19829
    b = 0.0565661 * tanh(-14.5546*concentr + 8.60942)
    z = 0.998390
    if z - concentr > 0:
        c = 0.0275479 * (1/pow(0.998432-concentr, 0.492465) - 1.90111)
    else:
        c = 0.0275479 * (1/pow(0.998432-z, 0.492465) - 1.90111)
    d = 0.157123 * exp(-0.04738 * concentr**8)
    e = 0.810239 * exp(-40*(concentr-0.133875))
    e2 = 0.810239 * exp(-40*(0.05-0.133875))

    f = 0.5-0.5*tanh(1000*(concentr - z))
    g = 0.0275479 * (1/pow(0.998432-z, 0.492465) - 1.90111)

    r = 0.5+0.5*tanh(1000*(concentr - 0.05))
    k = (a + b - c * f - (1-f) * g - d + e)
    return r * k + (1-r) * (a+e2)


def open_circuit_carbon(concentration):
    """Return open-circuit potential for Li_xC6

    param: concentration - Lithium concentration

    normalized concentration range: [0, 0.7]
    """
    # concentr = concentration / normalization_concentration
    concentr = concentration

    lower_capoff = -1
    r = 0.5+0.5*tanh(100*(concentr - lower_capoff))

    return (-0.16 + 1.32 * exp(-3 * concentr)) * r + (1-r) * (-0.16 + 1.32*exp(-3*lower_capoff))

x = np.arange(-2, 1, 0.001)
# plt.plot(x, np.vectorize(open_circuit_manganese)(x))
plt.plot(x, open_circuit_carbon(x))
plt.show()
