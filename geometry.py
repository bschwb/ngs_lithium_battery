"""Geometry for Lithium-Ion Battery Microstructural Model

Length scale in micrometers.
"""

from random import uniform

import numpy as np
from numpy.linalg import norm

import netgen.geom2d as geom2d
from ngsolve import Draw, Mesh


r = 6
w = 200
h = 300
n = 200
electrode_h = 50

circles = np.array([uniform(r, w-r), uniform(r+electrode_h, h-r)])
while len(circles) < n:
    center = np.array([uniform(r, w-r), uniform(r+electrode_h, h-r)]);
    for circle in circles:
        if norm(center - circle) <= 2*r:
            break
    else:
        circles = np.vstack([circles, center])

geo = geom2d.SplineGeometry()
geo.AddRectangle((0, 0), (w, h))

for circle in circles:
    geo.AddCircle(c=(circle[0], circle[1]), r=r, leftdomain=2, rightdomain=1)

mesh = geo.GenerateMesh(maxh=100)
Draw(Mesh(mesh))
