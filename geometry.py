"""Geometry for Lithium-Ion Battery Microstructural Model

Length scale in micrometers.
"""

from random import uniform

import numpy as np
from numpy.linalg import norm

import netgen.geom2d as geom2d


r = 8.5
thickness_cathode = 174
thickness_separator = 50
h = thickness_separator + thickness_cathode
w = 200
n = 50

circles = np.array([uniform(r, w-r), uniform(r+thickness_separator, h-r)])
while len(circles) < n:
    center = np.array([uniform(r, w-r), uniform(r+thickness_separator, h-r)]);
    for circle in circles:
        if norm(center - circle) <= 2*r:
            break
    else:
        circles = np.vstack([circles, center])

geo = geom2d.SplineGeometry()
pnts = [(0, 0), (w, 0), (w, h), (0, h)]
p1, p2, p3, p4 = [geo.AddPoint(*pnt) for pnt in pnts]
geo.Append(['line', p1, p2], leftdomain=1, rightdomain=0, bc='anode')
geo.Append(['line', p2, p3], leftdomain=1, rightdomain=0, bc='wall')
geo.Append(['line', p3, p4], leftdomain=1, rightdomain=0, bc='wall')
geo.Append(['line', p4, p1], leftdomain=1, rightdomain=0, bc='cathode')

for circle in circles:
    geo.AddCircle(c=(circle[0], circle[1]), r=r, leftdomain=2, rightdomain=1, bc='particle')

geo.SetMaterial(1, 'electrolyte')
geo.SetMaterial(2, 'particle')

mesh = geo.GenerateMesh(maxh=100)
mesh.Save('mesh.vol')
