"""Geometry for Lithium-Ion Battery Model

Length scale in micrometers (µm).
"""

import netgen.geom2d as geom2d

thickness_cathode = 174
thickness_separator = 50
h = thickness_separator + thickness_cathode
w = 200

geo = geom2d.SplineGeometry()
pnts = [(0, 0), (w, 0), (w, h), (0, h)]
p1, p2, p3, p4 = [geo.AddPoint(*pnt) for pnt in pnts]
geo.Append(['line', p1, p2], leftdomain=1, rightdomain=0, bc='anode')
geo.Append(['line', p2, p3], leftdomain=1, rightdomain=0, bc='wall')
geo.Append(['line', p3, p4], leftdomain=1, rightdomain=0, bc='cathode')
geo.Append(['line', p4, p1], leftdomain=1, rightdomain=0, bc='wall')

geo.SetMaterial(1, 'electrolyte')

mesh = geo.GenerateMesh(maxh=10)
mesh.Save('mesh.vol')
