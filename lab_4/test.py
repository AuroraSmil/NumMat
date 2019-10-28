# import symbol x from sympy so that you can define symbolic functions of x
import sympy as sp
from sympy.abc import x
# import symbolic integration
from sympy import integrate
import numpy as np
from functools import reduce


def lagrange_polys(xqs):
    n = len(xqs)
    Ls = []

    for i in range(n):
        L_i = sp.prod((x - xqs[j]) for j in range(n) if j!= i)

        # normalize:
        L_i = L_i / L_i.subs(x, xqs[i])
        Ls.append(L_i)

    return Ls


def newton_cotes_formula(n, a, b):
    xqs = np.linspace(a, b, n + 1)

    Ls = lagrange_polys(xqs)
    ws = []

    for i in range(n + 1):
        w_i = integrate(Ls[i], (x, a, b))
        ws.append(w_i)
    return (xqs, ws)


a, b = 0, 1
xqs = np.linspace(a,b,3)
lagrange_polys(xqs)

xs, ws = (newton_cotes_formula(3, 0,1))

n = 2
f = lambda x: x**n

def qr(f, xqs, ws):
    n = len(xqs)
    qr_f = np.sum(ws[i]*f(xqs[i]) for i in range(n))
    return qr_f



f = sp.cos(x)

integral = integrate(f, (x, -4, 5))
print(integral)