# from proj_utilities import *
import proj_utilities
import types

print([getattr(proj_utilities, a) for a in dir(proj_utilities)
  if isinstance(getattr(proj_utilities, a), types.FunctionType)])

class C():

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

def f(inst, extra):
    colset, op = extra.split("=")
    l = [(i, getattr(inst, i)) for i in lcol]
    d = dict(l)
    print(l, d)
    setattr(inst, colset, eval(op, d))
    print(inst.a, inst.b, inst.c)
    return inst

inst = C(None,1,2)

extra = "a=b+c"
lcol = ["a", "b", "c"]
inst = f(inst, extra)
print(inst.a, inst.b, inst.c)
