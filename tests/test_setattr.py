from inspect import signature

def auto_args(f):
    """
    Decorator for auto assignation of arguments during class instanciation.
    Code by judy2k from StackOverflow :
    https://stackoverflow.com/questions/3652851/what-is-the-best-way-to-do-automatic-attribute-assignment-in-python-and-is-it-a
    """
    sig = signature(f)  # Get a signature object for the target:
    def replacement(self, *args, **kwargs):
        # Parse the provided arguments using the target's signature:
        bound_args = sig.bind(self, *args, **kwargs)
        # Save away the arguments on `self`:
        for k, v in bound_args.arguments.items():
            if k != 'self':
                setattr(self, k, v)
        # Call the actual constructor for anything else:
        f(self, *args, **kwargs)
    return replacement

class A():

    def __init__(self, a):
        self.a = a

class C1():

    @auto_args
    def __init__(self, a, b, c):
        pass

class C2():

    @auto_args
    def __init__(self, alpha=1, beta=1, gamma=1):
        pass

    def __str__(self):
        return str(self.__dict__)

def getattr_recur(obj, elmt):
    elmts = elmt.split(".")
    if len(elmts)==1:
        return getattr(obj, elmts[0])
    else:
        return getattr_recur(obj, "".join(elmts[1:]))

if __name__ == "__main__":
    dict0 = {"a.a": "alpha",
    "b": "beta",
    "c": "gamma"}

    a1 = A(1)
    a2 = A(a1)
    c1 = C1(a2,2,3)
    c2 = C2()

    for i in dict0.items():
        print(i)
        setattr(c2,i[1],getattr_recur(c1,i[0]))
        print(getattr(c2,i[1]))
    print(c2)
    print(c1.a.a.a)
