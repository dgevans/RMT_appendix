__author__ = 'dgevans'
import numpy as np

class UQL(object):
    @staticmethod
    def u(c,l,Para):
        gamma = Para.gamma
        return c-l**(1+gamma)/(1+gamma)

    @staticmethod
    def uc(c,l,Para):
        return np.ones(c.shape)

    @staticmethod
    def ul(c,l,Para):
        return -l**(Para.gamma)

    @staticmethod
    def ucc(c,l,Para):
        return np.zeros(c.shape)

    @staticmethod
    def ull(c,l,Para):
        return -Para.gamma*l**(Para.gamma-1)

class UCES(object):
    @staticmethod
    def u(c,l,Para):
        sigma = Para.sigma
        gamma = Para.gamma
        if sigma == 1:
            return np.log(c) - l**(1+gamma)/(1+gamma)
        else:
            return c**(1-sigma)/(1-sigma)-l**(1+gamma)/(1+gamma)

    @staticmethod
    def uc(c,l,Para):
        sigma = Para.sigma
        return c**(-sigma)

    @staticmethod
    def ul(c,l,Para):
        return -l**(Para.gamma)

    @staticmethod
    def ucc(c,l,Para):
        return -Para.sigma*c**(-Para.sigma-1.0)

    @staticmethod
    def ull(c,l,Para):
        return -Para.gamma*l**(Para.gamma-1)

class UCES_AMSS(object):
    @staticmethod
    def u(c,l,Para):
        sigma_1 = Para.sigma_1
        sigma_2 = Para.sigma_2
        eta = Para.eta
        if sigma_1 == 1:
            u = np.log(c)
        else:
            u =  c**(1-sigma_1)/(1-sigma_1)
        if sigma_2 == 1:
            u += eta*np.log(1-l)
        else:
            u += eta*(1-l)**(1-sigma_2)/(1-sigma_2)
        return u

    @staticmethod
    def uc(c,l,Para):
        sigma_1 = Para.sigma_1
        return c**(-sigma_1)

    @staticmethod
    def ul(c,l,Para):
        sigma_2 = Para.sigma_2
        eta = Para.eta
        return -eta*(1-l)**(-sigma_2)

    @staticmethod
    def ucc(c,l,Para):
        return -Para.sigma_1*c**(-Para.sigma_1-1.0)

    @staticmethod
    def ull(c,l,Para):
        return -Para.eta*Para.sigma_2*(1-l)**(-Para.sigma_2-1.0)


class parameters(object):

    def __init__(self):
        pass

    sigma = 2.0

    gamma = 2.0

    theta = 1.0

    g = [.1,.2]

    beta = 0.9

    U = UCES

    P = np.array([[0.5,0.5],[0.5,0.5]])

    xmin = -2.0

    xmax = 2.0

    nx = 20

    cloud = False
    
    transfers = False
    
    port = None

    def I(self,c,l):
        return c*self.U.uc(c,l,self)+l*self.U.ul(c,l,self)


class DictWrap(object):
    """ Wrap an existing dict, or create a new one, and access with either dot
      notation or key lookup.

      The attribute _data is reserved and stores the underlying dictionary.
      When using the += operator with create=True, the empty nested dict is
      replaced with the operand, effectively creating a default dictionary
      of mixed types.

      args:
        d({}): Existing dict to wrap, an empty dict is created by default
        create(True): Create an empty, nested dict instead of raising a KeyError

      example:
        >>>dw = DictWrap({'pp':3})
        >>>dw.a.b += 2
        >>>dw.a.b += 2
        >>>dw.a['c'] += 'Hello'
        >>>dw.a['c'] += ' World'
        >>>dw.a.d
        >>>print dw._data
        {'a': {'c': 'Hello World', 'b': 4, 'd': {}}, 'pp': 3}

    """

    def __init__(self, d=None, create=True):
        if d is None:
            d = {}
        supr = super(DictWrap, self)
        supr.__setattr__('_data', d)
        supr.__setattr__('__create', create)

    def __getattr__(self, name):
        try:
            value = self._data[name]
        except KeyError:
            if not super(DictWrap, self).__getattribute__('__create'):
                raise
            value = {}
            self._data[name] = value

        if hasattr(value, 'items'):
            create = super(DictWrap, self).__getattribute__('__create')
            return DictWrap(value, create)
        return value

    def __setattr__(self, name, value):
        self._data[name] = value

    def __getitem__(self, key):
        try:
            value = self._data[key]
        except KeyError:
            if not super(DictWrap, self).__getattribute__('__create'):
                raise
            value = {}
            self._data[key] = value

        if hasattr(value, 'items'):
            create = super(DictWrap, self).__getattribute__('__create')
            return DictWrap(value, create)
        return value

    def __setitem__(self, key, value):
        self._data[key] = value

    def __iadd__(self, other):
        if self._data:
            raise TypeError("A Nested dict will only be replaced if it's empty")
        else:
            return other
