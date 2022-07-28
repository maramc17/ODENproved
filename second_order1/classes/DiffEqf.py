class DiffEqf:
    """
     w" + (1/t)*w' + (ha^2)(1- (w/1- aw));  ha = 1, a=1.
     Bc's:  w'(0) = 0 and w(1) = 0
    """

    def __init__(self, diffeqf, x, y, dydx, d2ydx2, alpha, h):
        """
    	diffeq : name of the differential equation used (ex: diffeq = "first  ode")
    	"""
        self.DiffEqf = diffeqf
        self.x = x
        self.y = y
        self.dydx = dydx
        self.d2ydx2 = d2ydx2

        if self.DiffEqf == "first":
            self.eqf = self.d2ydx2 + (1 / self.x) * self.dydx + h ** 2 * (1 - (self.y / 1 - alpha * self.y))
