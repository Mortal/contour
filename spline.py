import numpy as np


class BezierN:
    r'''
    >>> print(BezierN([42.0])([0, 1]))
    [ 42.  42.]
    >>> print(BezierN([[42.0, 24.0]])(0.5))
    [ 42.  24.]
    >>> print(BezierN([0, 100])(0.5))
    50.0
    >>> canvas = np.full((10, 10), ' ')
    >>> b = BezierN([[0, 0], [9, 0], [15, 14], [0, 3]])
    >>> xy = b(np.linspace(0, 1)).astype(np.intp)
    >>> canvas[tuple(xy.T)] = '*'
    >>> print('\n'.join(''.join(r).rstrip() for r in canvas))
    *  *
    *   *
    *   *
    *    *
    **   *
     *    *
      *   *
       *   *
        ****
         **
    '''

    _output_shape = (-1,)

    def __init__(self, control_points):
        control_points = np.asarray(control_points)
        self._dim_shape = control_points.shape[1:]
        if control_points.ndim == 2:
            self.c = control_points
        elif control_points.ndim == 1:
            self._output_shape = ()
            self.c = control_points.reshape(-1, 1)
        else:
            raise ValueError('invalid number of dimensions (expected 1 or 2)')

    def _eval(self, i, j, t, c):
        if i + 1 == j:
            return np.tile(c[i].reshape(1, -1), (len(t), 1))
        return (1 - t) * self._eval(i, j-1, t, c) + t * self._eval(i+1, j, t, c)

    def __call__(self, t):
        t = np.asarray(t)
        output_shape = t.shape + self._output_shape
        t = t.reshape((t.size,) + self._output_shape)
        c = self.c.reshape((-1,) + (1,)*t.ndim + self.c.shape[1:])
        return self._eval(0, len(c), t, c).reshape(output_shape)


class Bezier3:
    '''
    >>> c = [0, 9, 15, 0]
    >>> print(BezierN(c)(0.314159) - Bezier3(c)(0.314159))
    0.0
    >>> c2 = [[0, 0], [9, 0], [15, 14], [0, 3]]
    >>> t = np.linspace(0, 1)
    >>> np.allclose(BezierN(c2)(t), Bezier3(c2)(t))
    True
    '''

    _output_shape = (-1,)

    def __init__(self, control_points):
        control_points = np.asarray(control_points)
        self._dim_shape = control_points.shape[1:]
        if control_points.ndim == 2:
            self.c = control_points
        elif control_points.ndim == 1:
            self._output_shape = ()
            self.c = control_points.reshape(-1, 1)
        else:
            raise ValueError('invalid number of dimensions (expected 1 or 2)')
        if self.c.shape[0] != 4:
            raise ValueError('cubic bezier requires 4 control points')

    def __call__(self, t):
        t = np.asarray(t)
        output_shape = t.shape + self._output_shape
        t = t.reshape((t.size,) + self._output_shape)

        r = 1-t
        p0, p1, p2, p3 = self.c.reshape((4, 1) + self._output_shape)
        c01 = r * p0 + t * p1
        c12 = r * p1 + t * p2
        c23 = r * p2 + t * p3
        c02 = r * c01 + t * c12
        c13 = r * c12 + t * c23
        c03 = r * c02 + t * c13

        return c03.reshape(output_shape)


class Piecewise:
    '''
    >>> def curve(t):
    ...     print('curve() called with', t)
    ...     return t
    >>> curve._output_shape = curve._dim_shape = ()
    >>> c = Piecewise([curve, curve])
    >>> print(c([0, 0.5, 0.1, 0.2, 0.9, 1]))
    curve() called with [ 0.   0.2  0.4]
    curve() called with [ 0.   0.8  1. ]
    [ 0.   0.   0.2  0.4  0.8  1. ]
    '''

    def __init__(self, curves):
        if len(curves) == 0:
            raise ValueError('curves list must not be empty')
        self._curves = curves
        self._output_shape = curves[0]._output_shape
        self._dim_shape = curves[0]._dim_shape

    def __call__(self, t):
        t = np.asarray(t)
        output_shape = t.shape + self._output_shape
        t = t.ravel()

        res = np.empty((t.size,) + self._dim_shape)
        q, r = np.divmod(t * len(self._curves), 1.0)
        q = q.astype(np.intp)

        lo = q < 0
        hi = q >= len(self._curves)
        r[lo] = q[lo] = 0
        r[hi] = 1
        q[hi] = len(self._curves) - 1

        for idx in np.unique(q):
            f = (q == idx)
            res[f] = self._curves[idx](r[f])

        return res.reshape(output_shape)


class BSpline:
    '''
    Implementation of "relaxed uniform cubic B-spline curves" [DD].

    DD: http://www.math.ucla.edu/%7Ebaker/149.1.02w/handouts/dd_splines.pdf
    '''

    def __init__(self, control_points):
        control_points = np.asarray(control_points)
        shape = control_points.shape[1:]
        legs = np.diff(control_points, axis=0)
        v1 = legs / 3
        l1 = control_points[1:] - v1
        l2 = control_points[:-1] + v1
        subcontrol = np.empty((3 * len(control_points) - 2,) + shape)
        subcontrol[:] = -11
        subcontrol[0] = control_points[0]
        subcontrol[-1] = control_points[-1]
        mid = l1[:-1] + (l2[1:] - l1[:-1]) / 2
        subcontrol[3:-3:3] = mid
        subcontrol[1:-2:3] = l2
        subcontrol[2:-1:3] = l1
        pieces = []
        for i in range(len(control_points)-1):
            pieces.append(Bezier3(subcontrol[3*i:3*i+4]))
        self.subcontrol = subcontrol
        self.inner = Piecewise(pieces)

    def __call__(self, t):
        return self.inner(t)


def plot_spline(bs: BSpline, ax):
    subcontrol = bs.subcontrol
    assert (len(subcontrol) - 4) % 3 == 0

    for i in range(0, (len(subcontrol) - 4) // 3):
        apex = 2*subcontrol[3*i+2] - subcontrol[3*i+1]
        ax.plot(*np.transpose([subcontrol[3*i+2], apex, subcontrol[3*i+4]]),
                '-r')

    dashed = np.copy(subcontrol)
    dashed[3:-2:3] = dashed[2:-3:3] + (dashed[4:-1:3] - dashed[2:-3:3]) / 2
    ax.plot(*subcontrol.T, '--g')
    ax.plot(*subcontrol[1:-1].T, 'og')
    t = np.linspace(0, 1, 200)
    ax.plot(*bs(t).T, 'k')


def plot_construction(points=None, filename='construction.png'):
    import matplotlib.pyplot as plt

    if points is None:
        # Recreate [DD] Figure 11
        points = [
            [15.0301, 13.6246],
            [73.6897, 72.2842],
            [133.005, 73.2673],
            [251.307, 13.6246],
            [310.294, 72.9396],
            [192.647, 131.599],
        ]

    fig, ax = plt.subplots()
    points = np.asarray(points)
    bs = BSpline(points)
    plot_spline(bs, ax)
    ax.plot(*points.T, 'sb')
    fig.savefig(filename)


class SplineInterpolant:
    '''
    Implements "Interpolation by relaxed cubic splines" [DD].
    '''
    def __init__(self, data_points):
        data_points = np.asarray(data_points)
        if data_points.ndim == 1:
            data_points = data_points.reshape(-1, 1)
            self._output_shape = ()
        elif data_points.ndim == 2:
            self._output_shape = (-1,)
        else:
            raise ValueError('invalid number of dimensions (expected 1 or 2)')
        n = len(data_points)

        m = 4*np.eye(n - 2)
        m[1:, :-1] += np.eye(n - 3)
        m[:-1, 1:] += np.eye(n - 3)
        minv = np.linalg.inv(m)

        v = 6*data_points[1:-1]
        v[0] -= data_points[0]
        v[-1] -= data_points[-1]

        self.data_points = data_points
        self.control_points = np.concatenate(
            ([data_points[0]], minv @ v, [data_points[-1]]))
        self.b_spline = BSpline(self.control_points)

    def __call__(self, t):
        t = np.asarray(t)
        output_shape = t.shape + self._output_shape
        return self.b_spline(t.ravel()).reshape(output_shape)


def homework(points=None, filename='homework.png'):
    import matplotlib.pyplot as plt

    if points is None:
        # Recreate [DD] Figure 14
        points = [
            [1, -1],
            [-1, 2],
            [1, 4],
            [4, 3],
            [7, 5],
        ]

    curve = SplineInterpolant(points)
    fig, ax = plt.subplots()
    plot_spline(curve.b_spline, ax)
    fig.savefig(filename)


class PeriodicBSpline:
    '''
    Implementation of "periodic uniform cubic B-spline curves" [DD].
    Periodic instead of relaxed.

    DD: http://www.math.ucla.edu/%7Ebaker/149.1.02w/handouts/dd_splines.pdf
    '''

    def __init__(self, control_points):
        control_points = np.asarray(control_points)
        shape = control_points.shape[1:]
        legs = np.roll(control_points, -1, axis=0) - control_points
        # legs[i] = control_points[i+1] - control_points[i]
        subcontrol = np.empty((3 * len(control_points) + 1,) + shape)
        v1 = legs / 3
        subcontrol[1:-1:3] = control_points + v1
        subcontrol[2:-1:3] = control_points + 2 * v1
        subcontrol[:-1:3] = (subcontrol[1:-1:3] +
                             np.roll(subcontrol[2:-1:3], 1, axis=0)) / 2
        subcontrol[-1] = subcontrol[0]
        pieces = []
        for i in range(len(control_points)):
            pieces.append(Bezier3(subcontrol[3*i:3*i+4]))
        self.subcontrol = subcontrol
        self.inner = Piecewise(pieces)

    def __call__(self, t):
        return self.inner(t)


def plot_periodic_spline(bs: PeriodicBSpline, ax):
    subcontrol = bs.subcontrol

    for i in range(0, (len(subcontrol) - 1) // 3):
        apex = 2*subcontrol[3*i+2] - subcontrol[3*i+1]
        p = subcontrol[(3*i+4) % (len(subcontrol) - 1)]
        ax.plot(*np.transpose([subcontrol[3*i+2], apex, p]),
                '-r')

    dashed = np.copy(subcontrol)
    dashed[3:-2:3] = dashed[2:-3:3] + (dashed[4:-1:3] - dashed[2:-3:3]) / 2
    print(subcontrol)
    ax.plot(*subcontrol.T, 'o--g')
    t = np.linspace(0, 1, 200)
    ax.plot(*bs(t).T, 'k')


class PeriodicSplineInterpolant:
    '''
    Implements "Periodic spline interpolation", discussed briefly in [DD]
    end of section 9.  See also [PB] 5.2.

    PB: http://mirror.hmc.edu/ctan/graphics/pstricks/contrib/pst-bspline/pst-bspline-doc.pdf
    '''
    def __init__(self, data_points):
        data_points = np.asarray(data_points)
        if data_points.ndim == 1:
            data_points = data_points.reshape(-1, 1)
            self._output_shape = ()
        elif data_points.ndim == 2:
            self._output_shape = (-1,)
        else:
            raise ValueError('invalid number of dimensions (expected 1 or 2)')
        n = len(data_points)

        m = 4*np.eye(n)
        m[1:, :-1] += np.eye(n-1)
        m[:-1, 1:] += np.eye(n-1)
        m[-1, 0] += 1
        m[0, -1] += 1
        minv = np.linalg.inv(m)

        v = 6*data_points

        self.data_points = data_points
        self.control_points = minv @ v
        self.b_spline = PeriodicBSpline(self.control_points)

    def __call__(self, t):
        t = np.asarray(t)
        output_shape = t.shape + self._output_shape
        return self.b_spline(t.ravel()).reshape(output_shape)


def plot_periodic(points=None, filename='periodic.png'):
    import matplotlib.pyplot as plt

    if points is None:
        # Recreate [DD] Figure 14
        points = [
            [-1, 2],
            [1, 4],
            [4, 3],
            [1, 0],
        ]

    points = np.asarray(points)
    curve = PeriodicSplineInterpolant(points)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid()
    print(curve.control_points)
    plot_periodic_spline(curve.b_spline, ax)
    ax.plot(*points.T, 'ob')
    fig.savefig(filename, dpi=300)


if __name__ == '__main__':
    plot_periodic()
    homework()
    plot_construction()
