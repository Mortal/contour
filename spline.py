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
