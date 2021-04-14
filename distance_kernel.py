import numpy as np

class Kernel:
"""Base kernel class."""

    def __init__(self):
    """Create an instance of :class:`Kernel`."""
    self.precomputed = False
    self.kernel_computations = None


    def compute(self, arg_1, arg_2):
    """Compute the kernel value, given two arguments.

    :param arg_1: First kernel argument.
    :type arg_1: Object
    :param arg_2: Second kernel argument.
    :type arg_2: Object
    :raises: NotImplementedError (:class:`Kernel` is abstract)
    :returns: `float` -- kernel value.
    """
    raise NotImplementedError(
        'The base class does not implement the `compute` method')


    def __str__(self):
        """Return the string representation of a kernel."""
        return self.__repr__()

    def __eq__(self, other):
        """Check kernel equality w.r.t. other objects."""
        return type(self) == type(other)

    def __ne__(self, other):
        """Check kernel inequality w.r.t. other objects."""
        return not self == other

    @staticmethod
    def __nonzero__():
        """Check if a kernel is non-null."""
        return True

    def __hash__(self):
        """Generate hashcode for a kernel."""
        return hash(self.__repr__())

    @classmethod
    def get_default(cls):
        """Return the default kernel.

        :returns: `LinearKernel()` -- the default kernel.
        """
        return LengthDistanceKernel
    



class LengthDistanceKernel(Kernel):
"""Linear kernel class."""

    def _length_distance(ax1, ax2):
        return abs(len(ax1) - len(ax2)) / max(len(ax1), len(ax2))

    def compute(self, arg_1, arg_2):
        r"""
        Compute the kernel value.

        The value $k(x_1, x_2)$ of a linear kernel is equal to the dot product
        $x_1 \cdot x_2$, that is to $\sum_{i=1}^n (x_1)_i (x_2)_i$, $n$ being
        the common dimension of $x_1$ and $x_2$.

        :param arg_1: First kernel argument.
        :type arg_1: iterable of `float`
        :param arg_2: Second kernel argument.
        :type arg_2: iterable of `float`
        :returns: `float` -- kernel value.
        """
        
        def get_data_matrix(file, name, function):
            if os.path.isfile(file):
                print('revrieving cached {} data matrix'.format(name))
                data_matrix = np.load(file)
            else:
                print('generating and caching {} data matrix'
                    ' (could take considerable time)...'.format(name), end=' ')
                data_matrix = np.array([[function(ax1, ax2)
                                        for ax1 in labels] for ax2 in labels])
                np.save(file, data_matrix)
                print('done!')
            return data_matrix

        def _length_distance(ax1, ax2):
            return abs(len(ax1) - len(ax2)) / max(len(ax1), len(ax2))

        def length_distance(ax1, ax2):
            sign_negated = 1

            ax1_clean = ax1[2:-1]
            if ax1_clean[0] == '-':
                sign_negated *= -1
                ax1_clean = ax1_clean[1:]

            ax2_clean = ax2[2:-1]
            if ax2_clean[0] == '-':
                sign_negated *= -1
                ax2_clean = ax2_clean[1:]
            
            e = _length_distance(ax1_clean, ax2_clean)
            
            assert(0 <= e <= 1)

        dist_length = get_data_matrix('length_distance.npy',
                            'length distance', length_distance)

        return dist_length[arg_1,arg_2]

    def __repr__(self):
        """Return the python representation of the kernel."""
        return 'LengthDistanceKernel()'

