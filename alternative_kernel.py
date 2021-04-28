from mulearn.kernel import Kernel

class LengthDistanceKernel(Kernel):
    """Linear kernel class."""

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

        def _length_distance(ax1, ax2):
            return abs(len(ax1) - len(ax2)) / max(len(ax1), len(ax2))
        
        def length_distance(ax1, ax2):
            sign_negated = 1

            ax1 = ax1.reshape(-1)
            ax1_clean = ax1[0][2:-1]

            if ax1_clean[0] == '-':
                sign_negated *= -1
                ax1_clean = ax1_clean[1:]

            ax2 = ax2.reshape(-1)
            ax2_clean = ax2[0][2:-1]

            if ax2_clean[0] == '-':
                sign_negated *= -1
                ax2_clean = ax2_clean[1:]
            
            e = _length_distance(ax1_clean, ax2_clean)
            
            assert(0 <= e <= 1)
            
            return e if sign_negated == 1 else 1-e   
        return length_distance(arg_1,arg_2)

    def __repr__(self):
        """Return the python representation of the kernel."""
        return 'LengthDistanceKernel()'
    
class LevenshteinKernel(Kernel):
    """Linear kernel class."""

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

        def edit_distance(ax1, ax2):
            sign_negated = 1

            ax1 = ax1.reshape(-1)
            ax1_clean = ax1[0][2:-1]
            if ax1_clean[0] == '-':
                sign_negated *= -1
                ax1_clean = ax1_clean[1:]

            ax2 = ax2.reshape(-1)
            ax2_clean = ax2[0][2:-1]
            if ax2_clean[0] == '-':
                sign_negated *= -1
                ax2_clean = ax2_clean[1:]

        return edit_distance(arg_1,arg_2)

    def __repr__(self):
        """Return the python representation of the kernel."""
        return 'LevenshteinKernel()'

class HammingKernel(Kernel):
    """Linear kernel class."""

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

        def hamming(ax1, ax2):
            sign_negated = 1

            ax1 = ax1.reshape(-1)
            ax1_clean = ax1[0][2:-1]
            if ax1_clean[0] == '-':
                sign_negated *= -1
                ax1_clean = ax1_clean[1:]

            ax2 = ax2.reshape(-1)
            ax2_clean = ax2[0][2:-1]
            if ax2_clean[0] == '-':
                sign_negated *= -1
                ax2_clean = ax2_clean[1:]

            #print("frase1 : " + ax1_clean)
            #print("frase2 : " + ax2_clean)
            pairs = list(zip(ax1_clean, ax2_clean))
            #for c1,c2 in pairs:
                #print("coppia: " + c1 + c2)
                
            h = sum([ch1 != ch2
                    for ch1, ch2 in pairs]) / (min(len(ax1_clean), len(ax2_clean)))
            #print("distanza: " + str(h))
            assert(0 <= h <= 1)
            return h if sign_negated == 1 else 1-h

        return hamming(arg_1,arg_2)

    def __repr__(self):
        """Return the python representation of the kernel."""
        return 'HammingKernel()'