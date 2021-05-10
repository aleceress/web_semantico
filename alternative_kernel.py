from mulearn.kernel import Kernel
from nltk.metrics.distance import edit_distance 

class LengthDistanceKernel(Kernel):

    def compute(self, arg_1, arg_2):
        ax1 = arg_1.reshape(-1)
        ax1_clean = ax1[0][2:-1]

        ax2 = arg_2.reshape(-1)
        ax2_clean = ax2[0][2:-1]
        return abs(len(ax1_clean) - len(ax2_clean)) / max(len(ax1_clean), len(ax2_clean))

    def __repr__(self):
        """Return the python representation of the kernel."""
        return 'LengthDistanceKernel()'
    
class LevenshteinKernel(Kernel):

    def compute(self, arg_1, arg_2):
        ax1 = arg_1.reshape(-1)
        ax1_clean = ax1[0][2:-1]

        ax2 = arg_2.reshape(-1)
        ax2_clean = ax2[0][2:-1]

        return edit_distance(ax1_clean,ax2_clean)/(max(len(ax1), len(ax2)))

    def __repr__(self):
        """Return the python representation of the kernel."""
        return 'LevenshteinKernel()'

class HammingKernel(Kernel):

    def compute(self, arg_1, arg_2):
        ax1 = arg_1.reshape(-1)
        ax1_clean = ax1[0][2:-1]
           
        ax2 = arg_2.reshape(-1)
        ax2_clean = ax2[0][2:-1]
         
        pairs = list(zip(ax1_clean, ax2_clean))

        h = sum([ch1 != ch2
                    for ch1, ch2 in pairs]) / (min(len(ax1_clean), len(ax2_clean)))
        return h 

    def __repr__(self):
        """Return the python representation of the kernel."""
        return 'HammingKernel()'