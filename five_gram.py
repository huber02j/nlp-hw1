import collections
from itertools import chain
import math

class Five_gram:
    """A 5-gram language model.

    data: a list of lists of symbols. They should not contain `<EOS>`;
          the `<EOS>` symbol is automatically appended during
          training.
    """
    
    def __init__(self, data):
        self.count = collections.defaultdict(lambda: collections.defaultdict(int))
        for line in data:
            q = self.start()
            for a in list(line) + ['<EOS>']:
                self.count[q][a] += 1
                q = self.read(q, a)

    def start(self):
        """Return the language model's start state.""" 
        
        return ('<BOS>', '<BOS>', '<BOS>', '<BOS>')

    def read(self, q, a):
        """Return the state that the model would be in if it's in state `q`
        and reads symbol `a`. Ex. ('<BOS>', '<BOS>', '<BOS>', '<BOS>') -> 
                                ('<BOS>', '<BOS>', '<BOS>', 'word') """
        return q[1:4] + tuple([a])

    def logprob(self, q, a):
        """Return the log-probability of `a` when the model is in state `q`."""
        denom = sum(self.count[q][c] + 0.01 for c in self.count[q])
        return math.log((self.count[q][a] + 0.01) / denom)

    def best(self, q):
        """Return the symbol with highest probability when the model is in 
        state `q`."""
        maxval = 0
        best_a = None
        for c in self.count[q]:
            if math.exp(self.logprob(q, c)) >= maxval:
                maxval = math.exp(self.logprob(q,c))
                best_a = c

        return best_a
