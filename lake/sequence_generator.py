"""SequenceGenerator class."""
import numpy as np
import string

class SequenceGenerator:
  """XXXXXX"""
  letters = np.array(list(string.ascii_uppercase))

  def __init__(self, characters, seq_length):
    self.characters = characters
    self.length = seq_length
    self.core_sequence = self._create_core_sequence()
    self.sequence = self._create_sequence()

  def _create_core_sequence(self):
    seq_num = np.arange(0, self.characters)
    seq_letters = self.letters[seq_num]
    seq = [seq_letters[a] + seq_letters[a + 1] for a in range(0, self.characters, 2)]
    return seq

  def _create_sequence(self):
    first_letters = self.letters[range(0, self.characters, 2)]
    second_letters = self.letters[range(1, self.characters, 2)]
    core_idx = np.random.randint(0, self.characters/2)
    seq = list([self.core_sequence[core_idx]])

    while len(seq) < self.length:
      next_seq = np.delete(first_letters, core_idx)
      next_pair = [second_letters[core_idx] + next_seq[a] for a in range(0, len(next_seq))]
      idx = np.random.randint(0, len(next_seq))
      seq.extend([next_pair[idx]])
      core_idx = np.where(list(next_pair[idx])[1] == first_letters)[0][0]
      seq.extend([self.core_sequence[core_idx]])
    return seq




