import random
import torch
import numpy as np
from collections import namedtuple, deque

Transition = namedtuple('Transition', 'img scalars a r n_img n_scalars d')
Batch = namedtuple('Batch', 'img scalars a r n_img n_scalars d')

class ReplayBuffer(object):
    def __init__(self, capacity: int, device=None):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.device = device if device is not None else torch.device('cpu')

    # Push new data
    def push(self, transition: Transition) -> None:
        self.memory.appendleft(transition)

    # Check if enough data is available
    def ready_for(self, batch_size: int) -> bool:
        return len(self.memory) >= batch_size

    def sample(self, batch_size: int) -> Batch:
        batch = random.sample(self.memory, batch_size)
        batch = Batch(*zip(*batch))

        # Convert lists to arrays
        img_array = np.array(batch.img)  # Shape: (batch_size, H, W)
        scalars_array = np.array(batch.scalars)  # Shape: (batch_size, scalar_dim)
        a_array = np.array(batch.a)
        r_array = np.array(batch.r)
        n_img_array = np.array(batch.n_img)
        n_scalars_array = np.array(batch.n_scalars)
        d_array = np.array(batch.d)

        batch_size_actual = len(batch.img)  # In case batch_size differs

        # Convert to tensors
        img = torch.tensor(img_array, dtype=torch.float).to(self.device)  # Shape: (batch_size, H, W)
        img = img.unsqueeze(1)  # Shape: (batch_size, 1, H, W)
        img = img.repeat(1, 3, 1, 1)  # Shape: (batch_size, 3, H, W)

        scalars = torch.tensor(scalars_array, dtype=torch.float).to(self.device)  # Shape: (batch_size, scalar_dim)
        a = torch.tensor(a_array, dtype=torch.float).view(batch_size_actual, -1).to(self.device)
        r = torch.tensor(r_array, dtype=torch.float).view(batch_size_actual, 1).to(self.device)

        n_img = torch.tensor(n_img_array, dtype=torch.float).to(self.device)
        n_img = n_img.unsqueeze(1)
        n_img = n_img.repeat(1, 3, 1, 1)

        n_scalars = torch.tensor(n_scalars_array, dtype=torch.float).to(self.device)
        d = torch.tensor(d_array, dtype=torch.float).view(batch_size_actual, 1).to(self.device)

        return Batch(img, scalars, a, r, n_img, n_scalars, d)
