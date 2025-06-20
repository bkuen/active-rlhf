import torch as th


class RunningStat:
    """
    Tracks running mean and variance using Welford's algorithm.

    Works with 0-D scalars or multi-dimensional tensors and can be fed
    batches of samples at once.  All computations stay on the tensor's
    device (CPU or GPU).

    Example
    -------
    stat = RunningStat()
    stat.update(reward_batch)          # reward_batch shape (B,) or (B, …)
    normed = stat.normalize(reward)    # same shape as `reward`
    """

    def __init__(self, shape=(), device=None, dtype=th.float32):
        self.device = device or "cpu"
        self.mean = th.zeros(shape, device=self.device, dtype=dtype)
        self.var = th.ones(shape, device=self.device, dtype=dtype)
        self.count = th.tensor(0.0, device=self.device, dtype=dtype)

    # ------------------------------------------------------------------
    def update(self, x: th.Tensor):
        """
        Update running mean/var with a (batch of) sample(s).

        Parameters
        ----------
        x : torch.Tensor
            Shape (B,) or (B, …).  Leading batch dimension is collapsed;
            remaining dims define the statistic’s `shape`.
        """
        x = x.detach()  # avoid back-prop through stats
        if x.dim() == 0:  # make sure we have a batch dim
            x = x.unsqueeze(0)

        batch_count = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        # Welford’s parallel update
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        new_var = M2 / total_count

        # assign
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    # ------------------------------------------------------------------
    def normalize(self, x: th.Tensor, eps: float = 1e-8) -> th.Tensor:
        """Return a centred, unit-variance version of `x`."""
        return (x - self.mean) / (self.var.sqrt() + eps)

    def std(self, eps: float = 1e-8) -> th.Tensor:
        """Current running standard deviation."""
        return self.var.sqrt() + eps

    # convenient alias so you can call the object directly
    __call__ = normalize