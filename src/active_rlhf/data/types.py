from typing import NamedTuple

import torch as th

class Trajectory(NamedTuple):
  obs: th.Tensor
  acts: th.Tensor
  dones: th.Tensor

class TrajectoryWithRew(Trajectory):
  rews: th.Tensor

class TrajectoryWithRewPair(NamedTuple):
  first: TrajectoryWithRew
  second: TrajectoryWithRew

class PreferencedTrajectoryWithRewPair(NamedTuple):
  first: TrajectoryWithRew
  second: TrajectoryWithRew
  preference: th.Tensor

