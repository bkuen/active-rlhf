import unittest
import torch as th
from active_rlhf.rewards.reward_nets import PreferenceModel, RewardEnsemble
from active_rlhf.data.buffers import PreferenceBufferBatch

class TestRewardModelOverfitting(unittest.TestCase):
    def test_overfit_synthetic_preferences(self):
        obs_dim = 11
        act_dim = 3
        fragment_length = 50
        device = "cpu"

        # Generate dummy data
        N = 64
        obs = th.randn(N, fragment_length, obs_dim)
        acts = th.randn(N, fragment_length, act_dim)

        # Define returns and build preferences
        returns = th.linspace(0, 1, N)
        indices = th.randperm(N)
        first_idx = indices[:N // 2]
        second_idx = indices[N // 2:]

        prefs = (returns[first_idx] > returns[second_idx]).float().unsqueeze(1)
        prefs = th.cat([prefs, 1.0 - prefs], dim=1)

        # Build batch
        batch = PreferenceBufferBatch(
            first_obs=obs[first_idx],
            first_acts=acts[first_idx],
            first_rews=None,
            first_dones=None,
            second_obs=obs[second_idx],
            second_acts=acts[second_idx],
            second_rews=None,
            second_dones=None,
            prefs=prefs
        )

        # Create reward model
        reward_ensemble = RewardEnsemble(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=[64, 64],
            ensemble_size=1,
            device=device,
        )
        preference_model = PreferenceModel(ensemble=reward_ensemble, device=device)

        # Train
        optimizer = th.optim.Adam(preference_model.parameters(), lr=1e-3)
        for i in range(200):
            optimizer.zero_grad()

            _, _, probs = preference_model(batch.first_obs, batch.first_acts, batch.second_obs, batch.second_acts)
            print("Probs shape", probs.shape)
            loss = preference_model.loss(probs, prefs)
            loss.backward()
            optimizer.step()

        _, _, probs = preference_model(batch.first_obs, batch.first_acts, batch.second_obs, batch.second_acts)
        final_loss = preference_model.loss(probs, prefs).item()

        self.assertLess(final_loss, 0.05, "Reward model failed to overfit synthetic prefs")

if __name__ == "__main__":
    unittest.main()