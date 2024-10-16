from gym import Env
from gym import spaces
from gym.envs.registration import register

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class QuantumParticleNavigator(Env):
    def __init__(self):
        super(QuantumParticleNavigator, self).__init__()

        # Action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1, high=2 * np.pi, shape=(20,), dtype=np.float32
        )

        # Environment parameters
        self.max_steps = 500
        self.goal_radius = 0.1
        self.dt = 0.05

        # Potential field parameters
        self.A = np.random.uniform(0.5, 1.5, size=5)
        self.B = np.random.uniform(-1, 1, size=(5, 5))
        self.ω = np.random.uniform(1, 2, size=5)
        self.φ = np.random.uniform(0, 2 * np.pi, size=5)

        self.reset()

    def reset(self):
        self.position = np.random.uniform(-0.9, 0.9, size=5)
        self.momentum = np.zeros(5)
        self.goal = np.random.uniform(-0.9, 0.9, size=5)
        self.phase = np.random.uniform(0, 2 * np.pi, size=5)
        self.steps = 0
        return self._get_obs()

    def step(self, action):
        self.steps += 1

        # Update momentum (p' = p + F*dt - ∇V*dt)
        self.momentum += action * self.dt - self._potential_gradient() * self.dt

        # Update position (x' = x + p*dt)
        self.position += self.momentum * self.dt
        self.position = np.clip(self.position, -1, 1)

        # Update phase (φ' = φ + (p² + V)*dt)
        self.phase += (
            np.sum(self.momentum**2) + self._potential(self.position)
        ) * self.dt
        self.phase = self.phase % (2 * np.pi)

        # Calculate reward
        distance = np.linalg.norm(self.position - self.goal)
        reward = -(distance**2) - 0.01 * np.sum(self.momentum**2)
        done = distance < self.goal_radius or self.steps >= self.max_steps

        if done and distance < self.goal_radius:
            reward += 100

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.concatenate(
            [self.position, self.momentum, self._potential_field(), self.phase]
        )

    def _potential(self, x):
        V = np.sum(self.A * np.sin(self.ω * x + self.φ))
        V += np.sum(self.B * np.cos(np.outer(x, x)))
        return V

    def _potential_field(self):
        return np.array([self._potential(self.position + 0.01 * e) for e in np.eye(5)])

    def _potential_gradient(self):
        eps = 1e-6
        return np.array(
            [
                (
                    self._potential(self.position + eps * e)
                    - self._potential(self.position - eps * e)
                )
                / (2 * eps)
                for e in np.eye(5)
            ]
        )

    def render(self, mode="human"):
        print(
            f"Step: {self.steps}, Position: {self.position}, Distance to goal: {np.linalg.norm(self.position - self.goal)}"
        )

    def visualize(self, trajectory):
        trajectory = np.array(trajectory)

        fig = plt.figure(figsize=(20, 15))

        # 1. PCA Plot
        ax1 = fig.add_subplot(231, projection="3d")
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(trajectory[:, :5])  # PCA on position
        ax1.plot(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2])
        ax1.set_title("PCA: Particle Trajectory")

        # 2. Parallel Coordinates Plot
        ax2 = fig.add_subplot(232)
        ax2.set_title("Parallel Coordinates: All Dimensions")

        # 3. Potential Field Heatmap
        ax3 = fig.add_subplot(233)
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self._potential(np.array([X[i, j], Y[i, j], 0, 0, 0]))
        c = ax3.pcolormesh(X, Y, Z, shading="auto")
        fig.colorbar(c, ax=ax3)
        ax3.set_title("Potential Field (2D slice)")

        # 4. Phase Space Plot
        ax4 = fig.add_subplot(234)
        ax4.plot(
            trajectory[:, 0], trajectory[:, 5]
        )  # position vs momentum for first dimension
        ax4.set_xlabel("Position (dim 1)")
        ax4.set_ylabel("Momentum (dim 1)")
        ax4.set_title("Phase Space (dim 1)")

        # 5. Goal Distance Plot
        ax5 = fig.add_subplot(235)
        distances = np.linalg.norm(trajectory[:, :5] - self.goal, axis=1)
        ax5.plot(distances)
        ax5.set_xlabel("Step")
        ax5.set_ylabel("Distance to Goal")
        ax5.set_title("Distance to Goal over Time")

        plt.tight_layout()
        plt.show()


register(
    id="QuantumParticleNavigator-v0",
    entry_point="envs.quantum:QuantumParticleNavigator",
)


def run_episode(env, max_steps=500):
    obs = env.reset()
    trajectory = [obs]
    for _ in range(max_steps):
        action = env.action_space.sample()  # Replace with your agent's action
        obs, reward, done, _ = env.step(action)
        trajectory.append(obs)
        if done:
            break
    return trajectory


if __name__ == "__main__":
    env = QuantumParticleNavigator()
    trajectory = run_episode(env)
    env.visualize(trajectory)
