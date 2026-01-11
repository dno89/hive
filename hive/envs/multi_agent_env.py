"""Multi-agent environment with communication channels in observations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from hive.sim.simulator import MotionAction, Simulator


@dataclass
class CommEnvConfig:
    """Configuration for the communication-enabled environment."""

    num_agents: int = 2
    episode_length: int = 200
    comm_size: int = 2
    world_size: Tuple[float, float] = (800.0, 600.0)
    target_position: Tuple[float, float] = (0.0, 0.0)
    max_accel: float = 200.0
    max_angular_vel: float = 4.0


class HiveMultiAgentCommEnv(MultiAgentEnv):
    """Multi-agent env that exposes received messages as part of observations."""

    def __init__(self, config: CommEnvConfig | None = None) -> None:
        self.config = config or CommEnvConfig()
        self._steps = 0
        self._agent_ids = [f"agent_{idx}" for idx in range(self.config.num_agents)]

        self._state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )
        self._comm_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._max_neighbors(), self.config.comm_size),
            dtype=np.float32,
        )
        self._comm_mask_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._max_neighbors(),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "self": self._state_space,
                "comm": self._comm_space,
                "comm_mask": self._comm_mask_space,
            }
        )
        self.action_space = spaces.Dict(
            {
                "motion": spaces.Box(
                    low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                    high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "message": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.config.comm_size,),
                    dtype=np.float32,
                ),
            }
        )

        self._sim = self._create_sim()
        self._messages = self._reset_messages()

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        self._steps = 0
        self._sim = self._create_sim()
        self._messages = self._reset_messages()
        observations = {agent_id: self._build_observation(agent_id) for agent_id in self._agent_ids}
        return observations, {}

    def step(self, action_dict: Dict[str, Dict[str, np.ndarray]]):
        self._steps += 1
        actions: Dict[str, MotionAction] = {}

        for agent_id, action in action_dict.items():
            motion = np.asarray(action.get("motion", np.zeros(3, dtype=np.float32)), dtype=np.float32)
            message = np.asarray(action.get("message", np.zeros(self.config.comm_size)), dtype=np.float32)
            self._messages[agent_id] = np.clip(message, -1.0, 1.0)
            actions[agent_id] = self._motion_from_action(motion)

        self._sim.step(actions)

        observations = {agent_id: self._build_observation(agent_id) for agent_id in self._agent_ids}
        rewards = {agent_id: self._reward_for(agent_id) for agent_id in self._agent_ids}
        truncations = {agent_id: self._steps >= self.config.episode_length for agent_id in self._agent_ids}
        terminations = {agent_id: False for agent_id in self._agent_ids}

        done_all = self._steps >= self.config.episode_length
        truncations["__all__"] = done_all
        terminations["__all__"] = done_all

        infos = {agent_id: {} for agent_id in self._agent_ids}

        return observations, rewards, terminations, truncations, infos

    def _create_sim(self) -> Simulator:
        sim = Simulator(dt=1.0 / 30.0)
        width, height = self.config.world_size
        for idx, agent_id in enumerate(self._agent_ids):
            pos = (width * (0.2 + 0.6 * (idx / max(1, self.config.num_agents - 1))), height * 0.5)
            sim.add_entity(agent_id, position=pos, radius=15.0)
        return sim

    def _reset_messages(self) -> Dict[str, np.ndarray]:
        return {agent_id: np.zeros(self.config.comm_size, dtype=np.float32) for agent_id in self._agent_ids}

    def _build_observation(self, agent_id: str) -> Dict[str, np.ndarray]:
        state = self._sim.get_entity_state(agent_id)
        self_obs = np.array(
            [
                state["position"][0],
                state["position"][1],
                state["velocity"][0],
                state["velocity"][1],
                state["heading"],
                state["angular_velocity"],
            ],
            dtype=np.float32,
        )
        comm, comm_mask = self._collect_messages(agent_id)
        return {"self": self_obs, "comm": comm, "comm_mask": comm_mask}

    def _collect_messages(self, agent_id: str) -> tuple[np.ndarray, np.ndarray]:
        messages = np.zeros((self._max_neighbors(), self.config.comm_size), dtype=np.float32)
        mask = np.zeros((self._max_neighbors(),), dtype=np.float32)
        slot = 0
        for other_id in self._agent_ids:
            if other_id == agent_id:
                continue
            if slot >= self._max_neighbors():
                break
            messages[slot] = self._messages[other_id]
            mask[slot] = 1.0
            slot += 1
        return messages, mask

    def _max_neighbors(self) -> int:
        return max(0, self.config.num_agents - 1)

    def _motion_from_action(self, action: np.ndarray) -> MotionAction:
        action = np.clip(action, -1.0, 1.0)
        return MotionAction(
            longitudinal_accel=float(action[0] * self.config.max_accel),
            lateral_accel=float(action[1] * self.config.max_accel),
            angular_velocity=float(action[2] * self.config.max_angular_vel),
        )

    def _reward_for(self, agent_id: str) -> float:
        state = self._sim.get_entity_state(agent_id)
        pos = np.array(state["position"], dtype=np.float32)
        target = np.array(self.config.target_position, dtype=np.float32)
        distance = np.linalg.norm(pos - target)
        return -float(distance)
