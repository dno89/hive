"""Multi-agent environment with communication channels in observations."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from hive.sim.simulator import EntityKind, MotionAction, Simulator
from pydantic import BaseModel


class CommEnvConfig(BaseModel):
    """Configuration for the communication-enabled environment."""

    # general config
    num_agents: int = 1
    static_target_position: Tuple[float, float] = (0.0, 0.0)

    # sim config
    world_size: Tuple[float, float] = (100.0, 100.0)
    episode_length_s: float = 60.0
    sim_step_period_s: float = 1 / 20.0
    max_accel: float = 4.0
    max_angular_vel: float = 2.5

    # observation
    max_num_neighbors: int = 20
    comm_size: int = 8
    comm_range: float = 0.0
    obs_num_rays: int = 4
    obs_ray_length: float = 20.0
    obs_inv_distance_max: float = 10.0


class HiveMultiAgentCommEnv(MultiAgentEnv):
    """Multi-agent env that exposes received messages as part of observations."""

    def __init__(self, config: CommEnvConfig | None = None) -> None:
        self.config = config or CommEnvConfig()
        self._steps = 0
        self._max_steps = int(
            round(self.config.episode_length_s / self.config.sim_step_period_s)
        )
        self._agent_ids = [f"agent_{idx}" for idx in range(self.config.num_agents)]

        self._obs_distance_space = spaces.Box(
            low=0.0,
            high=self.config.obs_inv_distance_max,
            shape=(self.config.obs_num_rays,),
            dtype=np.float32,
        )
        self._obs_type_space = spaces.MultiDiscrete(
            [len(EntityKind) + 1] * self.config.obs_num_rays
        )
        self._obs_comm_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.config.max_num_neighbors, self.config.comm_size),
            dtype=np.float32,
        )
        self._obs_comm_mask_space = spaces.MultiBinary(self.config.max_num_neighbors)
        self.observation_space = spaces.Dict(
            {
                "ray_distance": self._obs_distance_space,
                "ray_kind": self._obs_type_space,
                "comm": self._obs_comm_space,
                "comm_mask": self._obs_comm_mask_space,
            }
        )

        self._act_motion_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._act_comm_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.config.comm_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Dict(
            {
                "motion": self._act_motion_space,
                "message": self._act_comm_space,
            }
        )

        self._sim = self._create_sim()
        self._messages = self._reset_messages()

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        self._steps = 0
        self._sim = self._create_sim()
        self._messages = self._reset_messages()
        observations = {
            agent_id: self._build_observation(agent_id) for agent_id in self._agent_ids
        }
        return observations, {}

    def step(self, action_dict: Dict[str, Dict[str, np.ndarray]]):
        self._steps += 1
        actions: Dict[str, MotionAction] = {}

        for agent_id, action in action_dict.items():
            motion = np.asarray(
                action.get("motion", np.zeros(3, dtype=np.float32)), dtype=np.float32
            )
            message = np.asarray(
                action.get("message", np.zeros(self.config.comm_size)), dtype=np.float32
            )
            self._messages[agent_id] = np.clip(message, -1.0, 1.0)
            actions[agent_id] = self._motion_from_action(motion)

        self._sim.step(actions)

        observations = {
            agent_id: self._build_observation(agent_id) for agent_id in self._agent_ids
        }
        rewards = {agent_id: self._reward_for(agent_id) for agent_id in self._agent_ids}
        truncations = {
            agent_id: self._steps >= self._max_steps for agent_id in self._agent_ids
        }
        terminations = {agent_id: False for agent_id in self._agent_ids}

        done_all = self._steps >= self._max_steps
        truncations["__all__"] = done_all
        terminations["__all__"] = done_all

        infos = {agent_id: {} for agent_id in self._agent_ids}

        return observations, rewards, terminations, truncations, infos

    def _create_sim(self) -> Simulator:
        sim = Simulator(dt=self.config.sim_step_period_s)
        width, height = self.config.world_size
        for idx, agent_id in enumerate(self._agent_ids):
            pos = (
                width * (0.2 + 0.6 * (idx / max(1, self.config.num_agents - 1))),
                height * 0.5,
            )
            sim.add_entity(agent_id, position=pos, radius=15.0)
        return sim

    def _max_neighbors(self) -> int:
        return self.config.max_num_neighbors

    def _reset_messages(self) -> Dict[str, np.ndarray]:
        return {
            agent_id: np.zeros(self.config.comm_size, dtype=np.float32)
            for agent_id in self._agent_ids
        }

    def _build_observation(self, agent_id: str) -> Dict[str, np.ndarray]:
        ray_distances, ray_kinds = self._collect_rays(agent_id)
        comm, comm_mask = self._collect_messages(agent_id)
        return {
            "ray_distance": ray_distances,
            "ray_kind": ray_kinds,
            "comm": comm,
            "comm_mask": comm_mask,
        }

    def _collect_rays(self, agent_id: str) -> tuple[np.ndarray, np.ndarray]:
        observation = self._sim.get_observation(
            agent_id,
            num_rays=self.config.obs_num_rays,
            ray_length=self.config.obs_ray_length,
        )
        inv_max = self.config.obs_inv_distance_max
        distances = np.zeros((self.config.obs_num_rays,), dtype=np.float32)
        kinds = np.zeros((self.config.obs_num_rays,), dtype=np.int64)
        for idx, ray in enumerate(observation.rays):
            inv_distance = 1.0 / max(ray.distance, 1e-6)
            distances[idx] = min(inv_distance, inv_max)
            kinds[idx] = self._encode_ray_kind(ray.hit_kind)
        return distances, kinds

    @staticmethod
    def _encode_ray_kind(hit_kind: EntityKind | None) -> int:
        if hit_kind is None:
            return 0
        return list(EntityKind).index(hit_kind) + 1

    def _collect_messages(self, agent_id: str) -> tuple[np.ndarray, np.ndarray]:
        messages = np.zeros(
            (self._max_neighbors(), self.config.comm_size), dtype=np.float32
        )
        mask = np.zeros((self._max_neighbors(),), dtype=bool)
        base_state = self._sim.get_entity_state(agent_id)
        base_pos = np.array(base_state.position, dtype=np.float32)
        slot = 0
        for other_id in self._agent_ids:
            if other_id == agent_id:
                continue
            if slot >= self._max_neighbors():
                break
            if self.config.comm_range > 0.0:
                other_state = self._sim.get_entity_state(other_id)
                other_pos = np.array(other_state.position, dtype=np.float32)
                if np.linalg.norm(other_pos - base_pos) > self.config.comm_range:
                    continue
            messages[slot] = self._messages[other_id]
            mask[slot] = True
            slot += 1
        return messages, mask

    def _motion_from_action(self, action: np.ndarray) -> MotionAction:
        action = np.clip(action, -1.0, 1.0)
        return MotionAction(
            longitudinal_accel=float(action[0] * self.config.max_accel),
            lateral_accel=float(action[1] * self.config.max_accel),
            angular_velocity=float(action[2] * self.config.max_angular_vel),
        )

    def _reward_for(self, agent_id: str) -> float:
        state = self._sim.get_entity_state(agent_id)
        pos = np.array(state.position, dtype=np.float32)
        target = np.array(self.config.static_target_position, dtype=np.float32)
        distance = np.linalg.norm(pos - target)
        return -float(distance)
