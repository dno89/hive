"""Simple top-down 2D simulator built on pymunk."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

import pymunk
from pydantic import BaseModel
import numpy as np

from hive.utils.util import clamp


class MotionAction(BaseModel):
    """Action in the entity-local frame."""

    longitudinal_accel: float = 0.0
    lateral_accel: float = 0.0
    angular_velocity: float = 0.0


class MotionConfig(BaseModel):
    """Per-entity motion limits in the local frame."""

    min_longitudinal_accel: float = -4.0
    max_longitudinal_accel: float = 4.0
    min_lateral_accel: float = -4.0
    max_lateral_accel: float = 4.0
    max_speed: float = 20.0
    min_angular_velocity: float = -10.0
    max_angular_velocity: float = 10.0


class MotionModel:
    """Maps motion actions into pymunk force and velocity updates."""

    def __init__(self, default_config: MotionConfig | None = None) -> None:
        self.default_config = default_config or MotionConfig()

    def apply_action(
        self,
        body: pymunk.Body,
        action: MotionAction,
        config: MotionConfig | None = None,
    ) -> None:
        cfg = config or self.default_config

        cos_heading = math.cos(body.angle)
        sin_heading = math.sin(body.angle)

        # Acceleration in body (b) coordinates
        a_x_b = clamp(
            action.longitudinal_accel,
            cfg.min_longitudinal_accel,
            cfg.max_longitudinal_accel,
        )
        a_y_b = clamp(
            action.lateral_accel, cfg.min_lateral_accel, cfg.max_lateral_accel
        )
        # Acceleration in world (w) coordinates
        a_x_w = a_x_b * cos_heading - a_y_b * sin_heading
        a_y_w = a_x_b * sin_heading + a_y_b * cos_heading

        # Forces
        f_x_w = a_x_w * body.mass
        f_y_w = a_y_w * body.mass
        body.apply_force_at_world_point((f_x_w, f_y_w), body.position)

        # Angular velocity (Z-axis) is the same in body and world coordinates.
        ang_vel = clamp(
            action.angular_velocity, cfg.min_angular_velocity, cfg.max_angular_velocity
        )
        body.angular_velocity = ang_vel

        # Limit the speed
        speed = body.velocity.length
        if speed > cfg.max_speed:
            body.velocity = body.velocity * (cfg.max_speed / speed)


Action = MotionAction


class Simulator:
    """Simulator with circular entities and targets."""

    def __init__(
        self,
        dt: float = 0.02,
        motion_model: MotionModel | None = None,
        world_size: Tuple[float, float] = (100.0, 200.0),
        wall_thickness: float = 1.0,
    ):
        self.dt = dt
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.world_size = world_size
        self.wall_thickness = wall_thickness
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._dynamic_targets: Dict[str, Dict[str, Any]] = {}
        self._static_targets: Dict[str, Dict[str, Any]] = {}
        self._motion_configs: Dict[str, MotionConfig] = {}
        self.motion_model = motion_model or MotionModel()
        self._shape_to_id: Dict[pymunk.Shape, str] = {}
        self._id_to_kind: Dict[str, str] = {}
        self._events: List[Dict[str, Any]] = []

        self._add_bounds()
        self.space.on_collision(begin=self._on_collision_begin)

    def add_entity(
        self,
        entity_id: str,
        position: Tuple[float, float],
        radius: float = 10.0,
        mass: float = 1.0,
        heading: float = 0.0,
        motion_config: MotionConfig | None = None,
    ) -> None:
        """Add a dynamic entity controlled via actions."""
        body, shape = self._create_dynamic_circle(position, radius, mass, heading)
        self._entities[entity_id] = {"body": body, "shape": shape}
        if motion_config is not None:
            self._motion_configs[entity_id] = motion_config
        self._register_shape(entity_id, shape, kind="entity")

    def add_static_target(
        self,
        target_id: str,
        position: Tuple[float, float],
        radius: float = 10.0,
    ) -> None:
        """Add a static target that never moves."""
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = position
        shape = pymunk.Circle(body, radius)
        self.space.add(body, shape)
        self._static_targets[target_id] = {"body": body, "shape": shape}
        self._register_shape(target_id, shape, kind="static_target")

    def add_dynamic_target(
        self,
        target_id: str,
        position: Tuple[float, float],
        radius: float = 1.0,
        mass: float = 1.0,
        heading: float = 0.0,
        motion_config: MotionConfig | None = None,
    ) -> None:
        """Add a dynamic target that can be moved like an entity."""
        body, shape = self._create_dynamic_circle(position, radius, mass, heading)
        self._dynamic_targets[target_id] = {"body": body, "shape": shape}
        if motion_config is not None:
            self._motion_configs[target_id] = motion_config
        self._register_shape(target_id, shape, kind="dynamic_target")

    def step(
        self, actions: Dict[str, MotionAction] | None = None, dt: float | None = None
    ) -> List[Dict[str, Any]]:
        """Advance the simulation by one step and return collision events."""
        self._events = []
        actions = actions or {}
        step_dt = dt if dt is not None else self.dt

        for entity_id, action in actions.items():
            body = self._get_dynamic_body(entity_id)
            if body is None:
                continue
            motion_config = self._motion_configs.get(entity_id)
            self.motion_model.apply_action(body, action, motion_config)

        self.space.step(step_dt)
        return list(self._events)

    def get_entity_state(self, entity_id: str) -> Dict[str, Any]:
        """Return state for a single entity or target."""
        body = self._get_body(entity_id)
        if body is None:
            raise KeyError(f"Unknown entity id: {entity_id}")
        return self._build_state(entity_id, body)

    def get_states(self) -> Dict[str, Dict[str, Any]]:
        """Return state for all entities and targets."""
        states: Dict[str, Dict[str, Any]] = {}
        for entity_id in self._all_ids():
            body = self._get_body(entity_id)
            if body is None:
                continue
            states[entity_id] = self._build_state(entity_id, body)
        return states

    def get_events(self) -> List[Dict[str, Any]]:
        """Return the most recent events list."""
        return list(self._events)

    def debug_dump(self) -> Dict[str, Any]:
        """Return a debug snapshot of the simulator's entities and targets."""
        return {
            "entities": sorted(self._entities.keys()),
            "dynamic_targets": sorted(self._dynamic_targets.keys()),
            "static_targets": sorted(self._static_targets.keys()),
            "states": self.get_states(),
        }

    def debug_draw(self, draw_options: pymunk.SpaceDebugDrawOptions) -> None:
        """Draw the current space using pymunk debug draw options."""
        self.space.debug_draw(draw_options)

    def _create_dynamic_circle(
        self,
        position: Tuple[float, float],
        radius: float,
        mass: float,
        heading: float,
    ) -> Tuple[pymunk.Body, pymunk.Shape]:
        moment = pymunk.moment_for_circle(mass, 0.0, radius)
        body = pymunk.Body(mass, moment)
        body.position = position
        body.angle = heading
        shape = pymunk.Circle(body, radius)
        self.space.add(body, shape)
        return body, shape

    def _add_bounds(self) -> None:
        width, height = self.world_size
        thickness = self.wall_thickness
        static_body = self.space.static_body
        segments = [
            pymunk.Segment(static_body, (0.0, 0.0), (width, 0.0), thickness),
            pymunk.Segment(static_body, (width, 0.0), (width, height), thickness),
            pymunk.Segment(static_body, (width, height), (0.0, height), thickness),
            pymunk.Segment(static_body, (0.0, height), (0.0, 0.0), thickness),
        ]
        for idx, segment in enumerate(segments):
            segment.elasticity = 1.0
            self.space.add(segment)
            self._register_shape(f"boundary_{idx}", segment, kind="boundary")

    def _register_shape(self, entity_id: str, shape: pymunk.Shape, kind: str) -> None:
        self._shape_to_id[shape] = entity_id
        self._id_to_kind[entity_id] = kind

    def _all_ids(self) -> Iterable[str]:
        return (
            list(self._entities.keys())
            + list(self._dynamic_targets.keys())
            + list(self._static_targets.keys())
        )

    def _get_body(self, entity_id: str) -> pymunk.Body | None:
        if entity_id in self._entities:
            return self._entities[entity_id]["body"]
        if entity_id in self._dynamic_targets:
            return self._dynamic_targets[entity_id]["body"]
        if entity_id in self._static_targets:
            return self._static_targets[entity_id]["body"]
        return None

    def _get_dynamic_body(self, entity_id: str) -> pymunk.Body | None:
        if entity_id in self._entities:
            return self._entities[entity_id]["body"]
        if entity_id in self._dynamic_targets:
            return self._dynamic_targets[entity_id]["body"]
        return None

    def _build_state(self, entity_id: str, body: pymunk.Body) -> Dict[str, Any]:
        kind = self._id_to_kind.get(entity_id, "unknown")
        return {
            "id": entity_id,
            "kind": kind,
            "position": (float(body.position.x), float(body.position.y)),
            "heading": float(body.angle),
            "velocity": (float(body.velocity.x), float(body.velocity.y)),
            "angular_velocity": float(body.angular_velocity),
        }

    def _on_collision_begin(
        self, arbiter: pymunk.Arbiter, _space: pymunk.Space, _data: dict
    ) -> bool:
        shapes = arbiter.shapes
        id_a = self._shape_to_id.get(shapes[0], "unknown")
        id_b = self._shape_to_id.get(shapes[1], "unknown")
        points = [
            (float(point.point_a.x), float(point.point_a.y))
            for point in arbiter.contact_point_set.points
        ]
        self._events.append(
            {
                "type": "collision",
                "a": id_a,
                "b": id_b,
                "points": points,
            }
        )
        return True
