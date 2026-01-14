"""Simple top-down 2D simulator built on pymunk."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, Iterable, List, Tuple

import pymunk
from pydantic import BaseModel
import numpy as np

from hive.utils.util import clamp


class LatLonAccMotionModel:
    """Maps motion actions into pymunk force and velocity updates."""

    class Config(BaseModel):
        """Per-entity motion limits in the local frame."""

        min_longitudinal_accel: float = -4.0
        max_longitudinal_accel: float = 4.0
        min_lateral_accel: float = -4.0
        max_lateral_accel: float = 4.0
        max_speed: float = 20.0
        min_angular_velocity: float = -10.0
        max_angular_velocity: float = 10.0

    class Action(BaseModel):
        """Action in the entity-local frame."""

        longitudinal_accel: float = 0.0
        lateral_accel: float = 0.0
        angular_velocity: float = 0.0

    def __init__(self, default_config: Config | None = None) -> None:
        self.default_config = default_config or self.Config()

    def apply_action(
        self,
        body: pymunk.Body,
        action: Action,
        config: Config | None = None,
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


class EntityKind(str, Enum):
    WALL = "wall"
    TARGET = "target"
    ENTITY = "entity"


class EntityState(BaseModel):
    """Snapshot of a single entity or target."""

    id: str
    kind: EntityKind
    position: Tuple[float, float]
    heading: float
    velocity: Tuple[float, float]
    angular_velocity: float


class CollisionEvent(BaseModel):
    """Collision event between two shapes."""

    type: str = "collision"
    a: str
    b: str
    points: List[Tuple[float, float]]


class RayObservation(BaseModel):
    """Single ray observation result."""

    distance: float
    hit_id: str | None = None
    hit_kind: EntityKind | None = None


class EntityObservation(BaseModel):
    """Ray-based observation for an entity."""

    entity_id: str
    rays: List[RayObservation]


MotionModel = LatLonAccMotionModel
MotionAction = LatLonAccMotionModel.Action
MotionConfig = LatLonAccMotionModel.Config


class Simulator:
    """Simulator with circular entities and targets."""

    def __init__(
        self,
        dt: float = 0.02,
        motion_model: MotionModel | None = None,
        world_size: Tuple[float, float] = (100.0, 200.0),
        wall_thickness: float = 1.0,
        friction: float = 0.8,
        linear_damping: float = 1.0,
    ):
        self.dt = dt
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.world_size = world_size
        self.wall_thickness = wall_thickness
        self.friction = friction
        self.linear_damping = linear_damping
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._dynamic_targets: Dict[str, Dict[str, Any]] = {}
        self._static_targets: Dict[str, Dict[str, Any]] = {}
        self._motion_configs: Dict[str, MotionConfig] = {}
        self.motion_model = motion_model or MotionModel()
        self._shape_to_id: Dict[pymunk.Shape, str] = {}
        self._id_to_kind: Dict[str, EntityKind] = {}
        self._events: List[CollisionEvent] = []

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
        shape.friction = self.friction
        self._entities[entity_id] = {"body": body, "shape": shape}
        if motion_config is not None:
            self._motion_configs[entity_id] = motion_config
        self._register_shape(entity_id, shape, kind=EntityKind.ENTITY)

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
        shape.friction = self.friction
        self.space.add(body, shape)
        self._static_targets[target_id] = {"body": body, "shape": shape}
        self._register_shape(target_id, shape, kind=EntityKind.TARGET)

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
        shape.friction = self.friction
        self._dynamic_targets[target_id] = {"body": body, "shape": shape}
        if motion_config is not None:
            self._motion_configs[target_id] = motion_config
        self._register_shape(target_id, shape, kind=EntityKind.TARGET)

    def step(
        self, actions: Dict[str, MotionAction] | None = None, dt: float | None = None
    ) -> List[CollisionEvent]:
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
        if self.linear_damping > 0.0:
            self._apply_linear_damping(step_dt)
        return list(self._events)

    def get_entity_state(self, entity_id: str) -> EntityState:
        """Return state for a single entity or target."""
        body = self._get_body(entity_id)
        if body is None:
            raise KeyError(f"Unknown entity id: {entity_id}")
        return self._build_state(entity_id, body)

    def get_states(self) -> Dict[str, EntityState]:
        """Return state for all entities and targets."""
        states: Dict[str, EntityState] = {}
        for entity_id in self._all_ids():
            body = self._get_body(entity_id)
            if body is None:
                continue
            states[entity_id] = self._build_state(entity_id, body)
        return states

    def get_events(self) -> List[CollisionEvent]:
        """Return the most recent events list."""
        return list(self._events)

    def get_observation(
        self, entity_id: str, num_rays: int, ray_length: float
    ) -> EntityObservation:
        """Return a ray-based observation for a single entity."""
        directions = self._ray_directions(num_rays)
        return self._build_observation(entity_id, directions, ray_length)

    def get_observations(
        self, entity_ids: Iterable[str], num_rays: int, ray_length: float
    ) -> Dict[str, EntityObservation]:
        """Return ray-based observations for multiple entities."""
        directions = self._ray_directions(num_rays)
        observations: Dict[str, EntityObservation] = {}
        for entity_id in entity_ids:
            observations[entity_id] = self._build_observation(
                entity_id, directions, ray_length
            )
        return observations

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
            segment.friction = self.friction
            self.space.add(segment)
            self._register_shape(f"boundary_{idx}", segment, kind=EntityKind.WALL)

    def _register_shape(
        self, entity_id: str, shape: pymunk.Shape, kind: EntityKind
    ) -> None:
        self._shape_to_id[shape] = entity_id
        self._id_to_kind[entity_id] = kind

    def _apply_linear_damping(self, dt: float) -> None:
        damping = self.linear_damping
        for entity_id in self._all_ids():
            body = self._get_dynamic_body(entity_id)
            if body is None:
                continue
            speed = body.velocity.length
            if speed <= 0.0:
                continue
            new_speed = max(0.0, speed - damping * dt)
            if new_speed == 0.0:
                body.velocity = (0.0, 0.0)
            else:
                body.velocity = body.velocity * (new_speed / speed)

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

    def _build_state(self, entity_id: str, body: pymunk.Body) -> EntityState:
        kind = self._id_to_kind.get(entity_id)
        if kind is None:
            raise KeyError(f"Unknown entity kind for id: {entity_id}")
        return EntityState(
            id=entity_id,
            kind=kind,
            position=(float(body.position.x), float(body.position.y)),
            heading=float(body.angle),
            velocity=(float(body.velocity.x), float(body.velocity.y)),
            angular_velocity=float(body.angular_velocity),
        )

    def _ray_directions(self, num_rays: int) -> List[Tuple[float, float]]:
        if num_rays <= 0:
            raise ValueError("num_rays must be positive")
        step = 2.0 * math.pi / num_rays
        return [(math.cos(step * idx), math.sin(step * idx)) for idx in range(num_rays)]

    def _build_observation(
        self,
        entity_id: str,
        directions: List[Tuple[float, float]],
        ray_length: float,
    ) -> EntityObservation:
        if ray_length <= 0.0:
            raise ValueError("ray_length must be positive")
        body = self._get_body(entity_id)
        if body is None:
            raise KeyError(f"Unknown entity id: {entity_id}")

        origin = body.position
        cos_heading = math.cos(body.angle)
        sin_heading = math.sin(body.angle)

        rays: List[RayObservation] = []
        for dir_x_b, dir_y_b in directions:
            dir_x_w = dir_x_b * cos_heading - dir_y_b * sin_heading
            dir_y_w = dir_x_b * sin_heading + dir_y_b * cos_heading
            end = (origin.x + dir_x_w * ray_length, origin.y + dir_y_w * ray_length)
            hit = self._segment_query_first(
                (origin.x, origin.y), end, ignore_id=entity_id
            )
            if hit is None:
                rays.append(RayObservation(distance=ray_length))
                continue
            hit_id = self._shape_to_id.get(hit.shape, "unknown")
            hit_kind = self._id_to_kind.get(hit_id)
            rays.append(
                RayObservation(
                    distance=float(hit.alpha * ray_length),
                    hit_id=hit_id,
                    hit_kind=hit_kind,
                )
            )

        return EntityObservation(entity_id=entity_id, rays=rays)

    def _segment_query_first(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        ignore_id: str | None = None,
    ) -> pymunk.SegmentQueryInfo | None:
        hits = self.space.segment_query(start, end, 0.0, pymunk.ShapeFilter())
        closest: pymunk.SegmentQueryInfo | None = None
        closest_alpha = float("inf")
        for hit in hits:
            hit_id = self._shape_to_id.get(hit.shape)
            if ignore_id is not None and hit_id == ignore_id:
                continue
            if hit.alpha < closest_alpha:
                closest = hit
                closest_alpha = hit.alpha
        return closest

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
        self._events.append(CollisionEvent(a=id_a, b=id_b, points=points))
        return True
