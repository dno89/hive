"""Pygame tester for the top-down pymunk simulator."""

from __future__ import annotations

import random
from typing import List

import pygame
import pymunk
from pymunk.pygame_util import DrawOptions

from hive.sim.simulator import MotionAction, Simulator, MotionConfig


WORLD_SIZE_M = (100.0, 200.0)
PIXELS_PER_METER = 20.0
WINDOW_SIZE = (1000, 600)
BG_COLOR = (18, 20, 24)
TEXT_COLOR = (240, 240, 240)
GRID_COLOR = (70, 75, 82)

SIM_ACCELERATION_VALUE = 6.0
SIM_ANG_SPEED_VALUE = 2.5
SIM_MAX_SPEED = 20.0
ENTITY_RADIUS_M = 0.5
TARGET_RADIUS_M = 0.5
SPAWN_MARGIN_M = 5.0
GRID_SPACING_M = 5.0
GRID_LINE_WIDTH = 1


def _format_events(events: List[dict]) -> List[str]:
    lines = []
    for event in events:
        if event.get("type") == "collision":
            lines.append(f"collision: {event.get('a')} <-> {event.get('b')}")
        else:
            lines.append(str(event))
    return lines


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def _camera_transform(
    target_pos: tuple[float, float],
    world_size: tuple[float, float],
    window_size: tuple[int, int],
    pixels_per_meter: float,
) -> pymunk.Transform:
    view_w_m = window_size[0] / pixels_per_meter
    view_h_m = window_size[1] / pixels_per_meter
    cam_x = _clamp(target_pos[0], view_w_m / 2, world_size[0] - view_w_m / 2)
    cam_y = _clamp(target_pos[1], view_h_m / 2, world_size[1] - view_h_m / 2)
    tx = window_size[0] / 2 - pixels_per_meter * cam_x
    ty = window_size[1] / 2 - pixels_per_meter * cam_y
    return pymunk.Transform(pixels_per_meter, 0.0, 0.0, pixels_per_meter, tx, ty)


def _draw_grid(
    surface: pygame.Surface,
    camera_transform: pymunk.Transform,
    world_size: tuple[float, float],
    spacing_m: float,
    line_width: int,
) -> None:
    if spacing_m <= 0.0:
        return
    step = spacing_m
    width_m, height_m = world_size
    x = 0.0
    while x <= width_m:
        start = camera_transform @ (x, 0.0)
        end = camera_transform @ (x, height_m)
        pygame.draw.line(surface, GRID_COLOR, start, end, line_width)
        x += step
    y = 0.0
    while y <= height_m:
        start = camera_transform @ (0.0, y)
        end = camera_transform @ (width_m, y)
        pygame.draw.line(surface, GRID_COLOR, start, end, line_width)
        y += step


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Pymunk Top-Down Sim Tester")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Courier", 16)

    sim = Simulator(dt=1.0 / 60.0, world_size=WORLD_SIZE_M, linear_damping=2.0)
    sim.add_entity(
        "agent",
        position=(WORLD_SIZE_M[0] / 2, WORLD_SIZE_M[1] / 2),
        radius=ENTITY_RADIUS_M,
        motion_config=MotionConfig(max_speed=SIM_MAX_SPEED),
    )

    target_pos = (
        random.uniform(SPAWN_MARGIN_M, WORLD_SIZE_M[0] - SPAWN_MARGIN_M),
        random.uniform(SPAWN_MARGIN_M, WORLD_SIZE_M[1] - SPAWN_MARGIN_M),
    )
    sim.add_static_target("target", position=target_pos, radius=TARGET_RADIUS_M)

    draw_options = DrawOptions(screen)

    running = True
    last_events: List[str] = []
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        longitudinal_accel = 0.0
        lateral_accel = 0.0
        angular_speed = 0.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            longitudinal_accel += SIM_ACCELERATION_VALUE
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            longitudinal_accel -= SIM_ACCELERATION_VALUE

        if keys[pygame.K_q]:
            angular_speed -= SIM_ANG_SPEED_VALUE
        if keys[pygame.K_e]:
            angular_speed += SIM_ANG_SPEED_VALUE

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            lateral_accel -= SIM_ACCELERATION_VALUE
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            lateral_accel += SIM_ACCELERATION_VALUE

        events = sim.step(
            {
                "agent": MotionAction(
                    longitudinal_accel=longitudinal_accel,
                    lateral_accel=lateral_accel,
                    angular_velocity=angular_speed,
                )
            }
        )
        last_events = _format_events(events) if events else last_events
        agent_state = sim.get_entity_state("agent")
        draw_options.transform = _camera_transform(
            agent_state["position"],
            WORLD_SIZE_M,
            WINDOW_SIZE,
            PIXELS_PER_METER,
        )

        screen.fill(BG_COLOR)
        _draw_grid(
            screen,
            draw_options.transform,
            WORLD_SIZE_M,
            GRID_SPACING_M,
            GRID_LINE_WIDTH,
        )
        sim.debug_draw(draw_options)

        info_lines = [
            "Controls: WASD/arrows + Q/E strafe",
            f"Events: {len(last_events)}",
        ] + last_events[:6]
        for idx, line in enumerate(info_lines):
            text_surface = font.render(line, True, TEXT_COLOR)
            screen.blit(text_surface, (10, 10 + idx * 18))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
