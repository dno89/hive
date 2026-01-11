"""Pygame tester for the top-down pymunk simulator."""

from __future__ import annotations

import random
from typing import List

import pygame
import pymunk
from pymunk.pygame_util import DrawOptions

from hive.sim.simulator import MotionAction, Simulator


WINDOW_SIZE = (800, 600)
BG_COLOR = (18, 20, 24)
TEXT_COLOR = (240, 240, 240)


def _format_events(events: List[dict]) -> List[str]:
    lines = []
    for event in events:
        if event.get("type") == "collision":
            lines.append(f"collision: {event.get('a')} <-> {event.get('b')}")
        else:
            lines.append(str(event))
    return lines


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Pymunk Top-Down Sim Tester")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Courier", 16)

    sim = Simulator(dt=1.0 / 60.0)
    sim.add_entity(
        "agent", position=(WINDOW_SIZE[0] / 2, WINDOW_SIZE[1] / 2), radius=18.0
    )

    target_pos = (
        random.uniform(100, WINDOW_SIZE[0] - 100),
        random.uniform(100, WINDOW_SIZE[1] - 100),
    )
    sim.add_static_target("target", position=target_pos, radius=14.0)

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
            longitudinal_accel += 200.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            longitudinal_accel -= 200.0
        if keys[pygame.K_q]:
            lateral_accel += 200.0
        if keys[pygame.K_e]:
            lateral_accel -= 200.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            angular_speed += 3.5
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            angular_speed -= 3.5

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

        screen.fill(BG_COLOR)
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
