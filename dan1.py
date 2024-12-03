#
# Simple 2D 2DOF IK solver
#

import pygame
import math
import random
from time import time

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Moving Points and Lines")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 100, 255)

class Point:
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.speed_x = random.uniform(-2, 2)
        self.speed_y = random.uniform(-2, 2)

    def move(self):
        # Update position
        self.x += self.speed_x
        self.y += self.speed_y

        # Bounce off walls
        if self.x < 0 or self.x > WIDTH:
            self.speed_x *= -1
        if self.y < 0 or self.y > HEIGHT:
            self.speed_y *= -1

    def draw(self, screen):
        pygame.draw.circle(screen, BLUE, (int(self.x), int(self.y)), 5)

# Create points
points = [Point() for _ in range(10)]

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear screen
    screen.fill(BLACK)

    # Move and draw points
    for point in points:
        point.move()
        point.draw(screen)

    # Draw lines between points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            # Calculate distance between points
            dist = math.sqrt((points[i].x - points[j].x)**2 + (points[i].y - points[j].y)**2)

            # Only draw lines between points that are close enough
            if dist < 150:
                # Make lines more transparent with distance
                alpha = int(255 * (1 - dist/150))
                color = (WHITE[0], WHITE[1], WHITE[2], alpha)

                # Create a new surface for the line with alpha channel
                line_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(line_surface, color, 
                               (int(points[i].x), int(points[i].y)),
                               (int(points[j].x), int(points[j].y)), 1)
                screen.blit(line_surface, (0, 0))

    # Update display
    pygame.display.flip()

    # Control frame rate
    clock.tick(60)

pygame.quit()
