import math
import pygame
import random

R2D = 180/math.pi

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH = 1000
HEIGHT = 1000
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Points and Lines")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

length1 = 200
angle1 = 0
length2 = 150
angle2 = 60

point1 = [WIDTH/2, HEIGHT/2]
target = [600, 350]

# Main game loop
running = True
clock = pygame.time.Clock()

def pdist(p1, p2):
    x, y = p1
    xx, yy = p2
    return((x-xx)**2 + (y-yy)**2) ** 0.5

def forward(point1, length1, angle1, length2, angle2):
    point2 = [0, 0]
    point2[0] = point1[0] + math.cos(angle1/R2D) * length1
    point2[1] = point1[1] - math.sin(angle1/R2D) * length1

    point3 = [0, 0]
    point3[0] = point2[0] + math.cos(angle2/R2D) * length2
    point3[1] = point2[1] - math.sin(angle2/R2D) * length2

    return point2, point3

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    point2, point3 = forward(point1, length1, angle1, length2, angle2)
    dist = pdist(point3, target)
    delta = dist/200

    p2, p3 = forward(point1, length1, angle1+delta, length2, angle2)
    d2 = pdist(p3, target)
    p2, p3 = forward(point1, length1, angle1-delta, length2, angle2)
    d3 = pdist(p3, target)
    if d3 < d2:
        angle1-=delta
    else:
        angle1+=delta

    p2, p3 = forward(point1, length1, angle1, length2, angle2+delta)
    d2 = pdist(p3, target)
    p2, p3 = forward(point1, length1, angle1, length2, angle2-delta)
    d3 = pdist(p3, target)
    if d3 < d2:
        angle2-=delta
    else:
        angle2+=delta

    # Clear screen
    screen.fill(BLACK)

    # Draw lines
    pygame.draw.line(screen, WHITE, point1, point2)
    pygame.draw.line(screen, WHITE, point2, point3)

    # Draw points
    pygame.draw.circle(screen, GREEN, target, 5)
    pygame.draw.circle(screen, RED, point1, 5)
    pygame.draw.circle(screen, RED, point2, 5)
    pygame.draw.circle(screen, RED, point3, 5)

    # Update display
    pygame.display.flip()

    # Control frame rate
    clock.tick(60)
    print ("running", dist, point1, point2, point3)

pygame.quit()
