import pygame
import random

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Points and Lines")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Fixed first point
point1 = [WIDTH//2, HEIGHT//2]  # Center of screen
point2 = [WIDTH//2 + 100, HEIGHT//2]  # Initial position
point3 = [WIDTH//2 + 200, HEIGHT//2]  # Initial position

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    point2[1] -= 1
#    point2[1] += random.randint(-2, 2)

    # Clear screen
    screen.fill(BLACK)

    # Draw lines
    pygame.draw.line(screen, WHITE, point1, point2)
    pygame.draw.line(screen, WHITE, point2, point3)

    # Draw points
    pygame.draw.circle(screen, RED, point1, 5)
    pygame.draw.circle(screen, RED, point2, 5)
    pygame.draw.circle(screen, RED, point3, 5)

    # Update display
    pygame.display.flip()

    # Control frame rate
    clock.tick(60)

pygame.quit()
