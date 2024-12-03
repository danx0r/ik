import math, time, sys
import pygame
import random

R2D = 180/math.pi
SHOW_PLAN = True

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
angle2 = 0

point1 = [WIDTH/2, HEIGHT/2]
target = [int(sys.argv[1]), int(sys.argv[2])]

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
    point3[0] = point2[0] + math.cos((angle1+angle2)/R2D) * length2
    point3[1] = point2[1] - math.sin((angle1+angle2)/R2D) * length2

    return point2, point3

phase = 1
ang1_orig = angle1
ang2_orig = angle2
dlast = 999999

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    point2, point3 = forward(point1, length1, angle1, length2, angle2)
    dist = pdist(point3, target)
    if phase==1:
        # if dist > dlast:
        #     print ("ERROR can not reach target")
        #     break
        dlast = dist
        if dist < 1:
            print ("PHASE 2")
            phase = 2
            nuang1 = random.random() * 360 - 180
            nuang2 = random.random() * 360 - 180
            add1 = (angle1 - nuang1) / 100
            add2 = (angle2 - nuang2) / 100
            angle1 = nuang1
            angle2 = nuang2
            do = 100

        delta = dist/40

        mindist = 999999
        for delta1, delta2 in ((delta, delta), (delta, -delta), (-delta, delta), (-delta, -delta), (0, delta), (0, -delta), (delta, 0), (-delta, 0)):
            p2, p3 = forward(point1, length1, angle1+delta1, length2, angle2+delta2)
            d2 = pdist(p3, target)
            if d2 < mindist:
                mindist = d2
                ang1 = angle1+delta1
                ang2 = angle2+delta2
        angle1 = ang1
        angle2 = ang2

        # p2, p3 = forward(point1, length1, angle1, length2, angle2+delta)
        # d2 = pdist(p3, target)
        # p2, p3 = forward(point1, length1, angle1, length2, angle2-delta)
        # d3 = pdist(p3, target)
        # if d3 < d2:
        #     angle2-=delta
        # else:
        #     angle2+=delta

    if phase==2:
        angle1 += add1
        angle2 += add2
        do -= 1
        if do < 0:
           break

    if phase==2 or SHOW_PLAN:
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
        print ("running", angle1, angle2)
    # input()

time.sleep(5)
pygame.quit()
