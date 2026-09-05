import pygame, random, sys

pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# Load your PNG map directly from the path
map_img = pygame.image.load(r"C:\Users\Admin\kyber (7)\GAME.png").convert()

# Player setup
player = pygame.Rect(100, 100, 20, 20)
speed = 4

# Weapons + ammo
current_weapon = "pistol"
ammo = {"pistol": 50}

# Demon setup
demons = []
def spawn_demon():
    x = random.randint(0, 780)
    y = random.randint(0, 580)
    demons.append(pygame.Rect(x, y, 20, 20))

# Bullets
bullets = []

def shoot():
    if ammo[current_weapon] > 0:
        ammo[current_weapon] -= 1
        bullets.append(pygame.Rect(player.x+10, player.y+10, 5, 5))

def win_game():
    print("You wake up... Reality restored.")
    pygame.quit()
    sys.exit()

# Game loop
running = True
while running:
    screen.blit(map_img, (0,0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]: player.y -= speed
    if keys[pygame.K_s]: player.y += speed
    if keys[pygame.K_a]: player.x -= speed
    if keys[pygame.K_d]: player.x += speed
    if keys[pygame.K_SPACE]: shoot()

    # Move bullets
    for b in bullets[:]:
        b.x += 10
        pygame.draw.rect(screen, (255,255,0), b)
        for d in demons[:]:
            if b.colliderect(d):
                demons.remove(d)
                bullets.remove(b)

    # Spawn demons randomly
    if random.randint(0,100) < 2:
        spawn_demon()

    # Draw demons
    for d in demons:
        pygame.draw.rect(screen, (255,0,0), d)

    # Draw player
    pygame.draw.rect(screen, (0,255,0), player)

    # Zone detection from PNG
    color = map_img.get_at((player.x, player.y))
    if color == (255,0,0,255):  # Red island (Volcano)
        win_game()

    pygame.display.flip()
    clock.tick(30)
