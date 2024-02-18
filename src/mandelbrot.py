# Mandelbrot fractal
from PIL import Image

# Drawing area
xa = -2.0
xb = 1.0
ya = -1.5
yb = 1.5

# Max iterations allowed
max_it = 255

# Image size
img_x = 1024
img_y = 1024
image = Image.new("RGB", (img_x, img_y))

for y in range(img_y):
    zy = y * (yb - ya) / (img_y - 1) + ya
    for x in range(img_x):
        zx = x * (xb - xa) / (img_x - 1) + xa
        z = zx + zy * 1j
        c = z
        for i in range(max_it):
            if abs(z) > 2.0:
                break
            z = z * z + c
        image.putpixel((x, y), (i % 4 * 64, i % 8 * 32, i % 16 * 16))

image.show()
