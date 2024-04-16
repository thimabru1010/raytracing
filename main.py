# sourcery skip: use-itertools-product
import torch
import numpy as np
from vector import *
import cv2

# if __name__ == '__main__':
#     from vector import Vec3, Ray, unit_vector, dot, cross, norm_squared
#     from camera import Camera
#     from hittable import HittableList, Sphere
#     from material import Lambertian, Metal, Dielectric
#     from renderer import Renderer
#     from utils import random_scene

#     aspect_ratio = 16.0 / 9.0
#     img_width = 800
#     img_height = int(img_width / aspect_ratio)
#     samples_per_pixel = 100
#     max_depth = 50

#     world = random_scene()

#     lookfrom = Vec3(13, 2, 3)
#     lookat = Vec3(0, 0, 0)
#     vup = Vec3(0, 1, 0)
#     dist_to_focus = 10.0
#     aperture = 0.1

#     cam = Camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus)

#     renderer = Renderer(img_width, img_height, samples_per_pixel, max_depth, world, cam)
#     renderer.render()

#     renderer.save_image('output.png')
# def ray_color(ray: Ray, world: HittableList, depth: int):
    
def ray_color(ray: Ray):
    # if depth <= 0:
    #     return Vec3(0.0, 0.0, 0.0)
    
    # rec = world.hit(ray, 0.001, float('inf'))
    # if rec:
    #     target = rec.p + rec.normal + random_in_unit_sphere()
    #     return 0.5 * ray_color(Ray(rec.p, target - rec.p), world, depth - 1)
    
    unit_direction = unit_vector(ray.direction)
    t = 0.5 * (unit_direction[2] + 1.0)
    return (1.0 - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0)

if __name__ == '__main__':
    # IMAGE
    aspect_ratio = 16.0 / 9.0
    img_width = 800
    img_height = 600
    
    #CAMERA
    viewport_height = 2.0
    viewport_width = aspect_ratio * viewport_height
    horizontal = Vec3(viewport_width, 0.0, 0.0)
    vertical = Vec3(0.0, viewport_height, 0.0)
    origin = Vec3(0.0, 0.0, 0.0)
    focal_length = 1.0
    lower_left_corner = origin - horizontal/2 - vertical/2 - Vec3(0.0, 0.0, focal_length)
    # Vec3(-viewport_width/2, -viewport_height/2, -1.0)
    # focal_length = 1.0

    image = torch.zeros(img_height, img_width, 3)
    
    for i in range(img_height):
        for j in range(img_width):
            #TODO: entender u e v
            u = (i -1) / (img_width - 1)
            v = 1.0 - (j -1) / (img_height - 1)
            ray = Ray(origin, lower_left_corner + u*horizontal + v*vertical - origin)
            # color = ray_color(ray)
            image[i, j] = ray_color(ray) * 255
    
    cv2.imwrite('output.png', image.numpy().astype(np.uint8)[:,:,::-1])