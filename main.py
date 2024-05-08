# sourcery skip: use-itertools-product
import torch
import numpy as np
from vector import *
import cv2
from tqdm import tqdm
    
def ray_color(ray: Ray, sphere: Sphere):
    t = sphere.hit(ray)
    if t > 0:
        # print('HERE!')
        p = ray.at(t)
        normal = unit_vector((p - sphere.center))
        ncolor = 0.5 * (normal + 1.0)
        # print(ncolor*255.999)
        return ncolor
    
    t = 0.5 * (ray.direction[1] + 1.0)
    return (1.0 - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0)

def radiance(light_point, hit_point, light_power):
    l = unit_vector(light_point - hit_point)
    # ray = Ray(hit_point, l)
    # hit_s = ComputeIntersection(ray, obj)
    # if abs(torch.sum(hit_point - light_point)) <= EPSILON:
    r = norm(light_point - hit_point)
    L = light_power / r**2
    return L, l
    # else:
    #     return Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0)

EPSILON = 1e-4

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # IMAGE
    aspect_ratio = 16.0 / 9.0
    img_width = 800
    # img_height = 600
    img_height = int(img_width / aspect_ratio)
    
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
    
    image = torch.zeros(img_height, img_width, 3)#.to(device)
    s1 = Sphere(Vec3(0, 0, -1), 0.5)
    
    # front_wall = Box(Vec3(-0.10,-0.10,-0.10), Vec3(5.65,5.65,0.0))
    # left_wall = Box(Vec3(-0.10,-0.1,0.0), Vec3(0.0,5.55,5.55))
    
    # left_wall = Box(Vec3(0.0, 0.0, 1.0), Vec3(1.0, 1.0, 2.0))
    floor = Sphere(Vec3(0.0, -1000 -0.5, -1.0), 1000)
    
    # Point light
    light_position = Vec3(0.7, 0.7, 0.7)
    # light_position = Vec3(2.775,5.55,2.775)
    light_power = Vec3(2.775, 5.55, 2.775)

    for j in tqdm(range(img_height)):
        for i in range(img_width):
            u = i / (img_width - 1)
            v = 1.0 - j / (img_height - 1)
            ray = Ray(origin, lower_left_corner + u*horizontal + v*vertical - origin)
            
            # # Check for intersections with objects in the scene
            # closest_t = float('inf')  # Initialize to positive infinity
            # closest_obj = None
        
            for obj in [s1, floor]:
                # record = HitRecord(0.0, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
                # t = obj.hit(ray)
                hit, record = obj.hit(ray)
                # print(hit)
                if hit:
                    # hit_point = ray.at(record.t)
                    hit_point = record.hit_point
                    # print(light_position, hit_point)
                    # print(torch.sum(hit_point - light_position))
                    if abs(torch.sum(hit_point - light_position)) <= EPSILON:
                        # print(hit_point, light_position)
                        r = record.t
                        pixel_color = light_power / r**2
                    else:
                        # normal = obj.normal_at_point(hit_point)
                        normal = record.normal
                        
                        # Phong Metrial model
                        pixel_color = torch.zeros(3)
                        # v = unit_vector(origin - hit_point)
                        L, l = radiance(light_position, hit_point, light_power)
                        m_dif = Vec3(0.5, 0.5, 0.5)
                        pixel_color += m_dif * L * max(0, dot(normal, l))
                        
                    image[j, i] = torch.tensor([pixel_color[0], pixel_color[1], pixel_color[2]]) * 255.999
    
    cv2.imwrite('output.png', image.numpy()[:, :, ::-1])