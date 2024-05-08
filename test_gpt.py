import torch
from tqdm import tqdm
import cv2

class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return Vec3(self.x * other, self.y * other, self.z * other)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length_squared(self):
        return self.x**2 + self.y**2 + self.z**2

    def normalize(self):
        length = self.length_squared() ** 0.5
        return Vec3(self.x / length, self.y / length, self.z / length)

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

class Box:
    def __init__(self, min_point, max_point):
        self.min_point = min_point
        self.max_point = max_point
        
    def hit(self, ray):
        t_near = (self.min_point - ray.origin) / ray.direction
        t_far = (self.max_point - ray.origin) / ray.direction

        t_min = torch.min(t_near, t_far)
        t_max = torch.max(t_near, t_far)

        t_enter = torch.max(t_min)
        t_exit = torch.min(t_max)

        if t_enter <= t_exit and t_exit >= 0:
            return t_enter
        else:
            return -1.0

class PointLight:
    def __init__(self, position, power):
        self.position = position
        self.power = power

class Material:
    def __init__(self, color):
        self.color = color

def trace_ray(ray, scene):
    hit_distance = float('inf')
    hit_point = None
    hit_material = None

    for obj in scene:
        t = obj.hit(ray)
        if 0 < t < hit_distance:
            hit_distance = t
            hit_point = ray.origin + ray.direction * t
            hit_material = obj.material

    return hit_point, hit_material

def shade(hit_point, hit_material, light):
    if hit_point is None or hit_material is None:
        return Vec3(0, 0, 0)

    # Diffuse shading
    light_direction = (light.position - hit_point).normalize()
    diffuse_factor = max(0, light_direction.dot(hit_normal))
    diffuse_color = hit_material.color * diffuse_factor * light.power / (light.position - hit_point).length_squared()

    return diffuse_color

# Define scene objects
box = Box(Vec3(-1, -1, -1), Vec3(1, 1, 1))
box.material = Material(Vec3(0.8, 0.2, 0.2))  # Red material
light = PointLight(Vec3(0.7, 0.7, 0.7), Vec3(2.775, 5.55, 2.775))

# Define camera parameters
origin = Vec3(0, 0, 0)
lower_left_corner = Vec3(-2.0, -1.0, -1.0)
horizontal = Vec3(4.0, 0.0, 0.0)
vertical = Vec3(0.0, 2.0, 0.0)

# Define image dimensions
img_width = 800
img_height = 600

# Create an empty image
image = torch.zeros(img_height, img_width, 3)

# Render loop
for j in tqdm(range(img_height)):
    for i in range(img_width):
        # Calculate ray direction for this pixel
        u = i / (img_width - 1)
        v = 1.0 - j / (img_height - 1)
        ray = Ray(origin, lower_left_corner + Vec3(u, u, u) * horizontal + Vec3(v, v, v) * vertical - origin)

        # Trace the ray and find the intersection point
        hit_point, hit_material = trace_ray(ray, [box])

        # Shade the intersection point
        pixel_color = shade(hit_point, hit_material, light)

        # Set pixel color in the image
        image[j, i] = torch.tensor([pixel_color.x, pixel_color.y, pixel_color.z])

# Display or save the rendered image
cv2.imwrite('output_gpt.png', image.numpy()[:, :, ::-1])
