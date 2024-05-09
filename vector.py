import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Vec3(x, y, z):
    return torch.tensor([x, y, z])#.to(device)

def norm(v):
    return torch.norm(v)

def norm_squared(v):
    return norm(v)**2

def dot(v1, v2):
    # Inner product
    return torch.dot(v1, v2)

def cross(v1, v2):
    # Cross product
    return torch.linalg.cross(v1, v2)

def unit_vector(v):
    return v / norm(v)

class World:
    def __init__(self):
        self.objects = []
    
    def add_object(self, obj):
        self.objects.append(obj)
        
    def hit(self, ray, t_min=0.0, t_max=float('inf')):
        hit_record = HitRecord(0.0, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
        hit_anything = False
        closest_so_far = t_max
        for obj in self.objects:
            hit, record = obj.hit(ray, t_min, closest_so_far)
            if hit:
                hit_anything = True
                closest_so_far = record.t
                # ver se vai precisar copiar coordenada a coordenada
                hit_record.t = record.t
                hit_record.hit_point = record.hit_point
                hit_record.normal = record.normal
                # hit_record = record
        return hit_anything, hit_record
    
class Ray:
    def __init__(self, origin: torch.float, direction: torch.float):
        self.origin = origin
        self.direction = direction

    def at(self, t:float):
        return self.origin + t*self.direction

    def __str__(self):
        return f'Ray(origin={self.origin}, direction={self.direction})'

class Sphere:
    def __init__(self, center: torch.float, radius: float, color: torch.float=torch.tensor([1.0, 1.0, 1.0]), material: str='diffuse'):
        self.center = center
        self.radius = radius
        self.color = color
        self.material = material
    def old_hit(self, ray):
        a = norm_squared(ray.direction)
        oc = self.center - ray.origin
        halfb = dot(oc, ray.direction)
        c = norm_squared(oc) - self.radius**2
        discriminant = halfb**2 - a*c
        
        # print(discriminant)
        if discriminant < 0:
            return -1.0
        # print('HERE2!')
        return (halfb - torch.sqrt(discriminant)) / (a)
    
    def hit(self, ray, t_min=0.0, t_max=float('inf')):
        a = norm_squared(ray.direction)
        oc = self.center - ray.origin
        halfb = dot(oc, ray.direction)
        c = norm_squared(oc) - self.radius**2
        discriminant = halfb**2 - a*c
        
        if discriminant < 0:
            # return -1.0
            return False, None
        else:
            sqrt = torch.sqrt(discriminant)
            t = (halfb - sqrt) / a
            # print(t)
            # print(t < t_min, t > t_max)
            if t < t_min or t > t_max:
                t = (halfb + sqrt) / a
                # print(t)
                if t < t_min or t > t_max:
                    return False, None
            # print(t)
            record = HitRecord(0.0, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
            record.t = t
            record.hit_point = ray.at(t)
            outward_normal = unit_vector((record.hit_point - self.center))
            front_face = dot(ray.direction, outward_normal) < 0
            record.normal = outward_normal if front_face else -outward_normal
            # record.set_face_normal(ray, outward_normal)
            return True, record
    
    def normal_at_point(self, p):
        return unit_vector(p - self.center)
        
class Box:
    def __init__(self, min_point: torch.Tensor, max_point: torch.Tensor):
        self.min_point = min_point
        self.max_point = max_point
        
    def hit(self, ray):
        t0 = (self.min_point - ray.origin) / ray.direction
        t1 = (self.max_point - ray.origin) / ray.direction
        
        t_near = torch.min(t0, t1)
        t_far = torch.max(t0, t1)
        
        t_min = torch.max(t_near)
        t_max = torch.min(t_far)
        # print(t_min, t_max)

        if t_min <= t_max and t_min >= 0:
            print('HERE!')
            # Ray intersects the box and the exit point is in front of the ray
            return t_min
        else:
            # No intersection or the exit point is behind the ray
            return -1.0
        
        # t = torch.min(t_min, t_max)
        # return -1.0 if t < 0 else t

    def normal_at_point(self, point):
        epsilon = 0.0001  # Small value to handle floating-point precision issues
        if abs(point[0] - self.min_point[0]) < epsilon:
            return Vec3(-1.0, 0.0, 0.0)  # Left face
        elif abs(point[0] - self.max_point[0]) < epsilon:
            return Vec3(1.0, 0.0, 0.0)   # Right face
        elif abs(point[1] - self.min_point[1]) < epsilon:
            return Vec3(0.0, -1.0, 0.0)  # Bottom face
        elif abs(point[1] - self.max_point[1]) < epsilon:
            return Vec3(0.0, 1.0, 0.0)   # Top face
        elif abs(point[2] - self.min_point[2]) < epsilon:
            return Vec3(0.0, 0.0, -1.0)  # Back face
        elif abs(point[2] - self.max_point[2]) < epsilon:
            return Vec3(0.0, 0.0, 1.0)   # Front face

class Plane:
    def __init__(self, point: torch.Tensor, normal: torch.Tensor):
        self.point = point
        self.normal = normal
        
    def hit(self, ray):
        t = dot(self.point - ray.origin, self.normal) / dot(ray.direction, self.normal)
        return -1.0 if t < 0 else t

    def normal_at_point(self, point):
        return self.normal

class HitRecord:
    def __init__(self, t: float, hit_point: torch.Tensor, normal: torch.Tensor):
        self.t = t
        self.hit_point = hit_point
        self.normal = normal

    def __str__(self):
        return f'HitRecord(t={self.t}, p={self.p}, normal={self.normal})'
    
if __name__ == '__main__':
    v = Vec3(1.0, 2.0, 3.0)
    print(v)
    
    v_norm = norm(v)
    print(v_norm)
    
    v_norm2 = norm_squared(v)
    print(v_norm2)
    
    v2 = Vec3(1.0, 0.0, 0.0)
    v1v2_dot = dot(v, v2)
    print(v1v2_dot)
    
    v3 = Vec3(0.0, 1.0, 0.0)
    v2v3_cross = cross(v2, v3)
    print(v2v3_cross)
    
    v_normed = unit_vector(v)
    print(v_normed)
    print(norm(v_normed))
    
    ray = Ray(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0))
    print(ray)
    