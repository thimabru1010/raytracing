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

class Ray:
    def __init__(self, origin: torch.float, direction: torch.float):
        self.origin = origin
        self.direction = direction

    def at(self, t:float):
        return self.origin + t*self.direction

    def __str__(self):
        return f'Ray(origin={self.origin}, direction={self.direction})'

class Sphere:
    def __init__(self, center: torch.float, radius: float):
        self.center = center
        self.radius = radius
        
    def hit(self, ray):
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
        
class Box:
    def __init__(self, min: torch.Tensor, max: torch.Tensor):
        self.min = min
        self.max = max
        
        
    def hit(self, ray):
        tmin = (self.min[0] - ray.origin[0]) / ray.direction[0]
        tmax = (self.max[0] - ray.origin[0]) / ray.direction[0]
        if tmin > tmax:
            tmin, tmax = tmax, tmin
        
        tymin = (self.min[1] - ray.origin[1]) / ray.direction[1]
        tymax = (self.max[1] - ray.origin[1]) / ray.direction[1]
        if tymin > tymax:
            tymin, tymax = tymax, tymin
        
        if (tmin > tymax) or (tymin > tmax):
            return False
        
        if tymin > tmin:
            tmin = tymin
        if tymax < tmax:
            tmax = tymax
        
        tzmin = (self.min[2] - ray.origin[2]) / ray.direction[2]
        tzmax = (self.max[2] - ray.origin[2]) / ray.direction[2]
        if tzmin > tzmax:
            tzmin, tzmax = tzmax, tzmin
        
        if (tmin > tzmax) or (tzmin > tmax):
            return False
        
        if tzmin > tmin:
            tmin = tzmin
        if tzmax < tmax:
            tmax = tzmax
        
        return True
    
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
    