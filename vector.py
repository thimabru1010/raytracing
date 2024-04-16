import torch
    
def Vec3(x, y, z):
    return torch.tensor([x, y, z])

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
    