import math


def calculate_position_similarity(pos1, pos2):
    if not pos1 or not pos2 or len(pos1) != 2 or len(pos2) != 2:
        return 0.0
    distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    return math.exp(-distance / 2.0)


def normalize_coordinates(coordinates, grid_size=(10, 10)):
    if not coordinates:
        return []
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    x_range = max_x - min_x if max_x != min_x else 1
    y_range = max_y - min_y if max_y != min_y else 1
    x_scale = (grid_size[0] - 1) / x_range
    y_scale = (grid_size[1] - 1) / y_range
    return [[(coord[0] - min_x) * x_scale, (coord[1] - min_y) * y_scale] for coord in coordinates]


def get_relative_position(pos1, pos2):
    if not pos1 or not pos2 or len(pos1) != 2 or len(pos2) != 2:
        return "unknown"
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    if abs(dx) < 0.1 and abs(dy) < 0.1:
        return "same"
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"
