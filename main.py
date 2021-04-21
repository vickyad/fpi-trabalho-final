import math

import numpy as np
import cv2 as cv

INT_MAX = 2147483647


class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def grab_cut(img_target, selected_area):
    z_mask = np.zeros(img_target.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask, bgModel, fgModel = cv.grabCut(img_target, z_mask, selected_area, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    output_mask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1)
    output_mask = (output_mask * 255).astype("uint8")
    output = cv.bitwise_and(img_target, img_target, mask=output_mask)

    # cv.imshow("Input", img_target)
    # cv.imshow("GrabCut Mask", output_mask)
    # cv.imshow("GrabCut Output", output)
    # cv.waitKey(0)
    return output_mask, output


def poisson_editing(img_src, img_dst):
    mask = 255 * np.ones(img_dst.shape, img_dst.dtype)

    width, height, channels = img_src.shape
    center = (round(height / 2), round(width / 2))

    mixed_clone = cv.seamlessClone(img_dst, img_src, mask, center, cv.MIXED_CLONE)

    cv.imwrite("images/opencv-mixed-clone-example.jpg", mixed_clone)


def optimized_boundary(img_s, img_t, obj_mask, rect):
    limit_pixel_found = False
    limit_line_init = (0, 0)

    # Encontra o primeiro ponto válido da máscara (linha a linha)
    for y in range(rect[1], rect[3]):
        for x, pixel in enumerate(obj_mask[y]):
            if pixel == 255:
                limit_line_init = (x, y)
                limit_pixel_found = True
                break
        if limit_pixel_found:
            break

    # Constante K
    k = calculate_k(img_s, img_t, rect)

    # Matriz de pesos para o Dijkstra
    weights = np.full((rect[3], rect[2]), -1, np.int32)

    for x in range(0, rect[2]):
        for y in range(0, rect[3]):
            sx = rect[0] + x
            sy = rect[1] + y
            mask_pixel = obj_mask[sy][sx]
            if mask_pixel != 255:
                pixel_s = img_s[sy][sx]
                pixel_t = img_t[sy][sx]
                weights[y][x] = pow(pixel_color_distance(pixel_t, pixel_s) - k, 2)

    # Pixels de inicio e fim do Dijkstra
    beginnings = []
    destinations = []
    rect_x = limit_line_init[0] - rect[0]
    for y in range(rect[1], limit_line_init[1]):
        rect_y = y - rect[0]
        beginnings.append([rect_x, rect_y])
        destinations.append([rect_x + 1, rect_y])

    # Calcula o menor caminho a partir de cada pixel inicial
    paths_list = np.full((len(beginnings), rect[3], rect[2]), INT_MAX, np.int32)
    parents_list = np.full((len(beginnings), rect[3], rect[2]), Coordinate(0, 0), Coordinate)
    for i in range(0, len(beginnings)):
        init = beginnings[i]
        min_path(weights, paths_list[i], parents_list[i], beginnings, destinations, init[0], init[1])

    # Encontra o menor caminho entre um pixel de destino e um pixel inicial nos resultados calculados
    min_weight = [0, 0, 0, INT_MAX]
    for destination in destinations:
        for index, paths in enumerate(paths_list):
            [x, y] = destination
            weight = paths[y][x]
            if weight < min_weight[3]:
                min_weight = [x, y, index, weight]

    path = paths_list[min_weight[2]]
    parent_list = parents_list[min_weight[2]]
    selected_boundary = np.full((rect[3], rect[2]), 0, np.uint8)
    current_x = min_weight[0]
    current_y = min_weight[1]
    while path[current_y][current_x] != 0:
        selected_boundary[current_y][current_x] = 255
        next_node = parent_list[current_y][current_x]
        current_y = next_node.y
        current_x = next_node.x

    selected_boundary[current_y][current_x] = 255

    cv.imshow("Result", selected_boundary)
    cv.waitKey(0)

    initial_inside_pixel = [0, 0]
    found = False

    for y, row in enumerate(selected_boundary):
        for x, pixel in enumerate(row):
            if pixel == 0 and x - 1 > 0 and y > 0:
                last_pixel = selected_boundary[y][x - 1]
                up_pixel = selected_boundary[y - 1][x]
                if last_pixel == 255:
                    if up_pixel == 255:
                        initial_inside_pixel = [x, y]
                        found = True
                        break
        if found:
            break

    fill(selected_boundary, initial_inside_pixel)

    cv.imshow("Result 2", selected_boundary)
    cv.waitKey(0)

    print("oi")


def fill(selected_boundary, initial_inside_pixel):
    queue = [initial_inside_pixel]
    height = len(selected_boundary)
    width = len(selected_boundary[0])

    while len(queue) > 0:
        [x, y] =   .pop(0)

        if selected_boundary[y][x] == 0 and width > x > 0 and height > y > 0:
            selected_boundary[y][x] = 255
            queue.append([x - 1, y])
            queue.append([x + 1, y])
            queue.append([x, y - 1])
            queue.append([x, y + 1])


def min_path(weights, paths, parent_list, beginnings, destinations, init_x, init_y):
    visited = np.full((weights.shape[0], weights.shape[1]), False, dtype=bool)
    paths[init_y, init_x] = 0
    visited[init_y, init_x] = True
    queue = []
    append_neighborhood(init_x, init_y, queue, weights, visited, beginnings, destinations)
    while len(queue) > 0:
        [x, y] = queue.pop(0)
        node_weight = weights[y][x]
        neighborhood = get_neighborhood(x, y, weights, visited, beginnings, destinations, True)
        paths[y][x] = node_weight + calc_min_weight(neighborhood, paths, parent_list, x, y)
        append_neighborhood(x, y, queue, weights, visited, beginnings, destinations)


def append_neighborhood(x, y, queue, weights, visited, beginnings, destinations):
    queue.extend(get_neighborhood(x, y, weights, visited, beginnings, destinations))


def get_neighborhood(x, y, weights, visited, beginnings, destinations, visited_neighborhood=False):
    neighborhood = []
    its_initial_coordinate = coordinate_belongs_to_list(x, y, beginnings)
    its_destiny_coordinate = coordinate_belongs_to_list(x, y, destinations)

    def validate_direction(target_x, target_y):
        column_size = len(weights)
        row_size = len(weights[0])
        if not is_safe(target_x, target_y, column_size, row_size) or weights[target_y][target_x] == -1:
            return [False, 0, 0]

        if its_initial_coordinate and coordinate_belongs_to_list(target_x, target_y, destinations):
            return [False, 0, 0]

        if its_destiny_coordinate and coordinate_belongs_to_list(target_x, target_y, beginnings):
            return [False, 0, 0]

        valid_direction = visited[target_y][target_x] if visited_neighborhood else not visited[target_y][target_x]
        return [valid_direction, target_x, target_y]

    top_y = y - 1
    bottom_y = y + 1
    left_x = x - 1
    right_x = x + 1

    directions = [
        validate_direction(left_x, y),
        validate_direction(right_x, y),
        validate_direction(x, top_y),
        validate_direction(x, bottom_y),
        validate_direction(left_x, top_y),
        validate_direction(left_x, bottom_y),
        validate_direction(right_x, top_y),
        validate_direction(right_x, bottom_y)
    ]

    for direction in directions:
        is_valid = direction[0]
        if is_valid:
            x = direction[1]
            y = direction[2]
            visited[y][x] = True
            neighborhood.append([x, y])

    return neighborhood


def coordinate_belongs_to_list(target_x, target_y, coordinates_list):
    for i in coordinates_list:
        [x, y] = i
        if target_x == x and target_y == y:
            return True
    return False


def is_safe(x, y, column_size, row_size):
    return row_size > x >= 0 and column_size > y >= 0


def calc_min_weight(paths_coordinates, paths, parent_list, x, y):
    min_weight = INT_MAX
    valid_coordinates = []
    for path_coordinates in paths_coordinates:
        if len(paths) > path_coordinates[1] >= 0 and len(paths[0]) > path_coordinates[0] >= 0:
            valid_coordinates.append(path_coordinates)
    for coordinate in valid_coordinates:
        [dest_x, dest_y] = coordinate
        weight = paths[dest_y][dest_x]
        if weight < min_weight:
            min_weight = weight
            parent_list[y][x] = Coordinate(dest_x, dest_y)
    return min_weight


def calculate_k(img_s, img_t, rect):
    rect_boundary_distance = 0
    for x in range(rect[1], rect[3]):
        y = rect[0]
        distance_left = pixel_color_distance(img_t[x, y], img_s[x, y])
        y = rect[0] + rect[2]
        distance_right = pixel_color_distance(img_t[x, y], img_s[x, y])
        rect_boundary_distance += distance_left + distance_right

    for y in range(rect[0], rect[2]):
        x = rect[1]
        distance_top = pixel_color_distance(img_t[x, y], img_s[x, y])
        x = rect[1] + rect[3]
        distance_bottom = pixel_color_distance(img_t[x, y], img_s[x, y])
        rect_boundary_distance += distance_top + distance_bottom

    return (1 / (2 * rect[2] + 2 * rect[3])) * rect_boundary_distance


def pixel_color_distance(pixel_one, pixel_two):
    return math.sqrt(
        pow(int(pixel_one[0]) - int(pixel_two[0]), 2) +
        pow(int(pixel_one[1]) - int(pixel_two[1]), 2) +
        pow(int(pixel_one[2]) - int(pixel_two[2]), 2))


if __name__ == '__main__':
    rect = (100, 100, 625, 235)
    img_s = cv.imread('./images/airplane2.jpg')
    img_t = cv.imread('./images/sky.jpg')
    obj_mask, obj = grab_cut(img_s, rect)
    optimized_boundary(img_s, img_t, obj_mask, rect)
    cv.imwrite("./images/result_mask.jpg", obj_mask)
    cv.imshow('obj', obj_mask)
    cv.imshow('image', img_s)
    cv.waitKey(0)
    cv.destroyAllWindows()
