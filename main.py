import math

import numpy as np
import cv2 as cv

INT_MAX = 2147483647


class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def grab_cut(img_target, selected_area, source_name):
    z_mask = np.zeros(img_target.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask, bgModel, fgModel = cv.grabCut(img_target, z_mask, selected_area, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    output_mask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1)
    output_mask = (output_mask * 255).astype("uint8")
    output = cv.bitwise_and(img_target, img_target, mask=output_mask)

    #cv.imwrite("results/" + source_name + "/object_mask.jpg", output_mask)
    cv.imshow("Object Mask", output_mask)
    cv.waitKey(0)

    return output_mask, output


def poisson_editing(img_src, img_dst, mask):
    width, height, channels = img_src.shape
    center = (round(height / 2), round(width / 2))

    mixed_clone = cv.seamlessClone(img_src, img_dst, mask, center, cv.MIXED_CLONE)

    return mixed_clone


def optimized_boundary(img_s, img_t, obj_mask, rect, source_name):
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

    # Encontra o primeiro ponto válido da máscara (linha a linha)
    limit_pixel_found = False
    limit_line_init = (0, 0)
    for y in range(rect[1], rect[1] + rect[3]):
        for x in range(rect[0], rect[0] + rect[2]):
            pixel = obj_mask[y][x]
            if pixel == 255:
                limit_line_init = (x, y)
                limit_pixel_found = True
                break
        if limit_pixel_found:
            break

    # Linha divisória para o calculo do menor caminho
    beginnings = []
    rect_x = limit_line_init[0] - rect[0]
    for y in range(rect[1], limit_line_init[1]):
        rect_y = y - rect[1]
        beginnings.append([rect_x, rect_y])

    limit_line = np.full((rect[3], rect[2]), 0, np.uint8)
    for y in range(rect[1], rect[1] + rect[3]):
        for x in range(rect[0], rect[0] + rect[2]):
            limit_line[y - rect[1]][x - rect[0]] = obj_mask[y][x]

    for beginning in beginnings:
        (bx, by) = beginning
        limit_line[by][bx] = 255

    #cv.imwrite("results/" + source_name + "/limit_line.jpg", limit_line)
    cv.imshow("Bounding Line", limit_line)
    cv.waitKey(0)

    # Calcula o menor caminho a partir de cada pixel inicial
    paths_list = np.full((len(beginnings), rect[3], rect[2]), INT_MAX, np.int64)
    parents_list = np.full((len(beginnings), rect[3], rect[2]), Coordinate(0, 0), Coordinate)
    for i in range(0, len(beginnings)):
        init = beginnings[i]
        min_path(weights, paths_list[i], parents_list[i], beginnings, init[0], init[1])

    # Encontra o menor caminho entre um pixel de destino e um pixel inicial nos resultados calculados
    min_weight = [0, 0, 0, INT_MAX]
    for bi, beginning in enumerate(beginnings):
        for ed in range(-1, 2):
            [x, y] = beginning
            end_x = x + 1
            end_y = y + ed
            if 0 <= end_y < len(paths_list[bi]):
                weight = paths_list[bi][end_y][end_x]
                if weight < min_weight[3]:
                    min_weight = [end_x, end_y, bi, weight]

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
    initial_inside_pixel = [0, 0]
    found = False

    #cv.imwrite("results/" + source_name + "/optimized_boundary.jpg", selected_boundary)
    cv.imshow("Optimized Boundary", selected_boundary)
    cv.waitKey(0)

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

    optimized_boundary_mask = np.full((obj_mask.shape[0], obj_mask.shape[1]), 0, np.uint8)

    for y, row in enumerate(selected_boundary):
        for x, pixel in enumerate(row):
            original_y = y + rect[1]
            original_x = x + rect[0]
            optimized_boundary_mask[original_y][original_x] = pixel

    #cv.imwrite("results/" + source_name + "/optimized_boundary_mask.jpg", optimized_boundary_mask)
    cv.imshow("Optimized Boundary Mask", optimized_boundary_mask)
    cv.waitKey(0)

    return optimized_boundary_mask


def fill(selected_boundary, initial_inside_pixel):
    queue = [initial_inside_pixel]
    height = len(selected_boundary)
    width = len(selected_boundary[0])

    while len(queue) > 0:
        [x, y] = queue.pop(0)

        if width > x > 0 and height > y > 0 and selected_boundary[y][x] == 0:
            selected_boundary[y][x] = 255
            queue.append([x - 1, y])
            queue.append([x + 1, y])
            queue.append([x, y - 1])
            queue.append([x, y + 1])


def min_path(weights, paths, parent_list, beginnings, init_x, init_y):
    visited = np.full((weights.shape[0], weights.shape[1]), False, dtype=bool)
    calculated = np.full((weights.shape[0], weights.shape[1]), False, dtype=bool)
    paths[init_y, init_x] = 0
    visited[init_y, init_x] = True
    calculated[init_y, init_x] = True
    queue = []
    append_neighborhood(init_x, init_y, queue, weights, visited, beginnings)
    while len(queue) > 0:
        [x, y] = queue.pop(0)
        node_weight = weights[y][x]
        neighborhood = get_neighborhood(x, y, weights, visited, beginnings, True, calculated)
        paths[y][x] = node_weight + calc_min_weight(neighborhood, paths, parent_list, x, y)
        calculated[y][x] = True
        append_neighborhood(x, y, queue, weights, visited, beginnings)


def append_neighborhood(x, y, queue, weights, visited, beginnings):
    queue.extend(get_neighborhood(x, y, weights, visited, beginnings))


def get_neighborhood(x, y, weights, visited, beginnings, calculated_neighborhood=False, calculated=None):
    if calculated is None:
        calculated = []
    neighborhood = []
    its_initial_coordinate = coordinate_belongs_to_list(x, y, beginnings)
    its_destiny_coordinate = coordinate_belongs_to_list(x - 1, y, beginnings)

    def validate_direction(target_x, target_y):
        column_size = len(weights)
        row_size = len(weights[0])
        if not is_safe(target_x, target_y, column_size, row_size) or weights[target_y][target_x] == -1:
            return [False, 0, 0]

        if its_initial_coordinate and coordinate_belongs_to_list(target_x - 1, target_y, beginnings):
            return [False, 0, 0]

        if its_destiny_coordinate and coordinate_belongs_to_list(target_x, target_y, beginnings):
            return [False, 0, 0]

        if calculated_neighborhood:
            valid_direction = calculated[target_y][target_x]
        else:
            valid_direction = not visited[target_y][target_x]

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


def get_user_mask(img_s, rect, source_name):
    user_mask = np.full((img_s.shape[0], img_s.shape[1]), 0, np.uint8)
    for y in range(rect[1], rect[1] + rect[3]):
        for x in range(rect[0], rect[0] + rect[2]):
            user_mask[y][x] = 255

    cv.imwrite("results/" + source_name + "/user_mask.jpg", user_mask)
    return user_mask


def run_test(rect, source_name, target_name, source_extension, target_extension):
    img_s = cv.imread('./images/' + source_name + '.' + source_extension)
    img_t = cv.imread('./images/' + target_name + '.' + target_extension)
    #cv.imwrite("results/" + source_name + "/source.jpg", img_s)
    #cv.imwrite("results/" + source_name + "/target.jpg", img_t)
    cv.imshow("Source", img_s)
    cv.waitKey(0)
    cv.imshow("Target", img_t)
    cv.waitKey(0)

    obj_mask, obj = grab_cut(img_s, rect, source_name)
    optimized_mask = optimized_boundary(img_s, img_t, obj_mask, rect, source_name)
    user_mask = get_user_mask(img_s, rect, source_name)
    full_mask = np.full((img_s.shape[0], img_s.shape[1]), 255, np.uint8)

    our_result = poisson_editing(img_s, img_t, optimized_mask)
    obj_result = poisson_editing(img_s, img_t, obj_mask)
    user_result = poisson_editing(img_s, img_t, user_mask)
    open_cv_result = poisson_editing(img_s, img_t, full_mask)

    #cv.imwrite("results/" + source_name + "/our_result.jpg", our_result)
    #cv.imwrite("results/" + source_name + "/object_directly_placed_result.jpg", obj_result)
    #cv.imwrite("results/" + source_name + "/user_result.jpg", user_result)
    #cv.imwrite("results/" + source_name + "/opencv_method_result.jpg", open_cv_result)

    cv.imshow("Our Result.jpg", our_result)
    cv.waitKey(0)
    cv.imshow("Object Directly Placed Result.jpg", obj_result)
    cv.waitKey(0)
    cv.imshow("User Selection Result", user_result)
    cv.waitKey(0)
    cv.imshow("Full Image Result", open_cv_result)
    cv.waitKey(0)


def run_airplane_test():
    rect = (100, 100, 625, 235)
    source_name = 'airplane'
    target_name = 'sky'
    source_extension = 'jpg'
    target_extension = 'jpg'
    run_test(rect, source_name, target_name, source_extension, target_extension)


def run_moon_test():
    rect = (66, 168, 546, 483)
    source_name = 'moon'
    target_name = 'night'
    source_extension = 'jpg'
    target_extension = 'jpg'
    run_test(rect, source_name, target_name, source_extension, target_extension)


def run_patrick_test():
    rect = (100, 5, 310, 280)
    source_name = 'patrick'
    target_name = 'ocean'
    source_extension = 'jpg'
    target_extension = 'jpg'
    run_test(rect, source_name, target_name, source_extension, target_extension)


if __name__ == '__main__':
    run_airplane_test()

