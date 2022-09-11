#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import scipy.ndimage as ndimg
import skimage.measure as skm
from heapq import heappush, heappop

# import messages
from messages import *

# cpg network

class HexapodExplorer:

    def __init__(self):
        s2 = math.sqrt(2)
        self.movements = [(1, 0, 1),
                     (0, 1, 1),
                     (-1, 0, 1),
                     (0, -1, 1),
                     (1, 1, s2),
                     (-1, 1, s2),
                     (-1, -1, s2),
                     (1, -1, s2)]

    def bresenham_line(self, start, goal):
        """Bresenham's line algorithm
        Args:
            start: (float64, float64) - start coordinate
            goal: (float64, float64) - goal coordinate
        Returns:
            interlying points between the start and goal coordinate
        """
        (x0, y0) = start
        (x1, y1) = goal
        line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                line.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                line.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        x = goal[0]
        y = goal[1]
        return line

    def fuse_laser_scan(self, grid_map, laser_scan, odometry):
        """ Method to fuse the laser scan data sampled by the robot with a given
            odometry into the probabilistic occupancy grid map
        Args:
            grid_map: OccupancyGrid - gridmap to fuse te laser scan to
            laser_scan: LaserScan - laser scan perceived by the robot
            odometry: Odometry - perceived odometry of the robot
        Returns:
            grid_map_update: OccupancyGrid - gridmap updated with the laser scan data
        """
        grid_map_update = copy.deepcopy(grid_map)

        if laser_scan is None or odometry is None:
            return grid_map_update

        d = np.array(laser_scan.distances[:-1])
        d[d < laser_scan.range_min] = np.nan
        d[d > laser_scan.range_max] = np.nan
        angles = np.arange(laser_scan.angle_min, laser_scan.angle_max + laser_scan.angle_increment,
                           laser_scan.angle_increment)
        x = d * np.cos(angles)
        y = d * np.sin(angles)
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        p = np.array([x, y, np.zeros(len(x))])
        o = odometry.pose.orientation.to_R() @ p + np.array(
            [[odometry.pose.position.x], [odometry.pose.position.y], [0]])
        origin = np.array([[grid_map.origin.position.x], [grid_map.origin.position.y]])
        o = o[[0, 1], :] - origin
        o = np.round(o / grid_map.resolution).astype(int)
        robot_pos = (np.array([[odometry.pose.position.x], [odometry.pose.position.y]]) - origin) / grid_map.resolution
        robot_pos = np.round(robot_pos).astype(int)
        free_points, occupied_points = set(), set()

        for i in range(np.size(o, 1)):
            pt = (o[0, i], o[1, i])
            for point in self.bresenham_line((robot_pos[0, 0], robot_pos[1, 0]), pt):
                free_points.add(point)
            occupied_points.add(pt)

        data = grid_map.data.reshape(grid_map_update.height, grid_map_update.width)
        for free_point in free_points:
            P_mi = data[free_point[1], free_point[0]]
            p_mi = 0.025 * P_mi / (0.025 * P_mi + 0.975 * (1 - P_mi))
            p_mi = 0.05 if p_mi < 0.05 else p_mi
            data[free_point[1], free_point[0]] = p_mi
        for occupied_point in occupied_points:
            P_mi = data[occupied_point[1], occupied_point[0]]
            p_mi = 0.975 * P_mi / (0.975 * P_mi + 0.025 * (1 - P_mi))
            p_mi = 0.95 if p_mi > 0.95 else p_mi
            data[occupied_point[1], occupied_point[0]] = p_mi
        grid_map_update.data = data.flatten()

        return grid_map_update

    def find_free_edge_frontiers(self, grid_map):
        """Method to find the free-edge frontiers (edge clusters between the free and unknown areas)
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """
        data_orig = grid_map.data.reshape(grid_map.height, grid_map.width)
        data_w = np.zeros(data_orig.shape)
        data_w[data_orig == 0.5] = 10
        data_w[data_orig < 0.5] = 1
        mask = np.ones((3, 3))
        mask[1][1] = 0
        data_c = ndimg.convolve(data_w, mask, mode='constant', cval=0.0)
        frontier = np.zeros(data_c.shape)
        frontier[(data_c > 10) & (data_c % 10 > 0)] = 1
        frontier[data_orig >= 0.5] = 0
        labeled_image, num_labels = skm.label(frontier, connectivity=2, return_num=True)
        poses = []
        for i in range(num_labels + 1):
            if np.any(frontier[labeled_image == i] == 0):
                continue
            pos = np.average(np.argwhere(labeled_image == i), 0)
            pose = Pose(Vector3(pos[1] * grid_map.resolution + grid_map.origin.position.x,
                         pos[0] * grid_map.resolution + grid_map.origin.position.y, 0), Quaternion(1, 0, 0, 0))
            poses.append(pose)
        return poses

    def find_inf_frontiers(self, grid_map):
        """Method to find the frontiers based on information theory approach
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """

        # TODO:[t1e_expl] find the information rich points in the environment
        return None

    def grow_obstacles(self, grid_map, robot_size):
        """ Method to grow the obstacles to take into account the robot embodiment
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            robot_size: float - size of the robot
        Returns:
            grid_map_grow: OccupancyGrid - gridmap with considered robot body embodiment
        """

        grid_map_grow = copy.deepcopy(grid_map)

        r_grid_size = round(robot_size / grid_map_grow.resolution)
        mask = [[False for i in range(2 * r_grid_size + 1)] for j in range(2 * r_grid_size + 1)]
        for i in range(-r_grid_size, r_grid_size + 1):
            for j in range(-r_grid_size, r_grid_size + 1):
                if math.hypot(i, j) < r_grid_size:
                    mask[i + r_grid_size][j + r_grid_size] = True

        for i in range(grid_map_grow.data.shape[0]):
            for j in range(grid_map_grow.data.shape[1]):
                if grid_map.data[i][j] >= 0.5:
                    grid_map_grow.data[i][j] = 1
                    for k in range(-r_grid_size, r_grid_size + 1):
                        for l in range(-r_grid_size, r_grid_size + 1):
                            if mask[k + r_grid_size][l + r_grid_size] and 0 <= i + k < grid_map_grow.data.shape[0] and 0 <= j + l < grid_map_grow.data.shape[1]:
                                grid_map_grow.data[i + k][j + l] = 1

        return grid_map_grow

    def plan_path(self, grid_map, start, goal):
        """ Method to plan the path from start to the goal pose on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """
        # code taken from https://github.com/richardos/occupancy-grid-a-star (MIT license), simplified and modified
        # to fit our framework
        path = Path()

        # get array indices of start and goal
        start_x = round((start.position.x - grid_map.origin.position.x) / grid_map.resolution)
        start_y = round((start.position.y - grid_map.origin.position.y) / grid_map.resolution)
        goal_x = round((goal.position.x - grid_map.origin.position.x) / grid_map.resolution)
        goal_y = round((goal.position.y - grid_map.origin.position.y) / grid_map.resolution)

        start_coord = (start_x, start_y)
        goal_coord = (goal_x, goal_y)
        visited = set()

        # add start node to front
        # front is a list of (total estimated cost to goal, total cost from start to node, node, previous node)
        start_node_estimate = math.hypot(goal_coord[0] - start_coord[0], goal_coord[1] - start_coord[1])
        front = [(start_node_estimate, 0, start_coord, None)]

        # use a dictionary to remember where we came from in order to reconstruct the path later on
        came_from = {}

        # while there are elements to investigate in our front.
        while front:
            # get smallest item and remove from front.
            element = heappop(front)

            # if this has been visited already, skip it
            total_cost, cost, pos, previous = element
            if pos in visited:
                continue

            # now it has been visited, mark with cost
            visited.add(pos)
            # set its previous node
            came_from[pos] = previous

            # if the goal has been reached, we are done!
            if pos == goal_coord:
                break

            # check all neighbors
            for dx, dy, deltacost in self.movements:
                # determine new position
                new_x = pos[0] + dx
                new_y = pos[1] + dy
                new_pos = (new_x, new_y)

                # check whether new position is inside the map
                # if not, skip node
                if not (0 <= new_x < grid_map.width and 0 <= new_y and grid_map.height):
                    continue

                # add node to front if it was not visited before and is not an obstacle
                if (new_pos not in visited) and (not grid_map.data[new_pos[1]][new_pos[0]]):
                    new_cost = cost + deltacost
                    new_total_cost = new_cost + math.hypot(goal_coord[0] - new_pos[0], goal_coord[1] - new_pos[1])
                    heappush(front, (new_total_cost, new_cost, new_pos, pos))

        # reconstruct path backwards (only if we reached the goal)
        path_coords = []
        if pos == goal_coord:
            while pos:
                path_coords.append(pos)
                pos = came_from[pos]

            # reverse so that path is from start to goal.
            path_coords.reverse()
        else:
            return None
        path_coords = path_coords[1:-1]
        path.poses.append(start)
        for pos in path_coords:
            path.poses.append(Pose(Vector3(pos[0] * grid_map.resolution + grid_map.origin.position.x, pos[1] * grid_map.resolution + grid_map.origin.position.y, 0), Quaternion(1, 0, 0, 0)))
        path.poses.append(goal)
        return path

    def simplify_path(self, grid_map, path):
        """ Method to simplify the found path on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            path: Path - path to be simplified
        Returns:
            path_simple: Path - simplified path
        """
        def x_pose_to_idx(x_pose):
            return round((x_pose - grid_map.origin.position.x) / grid_map.resolution)

        def y_pose_to_idx(y_pose):
            return round((y_pose - grid_map.origin.position.y) / grid_map.resolution)

        if path is None:
            return None

        # pseudocode taken from course ware page and modified
        pose_idx = 1
        path_simplified = Path()
        path_simplified.poses.append(path.poses[0])

        # iterate through the path and simplify the path
        while not path_simplified.poses[-1] == path.poses[-1]:  # until the goal is not reached
            last_pose = path_simplified.poses[-1]
            last_pos = (x_pose_to_idx(last_pose.position.x), y_pose_to_idx(last_pose.position.y))
            previous_pose = path_simplified.poses[-1]
            for pose in path.poses[pose_idx:]:
                pos = (x_pose_to_idx(pose.position.x), y_pose_to_idx(pose.position.y))
                line = self.bresenham_line(last_pos, pos)
                collide = False
                for point in line:
                    if grid_map.data[point[1]][point[0]]:
                        collide = True
                        break
                if collide:
                    path_simplified.poses.append(previous_pose)
                    break
                else:
                    previous_pose = pose
                    pose_idx = pose_idx + 1
                if pose == path.poses[-1]:
                    path_simplified.poses.append(pose)
                    break
            if pose == path.poses[-1]:
                path_simplified.poses.append(pose)
                break

        return path_simplified

    def plan_path_incremental(self, grid_map, start, goal):
        """ Method for incremental path planning from start to the goal pose on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
            rhs: float[] - one-step lookahead objective function in row-major order
            g: float[] - objective function value in row-major order
        """
        grid_map_c = copy.deepcopy(grid_map)
        grid_map_c.data = np.reshape(grid_map_c.data, (grid_map.height, grid_map.width))
        start_x = np.floor((start.position.x - grid_map.origin.position.x) / grid_map.resolution).astype(int)
        start_y = np.floor((start.position.y - grid_map.origin.position.y) / grid_map.resolution).astype(int)
        goal_x = np.floor((goal.position.x - grid_map.origin.position.x) / grid_map.resolution).astype(int)
        goal_y = np.floor((goal.position.y - grid_map.origin.position.y) / grid_map.resolution).astype(int)

        start_pos = (start_x, start_y)
        goal_pos = (goal_x, goal_y)
        if not hasattr(self, 'rhs'):  # first run of the function
            self.u = []
            self.g = np.full((grid_map.height, grid_map.width), np.inf)
            self.rhs = np.full((grid_map.height, grid_map.width), np.inf)
            self.prev = {}
            self.rhs[goal_y][goal_x] = 0
            heappush(self.u, (self.calculate_key(goal_pos), goal_pos))
            self.compute_shortest_path(grid_map_c, start_pos, goal_pos)
        elif np.any(grid_map_c.data != self.prev_grid_map):
            where = np.where(grid_map_c.data != self.prev_grid_map)
            for y, x in zip(where[0], where[1]):
                self.g[y][x], self.rhs[y][x] = np.inf, np.inf
                for dx, dy, cost in self.movements:
                    if 0 <= x + dx < grid_map.width and 0 <= y + dy < grid_map.height and self.prev[(x + dx, y + dy)] == (x, y):
                        self.update_vertex(grid_map_c, (x, y), goal_pos)
            new_u = []
            for s in self.u:
                heappush(new_u, (self.calculate_key(s[1]), s[1]))
            self.u = new_u
            self.compute_shortest_path(grid_map_c, start_pos, goal_pos)
        self.prev_grid_map = grid_map_c.data
        if start_pos not in self.prev:
            return None
        pos = start_pos
        path_coords = []
        while pos:
            path_coords.append(pos)
            pos = self.prev[pos]
        path_coords = path_coords[1:-1]
        path = Path()
        path.poses.append(start)
        for pos in path_coords:
            path.poses.append(Pose(Vector3((pos[0] + 1/2) * grid_map.resolution + grid_map.origin.position.x,
                                           (pos[1] + 1/2) * grid_map.resolution + grid_map.origin.position.y, 0),
                                   Quaternion(1, 0, 0, 0)))
        path.poses.append(goal)
        return path, self.rhs.flatten(), self.g.flatten()

    def compute_shortest_path(self, grid_map, start_pos, goal_pos):
        self.prev[goal_pos] = None
        while len(self.u) > 0 and self.u[0][0] < self.calculate_key(start_pos) or self.rhs[start_pos[1]][start_pos[0]] != self.g[start_pos[1]][start_pos[0]]:
            current_cost, pos = heappop(self.u)
            if self.g[pos[1]][pos[0]] > self.rhs[pos[1]][pos[0]]:
                self.g[pos[1]][pos[0]] = self.rhs[pos[1]][pos[0]]
            else:
                self.g[pos[1]][pos[0]] = np.inf
                self.update_vertex(grid_map, pos, goal_pos)
            for dx, dy, cost in self.movements:
                if 0 <= pos[0] + dx < grid_map.width and 0 <= pos[1] + dy < grid_map.height:
                    self.update_vertex(grid_map, (pos[0] + dx, pos[1] + dy), goal_pos)

    def update_vertex(self, grid_map, pos, goal_pos):
        if pos != goal_pos:
            min_rhs = np.inf
            min_pos = None
            for dx, dy, cost in self.movements:
                if 0 <= pos[0] + dx < grid_map.width and 0 <= pos[1] + dy < grid_map.height:
                    new_pos = (pos[0] + dx, pos[1] + dy)
                    if grid_map.data[new_pos[1], new_pos[0]]:
                        cost = np.inf
                    if cost + self.g[pos[1] + dy][pos[0] + dx] < min_rhs:
                        min_rhs = cost + self.g[pos[1] + dy][pos[0] + dx]
                        min_pos = (pos[0] + dx, pos[1] + dy)
            self.prev[pos] = min_pos
            self.rhs[pos[1]][pos[0]] = min_rhs
        self.u[:] = [x for x in self.u if x[1] != pos]
        if self.g[pos[1]][pos[0]] != self.rhs[pos[1]][pos[0]]:
            heappush(self.u, (self.calculate_key(pos), pos))

    def calculate_key(self, pos):
        return min(self.g[pos[1]][pos[0]], self.rhs[pos[1]][pos[0]])
