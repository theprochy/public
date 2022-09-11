#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import scipy.spatial

import collections
import heapq

import Environment as env

#import communication messages
from messages import *

from matplotlib import pyplot as plt

class RRRTNode():
    def __init__(self, parent, pose, cost, id):
        self.parent = parent
        self.pose = pose
        self.cost = cost
        self.id = id

    def total_cost(self):
        cost, cur = 0, self
        while cur.parent is not None:
            cost = cost + cur.cost
            cur = cur.parent
        return cost

class RRTPlanner():

    def __init__(self, environment, translate_speed, rotate_speed):
        """Initialize the sPRM solver
        Args:  environment: Environment - Map of the environment that provides collision checking
               translate_speed: float - speed of translation motion
               rotate_speed: float - angular speed of rotation motion
        """
        self.environment = environment
        self.translate_speed = translate_speed
        self.rotate_speed = rotate_speed
        self.nodes = []

    def duration(self, p1, p2):
        """Compute duration of movement between two configurations
        Args:  p1: Pose - start pose
               p2: Pose - end pose
        Returns: float - duration in seconds 
        """
        t_translate = (p1.position - p2.position).norm() / self.translate_speed
        t_rotate = p1.orientation.dist(p2.orientation) / self.rotate_speed
        return max(t_translate, t_rotate)

    def create_edge(self, p1, p2, collision_step):
        """Sample an edge between start and goal
        Args:  p1: Pose - start pose
               p2: Pose - end pose
               collision_step: float - minimal time for testing collisions [s]
        Returns: Pose[] - list of sampled poses between the start and goal 
        """
        t = self.duration(p1, p2)
        steps_count = math.ceil(t / collision_step)
        if steps_count <= 1:
            return [p1, p2]
        else:
            parameters = np.linspace(0.,1.,steps_count+1)
            return slerp_se3(p1, p2, parameters)  

    def check_collision(self, pth):
        """Check the collision status along a given path
        """
        for se3 in pth:
            if self.environment.check_robot_collision(se3):
                return True
        return False    

    def Steer(self, x_nearest, x_rand, steer_step):
        """Steer function of RRT algorithm 
        Args:  x_nearest: Pose - pose from which the tree expands
               x_rand: Pose - random pose towards which the tree expands
               steer_step: float - maximum distance from x_nearest [s]
        Returns: Pose - new pose to be inserted into the tree
        """
        t = self.duration(x_nearest, x_rand)
        if t < steer_step:
            return x_rand
        else:
            parameter = steer_step / t
            return slerp_se3(x_nearest, x_rand, [parameter])[0]

    def expand_tree(self, start, space, number_of_samples, neighborhood_radius, collision_step,
            isrrtstar, steer_step):
        """Expand the RRT(*) tree for finding the shortest path
        Args:  start: Pose - start configuration of the robot in SE(3) coordinates
               space: String - configuration space type
               number_of_samples: int - number of samples to be generated
               neighborhood_radius: float - neighborhood radius to connect samples [s]
               collision_step: float - minimum step to check collisions [s]
               isrrtstar: bool - determine RRT* variant
               steer_step: float - step utilized of steering function
        Returns:  NavGraph - the navigation graph for visualization of the built roadmap
        """

        root = RRRTNode(None, start, 0.0, 0)
        self.nodes = [root]
        next_id = 1

        for i in range(number_of_samples):
            x_rand = None
            x, y, z = np.random.uniform(self.environment.limits_x[0], self.environment.limits_x[1]), np.random.uniform(self.environment.limits_y[0], self.environment.limits_y[1]), np.random.uniform(self.environment.limits_z[0], self.environment.limits_z[1])
            v = Vector3(x, y, z)
            if space == "R2":
                x_rand = Pose(v, Quaternion( 0, 0, 0, 1))
            elif space == "SE(2)":
                while True:
                    xq, yq, zq, wq = 0, 0, np.random.uniform(-1, 1), np.random.uniform(-1, 1)
                    q = Quaternion(xq, yq, zq, wq)
                    if q.magnitude() <= 1:
                        q.normalize()
                        x_rand = Pose(v, q)
                        break

            x_nearest, d_nearest = None, math.inf
            for node in self.nodes:
                d = self.duration(x_rand, node.pose)
                if d < d_nearest:
                    d_nearest = d
                    x_nearest = node
            x_new = self.Steer(x_nearest.pose, x_rand, steer_step)
            if not self.check_collision(self.create_edge(x_nearest.pose, x_new, collision_step)):
                if not isrrtstar:
                    self.nodes.append(RRRTNode(x_nearest, x_new, self.duration(x_nearest.pose, x_new), next_id))
                    next_id = next_id + 1
                else:
                    X_near = [node for node in self.nodes if self.duration(x_new, node.pose) < neighborhood_radius]
                    x_min = x_nearest
                    c_min = x_nearest.total_cost() + self.duration(x_nearest.pose, x_new)
                    for x_near in X_near:
                        c_near = x_near.total_cost() + self.duration(x_near.pose, x_new)
                        if c_near < c_min and not self.check_collision(self.create_edge(x_near.pose, x_new, collision_step)):
                            c_min = c_near
                            x_min = x_near
                    x_new = RRRTNode(x_min, x_new, self.duration(x_min.pose, x_new), next_id)
                    self.nodes.append(x_new)
                    next_id = next_id + 1
                    for x_near in X_near:
                        if x_new.total_cost() + self.duration(x_new.pose, x_near.pose) < x_near.total_cost() and not\
                            self.check_collision(self.create_edge(x_new.pose, x_near.pose, collision_step)):
                            x_near.parent = x_new
                            x_near.cost = self.duration(x_near.pose, x_new.pose)

        navgraph = NavGraph()
        navgraph.poses = [node.pose for node in self.nodes]
        for node in self.nodes[1:]:
            navgraph.edges.append((node.parent.id, node.id))
        return navgraph

    def query(self, goal, neighborhood_radius, collision_step, isrrtstar):
        """Retrieve path for the goal configuration
        Args:  goal: Pose - goal configuration of the robot in SE(3) coordinates
               neighborhood_radius: float - neighborhood radius to connect samples [s]
               collision_step: float - minimum step to check collisions [s]
               isrrtstar: bool - determine RRT* variant
        Returns:  Path - the path between the start and the goal Pose in SE(3) coordinates
        """
        path = Path()

        x_nearest, d_nearest = None, math.inf
        for node in self.nodes:
            d = self.duration(goal, node.pose)
            if d < d_nearest:
                d_nearest = d
                x_nearest = node
        cur = x_nearest
        path.poses = self.create_edge(goal, cur.pose, collision_step)
        while cur.parent is not None:
            path.poses.extend(self.create_edge(cur.pose, cur.parent.pose, collision_step))
            cur = cur.parent
        path.poses.reverse()
        return path

########################################################
# HELPER functions
########################################################
def slerp_se3(start, end, parameters):
    """Method to compute spherical linear interpolation between se3 poses
    Args:  start: Pose - starting pose
           end: Pose - end pose
           parameters: float[] - array of parameter in (0,1), 0-start, 1-end
    Returns: steps: Pose[] - list of the interpolated se3 poses, always at least [start,end]
    """
    #extract the translation
    t1 = start.position
    t2 = end.position
    #extract the rotation
    q1 = start.orientation
    q2 = end.orientation
    #extract the minimum rotation angle
    theta = max(0.01, q1.dist(q2)) / 2 
    if (q1.x*q2.x + q1.y*q2.y + q1.z*q2.z + q1.w*q2.w) < 0:
        q2 = q2 * -1

    steps = []
    for a in parameters:
        ta = t1 + (t2-t1)*a
        qa = q1*np.sin( (1-a)*theta ) + q2*np.sin( a*theta )
        qa.normalize()
        pose = Pose(ta, qa)
        steps.append(pose)

    return steps

