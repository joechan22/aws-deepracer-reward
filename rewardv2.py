import math
import numpy as np

def reward_function(params):
    """
    params
    {
    "all_wheels_on_track": Boolean,        # flag to indicate if the agent is on the track
    "x": float,                            # agent's x-coordinate in meters
    "y": float,                            # agent's y-coordinate in meters
    "closest_objects": [int, int],         # zero-based indices of the two closest objects to the agent's current position of (x, y).
    "closest_waypoints": [int, int],       # indices of the two nearest waypoints.
    "distance_from_center": float,         # distance in meters from the track center 
    "is_crashed": Boolean,                 # Boolean flag to indicate whether the agent has crashed.
    "is_left_of_center": Boolean,          # Flag to indicate if the agent is on the left side to the track center or not. 
    "is_offtrack": Boolean,                # Boolean flag to indicate whether the agent has gone off track.
    "is_reversed": Boolean,                # flag to indicate if the agent is driving clockwise (True) or counter clockwise (False).
    "heading": float,                      # agent's yaw in degrees
    "objects_distance": [float, ],         # list of the objects' distances in meters between 0 and track_length in relation to the starting line.
    "objects_heading": [float, ],          # list of the objects' headings in degrees between -180 and 180.
    "objects_left_of_center": [Boolean, ], # list of Boolean flags indicating whether elements' objects are left of the center (True) or not (False).
    "objects_location": [(float, float),], # list of object locations [(x,y), ...].
    "objects_speed": [float, ],            # list of the objects' speeds in meters per second.
    "progress": float,                     # percentage of track completed
    "speed": float,                        # agent's speed in meters per second (m/s)
    "steering_angle": float,               # agent's steering angle in degrees
    "steps": int,                          # number steps completed
    "track_length": float,                 # track length in meters.
    "track_width": float,                  # width of the track
    "waypoints": [(float, float), ]        # list of (x,y) as milestones along the track center
    }
    """

    DEFAULT_REWARD = 1.0 # math.exp(-5 * distance_from_center)
    LOWEST_REWARD = 1e-3
    DIRECTION_THRESHOLD = 8.0
    ABS_STEERING_THRESHOLD = 20.0
    PROGRESS_THRESHOLD = 75
    REINFORCE_FACTOR_1 = 1.2
    REINFORCE_FACTOR_2 = 1.5
    REINFORCE_FACTOR_3 = 1.3
    REINFORCE_FACTOR_4 = 1.28
    PUNISH_FACTOR_1 = 0.8
    PUNISH_FACTOR_2 = 0.5
    PUNISH_FACTOR_3 = 0.6
    PUNISH_FACTOR_4 = 0.73

    # parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    is_left_of_center = params['is_left_of_center']
    all_wheels_on_track = params['all_wheels_on_track']
    abs_steering = abs(params['steering_angle'])
    speed = params['speed']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints'] 
    heading = params['heading']
    progress = params['progress']
    is_crashed = params['is_crashed']
    is_reversed = params['is_reversed']
    is_offtrack = params['is_offtrack']
    steps = params['steps']

    # reward class
    class RewardClass(object):
        def chk_exception(self, ret_reward, exception):
            if exception:
                return LOWEST_REWARD
            return DEFAULT_REWARD

        def chk_on_track(self, ret_reward, on_track):
            if on_track:
                return DEFAULT_REWARD
            return LOWEST_REWARD
        
        def chk_center_distance(self, ret_reward, width, distance):
            marker_1 = 0.1 * width
            marker_2 = 0.25 * width
            marker_3 = 0.5 * width

            if distance <= marker_1:
                ret_reward = ret_reward * REINFORCE_FACTOR_1
            elif distance <= marker_2:
                ret_reward = ret_reward * PUNISH_FACTOR_1
            elif distance <= marker_3:
                ret_reward = ret_reward * PUNISH_FACTOR_2
            else:
                ret_reward = LOWEST_REWARD
            return ret_reward
        
        def chk_straight_line(self, ret_reward, abs_steering, speed):
            if abs_steering < 0.1 and speed >= 2.8:
                ret_reward = ret_reward * REINFORCE_FACTOR_2
            elif abs_steering < 0.2 and speed > 2.2:
                ret_reward = ret_reward * REINFORCE_FACTOR_1
            return ret_reward
        
        def is_speed_up(self, ret_reward, waypoints, closest_waypoints, speed, min_step=3, future_step=8):
            speed_up = False
            next_waypoint = waypoints[closest_waypoints[1]]
            prev_waypoint = waypoints[closest_waypoints[0]]
            further_waypoint = waypoints[min(len(waypoints) - 1, closest_waypoints[1] + future_step)]

            direction_degree = math.degrees(math.atan2(prev_waypoint[1] - next_waypoint[1], 
                                                        prev_waypoint[0] - next_waypoint[0]))
            future_degree = math.degrees(math.atan2(prev_waypoint[1] - further_waypoint[1], 
                                                 prev_waypoint[0]-further_waypoint[0]))

            difference = abs(direction_degree - future_degree)
            
            distance = np.linalg.norm([next_waypoint[0] - further_waypoint[0],
                                      next_waypoint[1] - further_waypoint[1]]) 
            
            diff = difference if difference < 180 else 360 - difference

            if difference < DIRECTION_THRESHOLD:
                speed_up = True
            else:
                if distance < 1.1:
                    speed_up = False
                else:
                    further_waypoint = waypoints[min(len(waypoints) - 1, closest_waypoints[1] + min_step)]
                    future_degree = math.degrees(math.atan2(prev_waypoint[1] - further_waypoint[1], 
                                                            prev_waypoint[0]-further_waypoint[0]))
                    difference = abs(direction_degree - future_degree)
                    diff = difference if difference < 180 else 360 - difference
                    if diff < DIRECTION_THRESHOLD:
                        speed_up = True
            if speed_up and speed > 2.25:
                ret_reward = ret_reward * REINFORCE_FACTOR_4
            elif not speed_up and speed < 1.4:
                ret_reward = ret_reward * REINFORCE_FACTOR_1
            return ret_reward
        
        def chk_direction(self, ret_reward, waypoints, closest_waypoints, heading):
            next_waypoint = waypoints[closest_waypoints[1]]
            prev_waypoint = waypoints[closest_waypoints[0]]

            direction_degree = math.degrees(math.atan2(next_waypoint[1] - prev_waypoint[1], next_waypoint[0] - prev_waypoint[0]))

            difference = abs(direction_degree - heading)

            if difference > DIRECTION_THRESHOLD:
                ret_reward = ret_reward * PUNISH_FACTOR_3

            return ret_reward
        
        def chk_steering(self, ret_reward, steering):
            if abs(steering) > ABS_STEERING_THRESHOLD:
                ret_reward = ret_reward * 0.9
            return ret_reward

        def chk_steering_rate(self, ret_reward, speed, steering):
            if speed > 2.5 - (0.4 * abs(steering)):
                ret_reward = ret_reward * PUNISH_FACTOR_1
            return ret_reward
        
        def chk_is_left_of_center(self, ret_reward, is_left_of_center):
            if is_left_of_center:
                ret_reward = ret_reward * REINFORCE_FACTOR_1
            else:
                ret_reward = ret_reward * PUNISH_FACTOR_1
            return ret_reward
        
        def chk_progress(self, ret_reward, progress):
            if progress > PROGRESS_THRESHOLD:
                ret_reward = ret_reward * REINFORCE_FACTOR_3
            return ret_reward
        
        def chk_speed(self, ret_reward, speed):
            if speed < 1.8:
                ret_reward = ret_reward * PUNISH_FACTOR_4
            elif speed > 2.2:
                ret_reward = ret_reward * REINFORCE_FACTOR_4
            return ret_reward
        
    r = RewardClass()
    reward = DEFAULT_REWARD
    reward = r.chk_on_track(reward, all_wheels_on_track)
    reward = r.chk_exception(reward, is_offtrack)
    reward = r.chk_exception(reward, is_crashed)
    reward = r.chk_exception(reward, is_reversed)
    reward = r.chk_center_distance(reward, track_width, distance_from_center)
    reward = r.is_speed_up(reward, waypoints, closest_waypoints, speed)
    reward = r.chk_is_left_of_center(reward, is_left_of_center)
    reward = r.chk_progress(reward, progress)
    # reward = r.chk_speed(reward, speed)

    return float(reward)
