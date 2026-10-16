"""Relative bearing and approximate proximity for a player-centric radar."""

import math

import cv2
import numpy as np

VIEW_WIDTH = 320
NEAR_HEIGHT = 40
MID_HEIGHT = 18
RADAR_RADIUS = 52

PROXIMITY_RADIUS = {"near": 0.38, "mid": 0.62, "far": 0.86}


def bearing_degrees(x_center, view_width=VIEW_WIDTH):
    """Horizontal bearing in degrees. Ahead is 0, right is positive.

    Doom's horizontal field of view is 90 degrees, so half the view
    width is 45 degrees.
    """
    half = view_width / 2.0
    return math.degrees(math.atan((x_center - half) / half))


def proximity_label(box_height):
    if box_height >= NEAR_HEIGHT:
        return "near"
    if box_height >= MID_HEIGHT:
        return "mid"
    return "far"


def locate(bbox, view_width=VIEW_WIDTH):
    x1, y1, x2, y2 = bbox
    bearing = bearing_degrees((x1 + x2) / 2.0, view_width)
    proximity = proximity_label(y2 - y1)
    return bearing, proximity


def draw_radar(frame, blips, origin, radius=RADAR_RADIUS):
    """Draw a radar inset. ``blips`` are ``(bearing_degrees, proximity)``."""
    center_x, center_y = origin
    cv2.circle(frame, origin, radius, (20, 20, 20), -1)
    cv2.circle(frame, origin, radius, (180, 180, 180), 1)
    cv2.circle(frame, origin, int(radius * 0.55), (80, 80, 80), 1)
    cv2.line(frame, (center_x, center_y - radius), (center_x, center_y + radius), (70, 70, 70), 1)
    cv2.line(frame, (center_x - radius, center_y), (center_x + radius, center_y), (70, 70, 70), 1)
    cv2.circle(frame, origin, 3, (220, 220, 220), -1)
    for bearing, proximity in blips:
        radians = math.radians(bearing)
        distance = radius * PROXIMITY_RADIUS.get(proximity, 0.86)
        point_x = int(round(center_x + math.sin(radians) * distance))
        point_y = int(round(center_y - math.cos(radians) * distance))
        cv2.circle(frame, (point_x, point_y), 4, (40, 40, 220), -1)
    return frame


def enemy_blips(detections):
    blips = []
    for detection in detections:
        if detection.class_name != "enemy":
            continue
        blips.append(locate(detection.bbox))
    return blips
