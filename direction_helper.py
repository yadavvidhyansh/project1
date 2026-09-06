"""
direction_helper.py
Calculates spatial direction (left/center/right, near/far) of a detected object
within the camera frame.
"""

def get_direction(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    box_height = y2 - y1

    left_boundary = frame_width * 0.35
    right_boundary = frame_width * 0.65

    if center_x < left_boundary:
        horizontal = "on your left"
    elif center_x > right_boundary:
        horizontal = "on your right"
    else:
        horizontal = "straight ahead"

    relative_height = box_height / frame_height

    if relative_height > 0.5:
        proximity = "very close"
    elif relative_height > 0.25:
        proximity = "nearby"
    else:
        proximity = "a bit far"

    return f"{horizontal}, {proximity}"