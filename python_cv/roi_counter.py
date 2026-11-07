# ROI / virtual line crossing
# A vehicle is counted when its tracked center crosses the configured line.

def crossed_line(previous_point, current_point, line_y):
    if previous_point is None:
        return False
    return previous_point[1] < line_y <= current_point[1]
