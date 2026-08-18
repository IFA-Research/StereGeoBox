#@title Matching 
import numpy as np

def improved_match_boxes(left_boxes, right_boxes,
                         y_t=80, d_min=5, d_max=400,
                         s_t=0.4, a_t=0.5, minScore=-180):
    if not left_boxes or not right_boxes:
        return []

    def get_props(box):
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        area = w * h
        ar = w / h if h > 1e-5 else 1.0
        return cx, cy, area, ar

    left_sorted = sorted(enumerate(left_boxes), key=lambda x: get_props(x[1])[0])
    right_sorted = sorted(enumerate(right_boxes), key=lambda x: get_props(x[1])[0])

    match_list = []
    used_right = set()
    last_right_x = -np.inf

    for orig_l_idx, l_box in left_sorted:
        leftX, leftY, leftArea, leftAR = get_props(l_box)
        best_j, best_score, best_right_x, best_disp = None, -np.inf, None, 0

        for orig_r_idx, r_box in right_sorted:
            if orig_r_idx in used_right:
                continue
            rightX, rightY, rightArea, rightAR = get_props(r_box)
            if rightX <= last_right_x or leftX <= rightX:
                continue
            yDiff = abs(leftY - rightY)
            if yDiff >= y_t:
                continue
            xDiff = abs(leftX - rightX)
            if xDiff <= d_min or xDiff >= d_max:
                continue
            sizeRatio = min(leftArea, rightArea) / max(leftArea, rightArea)
            aRatio = min(leftAR, rightAR) / max(leftAR, rightAR)
            if sizeRatio <= s_t or aRatio <= a_t:
                continue
            score = (-yDiff - 0.4 * xDiff
                     - 0.5 * abs(1 - sizeRatio) * 100
                     - 0.3 * abs(1 - aRatio) * 100)
            if score > best_score:
                best_score = score
                best_j = orig_r_idx
                best_right_x = rightX
                best_disp = xDiff

        if best_j is not None and best_score > minScore:
            match_list.append((orig_l_idx, best_j, best_disp))
            used_right.add(best_j)
            last_right_x = best_right_x

    return match_list
