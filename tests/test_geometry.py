import numpy as np

from perception import geometry as geo


def test_iou_identical_boxes():
    box = (0.0, 0.0, 10.0, 10.0)
    assert geo.iou(box, box) == 1.0


def test_iou_disjoint_boxes():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (20.0, 20.0, 30.0, 30.0)
    assert geo.iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 5.0, 15.0, 15.0)
    # intersection = 5x5=25, union = 100+100-25=175
    assert abs(geo.iou(a, b) - 25 / 175) < 1e-6


def test_angle_between_centers_horizontal():
    a = (0.0, 0.0, 2.0, 2.0)  # center (1,1)
    b = (10.0, 0.0, 12.0, 2.0)  # center (11,1) -> same y, angle 0
    assert abs(geo.angle_between_centers(a, b) - 0.0) < 1e-6


def test_angle_between_centers_vertical():
    a = (0.0, 0.0, 2.0, 2.0)  # center (1,1)
    b = (0.0, 10.0, 2.0, 12.0)  # center (1,11) -> same x, angle 90
    assert abs(geo.angle_between_centers(a, b) - 90.0) < 1e-6


def test_region_brightness_solid_white():
    image = np.full((20, 20, 3), 255, dtype=np.uint8)  # BGR white
    brightness = geo.region_brightness(image, (0, 0, 20, 20))
    assert brightness > 250


def test_region_brightness_solid_black():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    brightness = geo.region_brightness(image, (0, 0, 20, 20))
    assert brightness < 5


def test_left_right_height_balance():
    detections = [
        {"label": "a", "bbox": (0, 0, 5, 20)},   # left, height 20
        {"label": "b", "bbox": (60, 0, 65, 5)},  # right, height 5
    ]
    result = geo.left_right_height_balance(detections, image_width=100)
    assert result["left_higher"] is True


def test_area_filter_drops_oversized_box():
    detections = [
        {"label": "curtain", "bbox": (0, 0, 90, 90)},  # covers 81% of a 100x100 frame
        {"label": "door", "bbox": (0, 0, 20, 40)},
    ]
    filtered = geo.area_filter(detections, image_area=100 * 100, max_ratio=0.6)
    assert len(filtered) == 1
    assert filtered[0]["label"] == "door"
