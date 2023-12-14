import math

import cv2
import numpy as np
import torch

BOX_CLASSES = ("car", "cyclist", "pedestrian")


def plot_boxes_on_image_and_in_bev(
    predictions_img,
    img,
    canvas_bev,
    plot_color,
    p2,
    box_class_list=BOX_CLASSES,
    use_classwise_color=False,
    show_3d=True,
    show_bev=True,
    thickness=4,
    bev_scale=30.0,
):
    # https://sashamaps.net/docs/resources/20-colors/
    class_color_map = {
        "car": (255, 51, 153),
        "cyclist": (255, 130, 48),  # Orange
        "bicycle": (255, 130, 48),  # Orange
        "pedestrian": (138, 43, 226),  # Violet
        "bus": (0, 0, 0),  # Black
        "construction_vehicle": (0, 130, 200),  # Blue
        "motorcycle": (220, 190, 255),  # Lavender
        "trailer": (170, 255, 195),  # Mint
        "truck": (128, 128, 99),  # Olive
        "traffic_cone": (255, 225, 25),  # Yellow
        "barrier": (128, 128, 128),  # Grey
    }

    if predictions_img is not None and predictions_img.size > 0:
        # Add dimension if there is a single point
        if predictions_img.ndim == 1:
            predictions_img = predictions_img[np.newaxis, :]

        N = predictions_img.shape[0]
        # Add projected 3d center information to predictions_img
        class_name = box_class_list
        cls = predictions_img[:, 0]
        h3d = predictions_img[:, 6]
        w3d = predictions_img[:, 7]
        l3d = predictions_img[:, 8]
        x3d = predictions_img[:, 9]
        y3d = predictions_img[:, 10] - h3d / 2
        z3d = predictions_img[:, 11]
        ry3d = predictions_img[:, 12]

        for j in range(N):
            box_class = class_name[int(cls[j])].lower()
            if box_class == "dontcare":
                continue
            if box_class in box_class_list:
                if use_classwise_color:
                    box_plot_color = class_color_map[box_class]
                else:
                    box_plot_color = plot_color

                box_plot_color = box_plot_color[::-1]
                if show_3d:
                    verts_cur, _ = project_3d(
                        p2,
                        x3d[j],
                        y3d[j],
                        z3d[j],
                        w3d[j],
                        h3d[j],
                        l3d[j],
                        ry3d[j],
                        return_3d=True,
                    )
                    draw_3d_box(
                        img,
                        verts_cur,
                        color=box_plot_color,
                        thickness=thickness,
                    )
                if show_bev:
                    draw_bev(
                        canvas_bev,
                        z3d[j],
                        l3d[j],
                        w3d[j],
                        x3d[j],
                        ry3d[j],
                        color=box_plot_color,
                        scale=bev_scale,
                        thickness=thickness,
                        text=None,
                    )


def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """

    if type(x3d) == np.ndarray:
        p2_batch = np.zeros([x3d.shape[0], 4, 4])
        p2_batch[:, :, :] = p2[np.newaxis, :, :]

        ry3d_cos = np.cos(ry3d)
        ry3d_sin = np.sin(ry3d)

        R = np.zeros([x3d.shape[0], 4, 3])
        R[:, 0, 0] = ry3d_cos
        R[:, 0, 2] = ry3d_sin
        R[:, 1, 1] = 1
        R[:, 2, 0] = -ry3d_sin
        R[:, 2, 2] = ry3d_cos

        corners_3d = np.zeros([x3d.shape[0], 3, 8])

        # setup X
        corners_3d[:, 0, :] = -l3d[:, np.newaxis] / 2
        corners_3d[:, 0, 1:5] = l3d[:, np.newaxis] / 2

        # setup Y
        corners_3d[:, 1, :] = -h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 2:4] = h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 6:8] = h3d[:, np.newaxis] / 2

        # setup Z
        corners_3d[:, 2, :] = -w3d[:, np.newaxis] / 2
        corners_3d[:, 2, 3:7] = w3d[:, np.newaxis] / 2

        # rotate
        corners_3d = R @ corners_3d

        # translate
        corners_3d[:, 0, :] += x3d[:, np.newaxis]
        corners_3d[:, 1, :] += y3d[:, np.newaxis]
        corners_3d[:, 2, :] += z3d[:, np.newaxis]
        corners_3d[:, 3, :] = 1

        # project to 2D
        corners_2d = p2_batch @ corners_3d

        corners_2d[:, :2, :] /= corners_2d[:, 2, :][:, np.newaxis, :]

        verts3d = corners_2d

    elif type(x3d) == torch.Tensor:
        p2_batch = torch.zeros(x3d.shape[0], 4, 4)
        p2_batch[:, :, :] = p2[np.newaxis, :, :]

        ry3d_cos = torch.cos(ry3d)
        ry3d_sin = torch.sin(ry3d)

        R = torch.zeros(x3d.shape[0], 4, 3)
        R[:, 0, 0] = ry3d_cos
        R[:, 0, 2] = ry3d_sin
        R[:, 1, 1] = 1
        R[:, 2, 0] = -ry3d_sin
        R[:, 2, 2] = ry3d_cos

        corners_3d = torch.zeros(x3d.shape[0], 3, 8)

        # setup X
        corners_3d[:, 0, :] = -l3d[:, np.newaxis] / 2
        corners_3d[:, 0, 1:5] = l3d[:, np.newaxis] / 2

        # setup Y
        corners_3d[:, 1, :] = -h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 2:4] = h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 6:8] = h3d[:, np.newaxis] / 2

        # setup Z
        corners_3d[:, 2, :] = -w3d[:, np.newaxis] / 2
        corners_3d[:, 2, 3:7] = w3d[:, np.newaxis] / 2

        # rotate
        corners_3d = torch.bmm(R, corners_3d)

        corners_3d = corners_3d.to(x3d.device)
        p2_batch = p2_batch.to(x3d.device)

        # translate
        corners_3d[:, 0, :] += x3d[:, np.newaxis]
        corners_3d[:, 1, :] += y3d[:, np.newaxis]
        corners_3d[:, 2, :] += z3d[:, np.newaxis]
        corners_3d[:, 3, :] = 1

        # project to 2D
        corners_2d = torch.bmm(p2_batch, corners_3d)

        corners_2d[:, :2, :] /= corners_2d[:, 2, :][:, np.newaxis, :]

        verts3d = corners_2d

    else:
        # compute rotational matrix around yaw axis
        R = np.array(
            [
                [+math.cos(ry3d), 0, +math.sin(ry3d)],
                [0, 1, 0],
                [-math.sin(ry3d), 0, +math.cos(ry3d)],
            ]
        )

        # 3D bounding box corners
        x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
        y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
        z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

        x_corners += -l3d / 2
        y_corners += -h3d / 2
        z_corners += -w3d / 2

        # bounding box in object co-ordinate
        corners_3d = np.array([x_corners, y_corners, z_corners])

        # rotate
        corners_3d = R.dot(corners_3d)

        # translate
        corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

        corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
        corners_2D = p2.dot(corners_3D_1)
        corners_2D = corners_2D / corners_2D[2]

        bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

        verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    if return_3d:
        return verts3d, corners_3d
    else:
        return verts3d


def draw_3d_box(im, verts, color=(0, 200, 200), thickness=1):
    for lind in range(0, verts.shape[0] - 1):
        v1 = verts[lind]
        v2 = verts[lind + 1]
        cv2.line(
            im,
            (int(v1[0]), int(v1[1])),
            (int(v2[0]), int(v2[1])),
            color,
            thickness,
        )

    draw_transparent_polygon(im, verts[5:9, :], blend=0.5, color=color)


def draw_transparent_polygon(im, verts, blend=0.5, color=(0, 255, 255)):
    mask = get_polygon_grid(im, verts[:4, :])

    im[mask, 0] = im[mask, 0] * blend + (1 - blend) * color[0]
    im[mask, 1] = im[mask, 1] * blend + (1 - blend) * color[1]
    im[mask, 2] = im[mask, 2] * blend + (1 - blend) * color[2]


def get_polygon_grid(im, poly_verts):
    from matplotlib.path import Path

    nx = im.shape[1]
    ny = im.shape[0]
    # poly_verts = [(1, 1), (5, 1), (5, 9), (3, 2), (1, 1)]

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    path = Path(poly_verts)
    grid = path.contains_points(points)
    grid = grid.reshape((ny, nx))

    return grid


def draw_bev(
    canvas_bev,
    z3d,
    l3d,
    w3d,
    x3d,
    ry3d,
    color=(0, 200, 200),
    scale=1,
    thickness=2,
    text=None,
):
    w = l3d * scale
    ln = w3d * scale
    x = x3d * scale
    z = z3d * scale
    r = ry3d * -1

    corners1 = np.array(
        [
            [-w / 2, -ln / 2, 1],
            [+w / 2, -ln / 2, 1],
            [+w / 2, +ln / 2, 1],
            [-w / 2, +ln / 2, 1],
        ]
    )

    ry = np.array(
        [
            [+math.cos(r), -math.sin(r), 0],
            [+math.sin(r), math.cos(r), 0],
            [0, 0, 1],
        ]
    )

    corners2 = ry.dot(corners1.T).T

    corners2[:, 0] += x + canvas_bev.shape[1] / 2
    corners2[:, 1] += z

    draw_line(
        canvas_bev, corners2[0], corners2[1], color=color, thickness=thickness
    )
    draw_line(
        canvas_bev, corners2[1], corners2[2], color=color, thickness=thickness
    )
    draw_line(
        canvas_bev, corners2[2], corners2[3], color=color, thickness=thickness
    )
    draw_line(
        canvas_bev, corners2[3], corners2[0], color=color, thickness=thickness
    )

    if text is not None:
        thickness = 2
        cv2.putText(
            canvas_bev,
            text,
            (int(corners2[0, 0]), int(corners2[0, 1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )


def draw_line(im, v1, v2, color=(0, 200, 200), thickness=1):
    cv2.line(
        im,
        (int(v1[0]), int(v1[1])),
        (int(v2[0]), int(v2[1])),
        color,
        thickness,
    )
