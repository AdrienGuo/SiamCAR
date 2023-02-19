import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np


def draw_boxes(image, boxes, type=None, scores: np = None):
    """
    Args:
        image: type=array
        boxes: (box_num, [x1, y1, x2, y2])
        type: template / pred / gt
        scores: type=array
    """
    image_new = np.copy(image)
    image_new = np.ascontiguousarray(image_new)
    boxes = np.asarray(boxes, dtype=np.int32)

    if type == "template":
        color = (0, 0, 255)    # red
        thickness = 2
    elif type == "pred":
        color = (0, 255, 0)    # green
        thickness = 1
    elif type == "gt":
        color = (255, 0, 0)    # blue
        thickness = 3
    else:
        color = (255, 255, 255)    # white?
        thickness = 1

    # draw targets
    for idx, box in enumerate(boxes):
        # 畫框框
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        cv2.rectangle(image_new, pt1, pt2, color=color, thickness=thickness)

        # 在框框上面打分數
        if np.any(scores):
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            score = f"{scores[idx]:.3f}"
            labelSize = cv2.getTextSize(
                score, fontFace, fontScale, thickness=1)
            _x1 = box[0]    # bottomleft x of text
            _y1 = box[1]    # bottomleft y of text
            _x2 = box[0] + labelSize[0][0]    # topright x of text
            _y2 = box[1] + labelSize[0][1]    # topright y of text
            cv2.rectangle(image_new, (_x1, _y1), (_x2, _y2),
                          (0, 255, 0), cv2.FILLED)   # text background
            cv2.putText(image_new, score, (_x1, _y2), fontFace,
                        fontScale, color=(0, 0, 0), thickness=1)

    return image_new


# Ref: https://medium.com/%E6%89%8B%E5%AF%AB%E7%AD%86%E8%A8%98/grad-cam-introduction-d0e48eb64adb
def draw_heatmap(img, heatmap, alpha=0.9):
    assert len(img.shape) == 3, "ERROR, img is wrong!!"
    img_h, img_w = img.shape[:2]

    # Convert bgr to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.uint8(img)

    # Resize heatmap as image
    heatmap = cv2.resize(heatmap, (img_w, img_h))

    # Display
    # plt will cause memory leak if not clear it.
    fig, ax = plt.subplots(num=1, clear=True)
    # Probability Info
    ax.text(img_w // 2, -10,
            f"Max: {str(np.around(heatmap.max(), 3))}", fontsize=12)
    plt.imshow(img, alpha=alpha)
    # Set the colorbar range (vmin, vmax)
    plt.imshow(heatmap, alpha=0.2, cmap="jet", vmin=0.0, vmax=1.0)
    plt.colorbar()

    return plt
