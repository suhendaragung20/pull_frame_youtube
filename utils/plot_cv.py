import cv2


def plot_object(frame, box, idx_face, condition, conf):
    overlay = frame.copy()

    (H, W) = frame.shape[:2]

    box_border = int(W / 400)

    font_size = 0.6
    (startX, startY, endX, endY) = box

    y = startY - 10 if startY - 10 > 10 else startY + 10

    yBox = y + 5

    if condition == 'Masked':
        fill_color = (228, 108, 104)
    else:
        fill_color = (0, 0, 200)

    cv2.rectangle(overlay, (startX, startY), (endX, endY),
                  (255, 255, 255), box_border+4)


    cv2.rectangle(overlay, (startX, startY), (endX, endY),
                  fill_color, box_border+2)

    text = condition + ', ' + idx_face + ' (' + str(int(conf*100)) + '%)'
    font_scale = (0.4*box_border)
    thick = box_border

    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thick)[0]
    # set the text start position
    text_offset_x = startX
    text_offset_y = yBox
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(overlay, box_coords[0], box_coords[1], fill_color, cv2.FILLED)

    cv2.putText(overlay, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick)

    alpha = 0.6  # Transparency factor.

    # Following line overlays transparent rectangle over the image
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame