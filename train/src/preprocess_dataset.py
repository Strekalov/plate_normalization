import os
import cv2
from pathlib import Path

names = [
    "-",
    "-",
    "Auto-orientation",
    "-",
    "Resize",
    "annotate",
    "-",
    "collaborate",
    "-",
    "collect",
    "-",
    "-",
    "-",
    "understand",
    "-",
    "use",
    "0",
    "1",
    "19",
    "2",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "Auto",
    "AutoPlate",
    "B",
    "C",
    "E",
    "For",
    "H",
    "K",
    "M",
    "No",
    "O",
    "P",
    "Roboflow",
    "T",
    "The",
    "The following",
    "This dataset",
    "To find",
    "X",
    "Y",
    "visit",
]


main_folder = 'datasets/car-plates-text'

for folder in os.listdir(main_folder):
    if not os.path.isdir(os.path.join(main_folder, folder)):
        continue
    for small_folder in os.listdir(os.path.join(main_folder, folder)):
        if small_folder == "labels":
            for file_path in os.listdir(
                os.path.join(main_folder, folder, small_folder)
            ):
                im_path = file_path.rsplit(".", 1)[0]
                with open(
                    os.path.join(main_folder, folder, small_folder, file_path), "r"
                ) as file:
                    lines = file.readlines()

                boxes = [list(map(float, line.split())) for line in lines]
                output_dir_path = os.path.join(
                    "datasets/datasets/car-plates-text_new/output_data", folder, small_folder
                )
                if not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path)

                def denormalize_coordinates(x, y, w, h, img_width, img_height):
                    x = int(x * img_width)
                    y = int(y * img_height)
                    w = int(w * img_width)
                    h = int(h * img_height)
                    return x, y, w, h

                def is_inside(inner, outer):
                    _, x1, y1, w1, h1 = inner
                    _, x2, y2, w2, h2 = outer
                    x1_min, x1_max = x1 - w1 / 2, x1 + w1 / 2
                    y1_min, y1_max = y1 - h1 / 2, y1 + h1 / 2
                    x2_min, x2_max = x2 - w2 / 2, x2 + w2 / 2
                    y2_min, y2_max = y2 - h2 / 2, y2 + h2 / 2
                    return (
                        x2_min <= x1_min
                        and x2_max >= x1_max
                        and y2_min <= y1_min
                        and y2_max >= y1_max
                    )

                image = cv2.imread(
                    os.path.join(main_folder, folder, "images", f"{im_path}.jpg")
                )
                image_height, image_width, _ = image.shape

                num_plates = sum(1 for box in boxes if int(box[0]) == 29)

                k = 1
                for i, box29 in enumerate(boxes):
                    if int(box29[0]) == 29:
                        cls29, x29, y29, w29, h29 = box29
                        x29, y29, w29, h29 = denormalize_coordinates(
                            x29, y29, w29, h29, image_width, image_height
                        )
                        top_left = (x29 - w29 // 2, y29 - h29 // 2)
                        bottom_right = (x29 + w29 // 2, y29 + h29 // 2)
                        cropped_image = image[
                            top_left[1] : bottom_right[1],
                            top_left[0] : bottom_right[0],
                        ]

                        if num_plates > 1:
                            cropped_image_path = os.path.join(
                                "datasets/car-plates-text_new/output_data",
                                folder,
                                "images",
                                f"{im_path}_{k}.jpg",
                            )
                        else:
                            cropped_image_path = os.path.join(
                                "datasets/car-plates-text_new/output_data", folder, "images", f"{im_path}.jpg"
                            )

                        if not os.path.exists(
                            os.path.join("datasets/car-plates-text_new/output_data", folder, "images")
                        ):
                            os.makedirs(
                                os.path.join("datasets/car-plates-text_new/output_data", folder, "images")
                            )

                        cv2.imwrite(cropped_image_path, cropped_image)

                        filename_without_extens = os.path.join(
                            "datasets/car-plates-text_new/output_data", folder, small_folder, im_path
                        )
                        if num_plates > 1:
                            filename_without_extens += f"_{k}"

                        if not os.path.exists(
                            os.path.join("datasets/car-plates-text_new/output_data", folder, small_folder)
                        ):
                            os.makedirs(
                                os.path.join("datasets/car-plates-text_new/output_data", folder, small_folder)
                            )

                        with open(filename_without_extens + ".txt", "w") as f:
                            for box in boxes:
                                if box != box29 and is_inside(box, box29):
                                    cls, x_s, y_s, w_s, h_s = box
                                    if int(cls) == 1:
                                        continue
                                    x_s, y_s, w_s, h_s = denormalize_coordinates(
                                        x_s,
                                        y_s,
                                        w_s,
                                        h_s,
                                        image_width,
                                        image_height,
                                    )
                                    x_new = (x_s - top_left[0]) / w29
                                    y_new = (y_s - top_left[1]) / h29
                                    w_new = w_s / w29
                                    h_new = h_s / h29
                                    f.write(
                                        " ".join(
                                            map(
                                                str, [0, x_new, y_new, w_new, h_new]
                                            )
                                        )
                                        + "\n"
                                    )

                        k += 1

root_folder = 'datasets'

for dataset_folder in os.listdir(root_folder):
    if dataset_folder == 'new_data':
        continue
    if dataset_folder == 'plates_keypoints':
        continue
    for tr_test_val_folder in os.listdir(root_folder+'/'+dataset_folder):
        if not tr_test_val_folder in ['train', 'test', 'val', 'valid']:
            continue
        cur_label_folder = root_folder+'/'+dataset_folder+'/'+tr_test_val_folder+'/labels/'
        for file_path in os.listdir(cur_label_folder):
            with open(cur_label_folder+file_path, 'r') as file:
                lines = file.readlines()
            # Ignore the first line (header) and strip any whitespace/newline characters from the lines
            updated_lines = []
            for line in lines:
                parts = line.split()
                parts[0] = '0'
                updated_lines.append(' '.join(parts))
            with open(cur_label_folder+file_path, 'w') as outfile:
                outfile.write('\n'.join(updated_lines))