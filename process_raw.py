from pathlib import Path
from tqdm import tqdm

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT_DIR = Path("/app/nyuv2_rawdata")

ROOM_TYPE = [
    # 'reception',
    # 'nyu_office',

    # 'living_room',
    # 'living',
    # 'conference_room',

    # 'office',
    # 'laundry_room',
    # 'dinette',
    # 'bookstore',

    # 'furniture_store',
    # 'home_office',
    # 'home',
    # 'printer_room',
    # 'indoor_balcony',
    # 'cafe',

    # 'reception_room',
    # 'student_lounge',
    # 'kitchen',
    # 'dining',
    # 'basement',
    # 'computer',

    # 'dining_room',
    # 'playroom',
    # 'computer_lab',
    # 'home_storage',
    'bedroom',
    'study_room',

    # 'excercise_room',
    # 'study',
    # 'office_kitchen',
    # 'bathroom',
    # 'classroom',
    # 'foyer',
]


def path_to_time(path: Path) -> float:
    return float(path.parts[-1].split("-")[1])


for room_type in ROOM_TYPE:
    N = len(list(ROOT_DIR.glob(f"{room_type}*")))
    for room_dir in tqdm(ROOT_DIR.glob(f"{room_type}*"), desc=room_type, total=N):
        rgb_files = sorted([(path_to_time(x), x) for x in room_dir.glob("r-*")])
        rgb_time_arr = np.array([x for x, _ in rgb_files])
        depth_files = sorted([(path_to_time(x), x) for x in room_dir.glob("d-*")])

        seen = set()
        for i, (t_depth, f_depth) in tqdm(enumerate(depth_files)):
            idx = np.argmin(np.abs(rgb_time_arr - t_depth))
            if idx in seen:
                print(f"WARNING: {idx} already seen: {rgb_files[idx]}")
            seen.add(idx)
            t_rgb, f_rgb = rgb_files[idx]

            try:
                img = Image.open(f_rgb)
            except UnidentifiedImageError:
                print("ERROR BAD FILE:", f_rgb)
                continue
            try:
                depth = Image.open(f_depth)
            except UnidentifiedImageError:
                print("ERROR BAD FILE:", f_depth)
                continue

            img.save(f_rgb.parent / f"rgb_{i:05d}.jpg")
            depth.save(f_depth.parent / f"sync_depth_{i:05d}.png")
