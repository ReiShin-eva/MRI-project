import os
import argparse
import nibabel as nib
import numpy as np
import random
from scipy.ndimage import zoom
import imageio

# ── config ─────────────────────────────────────────────
SLICE_SIZE = 96
Z_MIN = 60
Z_MAX = 130
# ───────────────────────────────────────────────────────


def get_random_z(seg_data):
    max_z = seg_data.shape[2] - 1
    z_min = max(0, Z_MIN)
    z_max = min(max_z, Z_MAX)
    return random.randint(z_min, z_max)


def extract_slice(nifti_path, z, size=SLICE_SIZE, order=1):
    data = nib.load(nifti_path).get_fdata()
    slice_2d = data[:, :, z]

    h, w = slice_2d.shape
    resized = zoom(slice_2d, (size / h, size / w), order=order)

    return resized


def process_patient(root_dir, patient, case_dir):
    base = os.path.join(root_dir, patient)

    seg_path = os.path.join(base, f"{patient}-seg.nii.gz")
    t1c_path = os.path.join(base, f"{patient}-t1c.nii.gz")
    t1n_path = os.path.join(base, f"{patient}-t1n.nii.gz")
    t2f_path = os.path.join(base, f"{patient}-t2f.nii.gz")
    t2w_path = os.path.join(base, f"{patient}-t2w.nii.gz")

    # Load seg to determine z + pixel count
    seg_data = nib.load(seg_path).get_fdata()
    z = get_random_z(seg_data)

    seg_slice = extract_slice(seg_path, z, order=0)
    pixel_count = int(np.count_nonzero(seg_slice))

    # Normalize helper
    def normalize(img):
        if np.max(img) > 0:
            img = img / np.max(img)
        return (img * 255).astype(np.uint8)

    # Save segmentation
    seg_filename = f"tumor_seg_{pixel_count}.png"
    imageio.imwrite(os.path.join(case_dir, seg_filename), normalize(seg_slice))

    # Save modalities (same z)
    modalities = {
        "sample1_t1c.png": t1c_path,
        "sample1_t1n.png": t1n_path,
        "sample1_t2f.png": t2f_path,
        "sample1_t2w.png": t2w_path,
    }

    for name, path in modalities.items():
        img_slice = extract_slice(path, z)
        imageio.imwrite(os.path.join(case_dir, name), normalize(img_slice))

    return pixel_count, z


def create_cases(root_dir):
    output_root = os.path.join(root_dir, "cases")
    os.makedirs(output_root, exist_ok=True)

    patients = sorted([
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f))
        and os.path.isfile(os.path.join(root_dir, f, f"{f}-seg.nii.gz"))
    ])

    print(f"Found {len(patients)} patients\n")

    for i, patient in enumerate(patients, start=1):
        case_dir = os.path.join(output_root, f"case_{i}")
        os.makedirs(case_dir, exist_ok=True)

        try:
            px, z = process_patient(root_dir, patient, case_dir)
            print(f"{patient} -> case_{i} | px={px} z={z}")
        except Exception as e:
            print(f"[SKIP] {patient}: {e}")

    print("\nDone.")
    print(f"Cases saved in: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="Path to BraTS-MEN-Train")
    args = parser.parse_args()

    create_cases(args.root_dir)