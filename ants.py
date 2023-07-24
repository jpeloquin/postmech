import concurrent.futures
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from numbers import Number
import os
from pathlib import Path
import shlex
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple, Union
from zipfile import ZipFile

import cv2
from matplotlib import font_manager
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat, savemat

from .images import get_frame_size
from .imagej import mask_from_xy
from .util import clean_dir, ensure_dir


def read_affine(pth: Union[str, Path]):
    ants_affine = loadmat(pth)
    a = ants_affine["AffineTransform_double_2_2"]
    affine = np.array(
        [[a[0][0], a[1][0], a[-2][0]], [a[2][0], a[3][0], a[-1][0]], [0, 0, 1]]
    )
    return affine


def write_affine(affine: NDArray, pth: Union[str, Path]):
    serialized = np.hstack([affine[:-1, :-1].ravel(), affine[:-1, -1]])
    mat = {
        "AffineTransform_double_2_2": np.atleast_2d(serialized).T,
        "fixed": np.zeros((affine.shape[0] - 1, 1)),
    }
    savemat(pth, mat, format="4")  # ANTs requires format 4


def plan_registration(frames, reference_frame=None) -> List[Tuple[str, Optional[str]]]:
    """Return list of frames and what to use to initialize their registration

    Returns a list of tuples, `(frame, frame to use for initialization)`, in the order
    in which the registrations should be done.  If there should not be an initialization
    for that frame, the second element of the tuple is None.

    """
    if reference_frame is None:
        reference_frame = frames[0]
    first = []
    second = []
    before_reference = True
    for i, frame in enumerate(frames):
        if frame == reference_frame:
            first.append((frame, None))
            before_reference = False
        elif before_reference:
            second.append((frame, frames[i + 1]))
        else:  # after reference
            first.append((frame, frames[i - 1]))
    plan = first + second[::-1]
    return plan


def register(
    fixed,
    moving,
    ref_mask,
    dir_out,
    parameters,
    initial_affine=None,
    verbose=False,
    env={},
):
    """Register two images using ANTs

    :param fixed: Path to fixed (reference) image.

    :param moving: Path moving (deformed) image.

    :param ref_mask: Mask marking which pixels to consider in registration (value =
    1) or ignore (value = 0).  The mask is appplied to the fixed (reference) image.

    :param dir_out: Directory to which to write ANTs files.  If it does not exist,
    it will be created.

    :param parameters: Comma-delimited list of parameters (strings) that will be
    passed to the the ANTs call via `subprocess`.  Its primary use is to control the
    ANTs registration stages using `--transform`, `--metric`, `--convergence`,
    `--shrink-factors`, and `--smothing-sigmas`.  :func:`register` will add the other
    required parameters automatically.  In the `--metric` parameter, write `{fixed}`
    in place of the fixed image path, and write `{moving}` in place of the moving
    image path.

    :param initial_affine: (Optional) Path to affine file (ITK .mat format),
    which will be used to initialize the moving → fixed image registration.  The
    affine therefore is interpreted as a transformation from moving image
    /coordinates/ to fixed image coordinates.

    """
    dir_out = ensure_dir(dir_out)
    # Get frame IDs
    fixed_name = str(Path(fixed).with_suffix("").name)
    moving_name = str(Path(moving).with_suffix("").name)
    # Put ANTs on PATH
    env = {k: str(v) for k, v in env.items()}
    if "ANTSPATH" not in env:
        if os.getenv("ANTSPATH") is None:
            raise ValueError(f"The ANTSPATH environment variable is not set.")
        else:
            env["ANTSPATH"] = os.getenv("ANTSPATH")
    env["PATH"] = f"{env['ANTSPATH']}:{os.getenv('PATH')}"
    # Call antsRegistration
    # fmt: off
    cmd = [
        "antsRegistration",
        "--dimensionality", "2",
        "--collapse-output-transforms",
        "--masks", f"{ref_mask}",
    ]
    # fmt: on
    if initial_affine is not None:
        cmd += ["--initial-moving-transform", str(initial_affine)]
    for c in parameters:
        c = c.replace("{fixed}", str(fixed))
        c = c.replace("{moving}", str(moving))
        cmd.append(c)
    cmd += ["-o", f"{dir_out}/{moving_name}_to_{fixed_name}_"]
    if verbose:
        cmd += ["--verbose"]
    subprocess.run(cmd, env=env, cwd=os.getcwd(), check=True)
    p = f"{dir_out}/{moving_name}_to_{fixed_name}_0GenericAffine.mat"
    return p, cmd


def plot_roi(img: Union[str, Path, Image.Image], vertices, center=None, name=None):
    """Plot ROI pts on an image

    Note that if an image object is provided, it will be modified.

    """
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    artist = ImageDraw.Draw(img)
    artist.polygon([(x, y) for x, y, *_ in vertices], fill=None, outline="white")
    if center is not None:
        artist.line(
            ((center[0] - 3, center[1]), (center[0] + 3, center[1])), fill="white"
        )
        artist.line(
            ((center[0], center[1] - 3), (center[0], center[1] + 3)), fill="white"
        )
    if name is not None:
        font = font_manager.FontProperties(family="sans-serif", weight="normal")
        fname = font_manager.findfont(font)
        fnt = ImageFont.truetype(fname, 16)
        color = (255, 255, 255, 255) if img.mode == "RGB" else 255
        artist.text(center, name, font=fnt, fill=color)
    return img


def plot_rois(
    rois: Dict,
    rois_table: DataFrame,
    root: Union[str, Path],
    dir_out: Union[str, Path],
):
    """Plot tracks of one or more ROIs onto images"""
    root = Path(root)
    dir_out = ensure_dir(Path(dir_out))
    roi_names = [
        c.removesuffix(" affine") for c in rois_table.columns if c.endswith(" affine")
    ]
    for i in rois_table.index:
        nm_img = rois_table.loc[i, "Name"]
        p_img = root / rois_table.loc[i, "Image"]
        img = Image.open(p_img)
        for roi_name in roi_names:
            roi = rois[roi_name]
            p_affine = root / rois_table.loc[i, f"{roi_name} affine"]
            A = read_affine(p_affine)
            plot_roi(img, *transformed_roi(roi, A), name=roi_name)
        img.save((dir_out / nm_img).with_suffix(".png"))


def read_roi_tracks(p):
    """Read ROI tracks .csv with ROI centroid positions as Numpy arrays"""
    tab = pd.read_csv(p)
    for c in tab.columns:
        if not c.endswith("centroid"):
            continue
        tab[c] = list(map(lambda s: np.fromstring(s[1:-1], sep=" "), tab[c]))
    return tab


def track_ROI(
    archive,
    frames,
    roi: List[Tuple],
    workdir,
    cmd,
    id_,
    exclusion_mask: Optional[Union[str, Path]] = None,
    reference_frame: Optional[str] = None,
    nproc: Optional[int] = None,
    **kwargs,
):
    """Track an ROI through a list of frames

    This is a convenience function for calling track_ROIs() with a single ROI"""
    return track_ROIs(
        archive,
        frames,
        {"ROI": roi},
        workdir,
        cmd,
        id_,
        exclusion_mask,
        reference_frame,
        nproc,
        **kwargs,
    )


def track_ROIs(
    archive,
    frames: List[str],
    rois: Dict[str, List[Tuple]],
    workdir,
    cmd,
    id_,
    exclusion_mask: Optional[Union[str, Path]] = None,
    reference_frame: Optional[str] = None,
    nproc: int = 1,
    **kwargs,
):
    """Track ROIs through a list of frames

    ROIs should be drawn in the reference frame.  If no reference frame is specified,
    it is assumed to be the first frame in `frames`.

    kwargs are passed directly to `register()`.

    """
    workdir = Path(workdir)
    # Sanitize list of frames
    frames = [Path(f).name for f in frames]
    # Assume reference frame is first frame unless otherwise specified
    if reference_frame is None:
        reference_frame = frames[0]
    # Extract images from zip archive once up-front, if necessary
    archive = Path(archive)
    if archive.is_dir():
        dir_images = archive
    elif archive.suffix == ".zip":
        # Images are in a .zip archive
        dir_images = clean_dir(workdir / f"{id_}_-_images")
        with ZipFile(archive) as a:
            for frame in frames:
                with open(dir_images / frame, "wb") as f:
                    f.write(a.read(frame))
    else:
        raise ValueError("Image archive must be a zip file or a directory.")
    # Get metadata about images
    frame_size = get_frame_size(archive)
    if exclusion_mask is not None:
        exclusion_mask = Image.open(exclusion_mask).convert("L")
    # Create masks
    mask_image = {}
    if exclusion_mask is not None:
        exclusion_mask = Image.open(exclusion_mask).convert("L")
    for nm, roi in rois.items():
        m = mask_from_xy(roi, frame_size)
        if exclusion_mask is not None:
            m.paste(Image.new("L", frame_size, 0), mask=exclusion_mask)
        mask_image[nm] = m
        p = workdir / f"{id_}_-_roi={nm}_-_mask.tiff"
        m.save(p)
    # Set up data structures for ROI tracking in all frames
    all_tracks = defaultdict(dict)
    frame_order = plan_registration(frames, reference_frame)
    queue = {nm: copy(frame_order[::-1]) for nm in rois}
    affines = {nm: {} for nm in rois}  # dict of roi name → dict of frame → affine
    # Handle reference frame
    ## Plot all ROIs, with labels, in reference frame
    dir_ROI_tracks = clean_dir(workdir / f"{id_}_-_all_ROIs_-_tracks")
    roi_label_img = Image.open(dir_images / reference_frame)
    for nm, pts in rois.items():
        plot_roi(roi_label_img, pts, np.mean(pts, axis=0), nm)
    ref_name = Path(reference_frame).with_suffix("").name
    roi_label_img.save(dir_ROI_tracks / f"{ref_name}.png")
    ## Create affines for reference frame
    for nm in rois:
        affine = np.eye(3)
        frame = str(Path(reference_frame).with_suffix("").name)
        dir_affines = clean_dir(workdir / f"{id_}_-_roi={nm}_-_affines")
        p_affine = dir_affines / f"{frame}_to_{frame}_0GenericAffine.mat"
        write_affine(affine, p_affine)

    def register_roi(frame, roi_name, initial_affine: Optional[Path], logf):
        p_def = dir_images / frame
        p_ref = dir_images / reference_frame
        p_mask = workdir / f"{id_}_-_roi={roi_name}_-_mask.tiff"
        dir_affines = workdir / f"{id_}_-_roi={roi_name}_-_affines"
        p_affine, cmd_out = register(
            p_ref, p_def, p_mask, dir_affines, cmd, initial_affine, **kwargs
        )
        logf.write(" ".join([shlex.quote(c) for c in cmd_out]) + "\n")
        affine = read_affine(p_affine)
        vertices, center = transformed_roi(rois[roi_name], affine=affine)
        info = {
            "Name": frame,
            "Image": os.path.relpath(dir_images / frame, workdir),
            "ROI": roi_name,
            "Affine": str(Path(p_affine).relative_to(workdir)),
            "Centroid": center,
        }
        return info

    def get_init_affine(init_frame, roi_name, affines=affines):
        if init_frame is not None:
            init_affine = affines[roi_name][init_frame]
        else:
            init_affine = None
        return init_affine

    p_log = workdir / f"{id_}_-_commands.log"
    futures = set()
    rois_completed = defaultdict(int)  # dict: frame → # rois done
    with open(p_log, "w", encoding="utf-8", buffering=1) as logf:
        with ThreadPoolExecutor(max_workers=nproc) as executor:
            for nm in rois:
                frame, init_frame = queue[nm].pop()
                init_affine = get_init_affine(init_frame, nm)
                futures.add(executor.submit(register_roi, frame, nm, init_affine, logf))
            while futures:
                done, futures = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    result = future.result()
                    affines[result["ROI"]][result["Name"]] = workdir / result["Affine"]
                    all_tracks[(result["Name"], result["Image"])].update(
                        {
                            f"{result['ROI']} centroid": result["Centroid"],
                            f"{result['ROI']} affine": result["Affine"],
                        }
                    )
                    # If there are more frames to run, run them
                    if queue[result["ROI"]]:
                        frame, init_frame = queue[result["ROI"]].pop()
                        init_affine = get_init_affine(init_frame, result["ROI"])
                        futures.add(
                            executor.submit(
                                register_roi, frame, result["ROI"], init_affine, logf
                            )
                        )
                    rois_completed[result["Name"]] += 1
                    # If a frame was completed, plot the result
                    if rois_completed[result["Name"]] == len(rois):
                        img = Image.open(workdir / result["Image"])
                        for nm, roi in rois.items():
                            A = read_affine(affines[nm][result["Name"]])
                            img = plot_roi(img, *transformed_roi(roi, A), name=nm)
                        img.save(
                            dir_ROI_tracks
                            / Path(result["Image"]).with_suffix(".png").name
                        )
    # Tabulate all ROI tracks together
    table = DataFrame.from_dict(all_tracks, orient="index").reset_index()
    table = table.rename({"level_0": "Name", "level_1": "Image"}, axis=1)
    return table


def transformed_roi(boundary: Iterable[Tuple[Number, Number]], affine=None):
    if affine is None:
        affine = np.eye(3)
    pts = [(affine @ [x, y, 1])[:2] for x, y in boundary]
    center = np.mean(pts, axis=0)
    return pts, center


def make_video(table, dir_tracks):
    dir_tracks = Path(dir_tracks)
    dir_out = dir_tracks.parent
    p_img = (dir_tracks / table.loc[0, "Name"]).with_suffix(".png")
    size = Image.open(str(p_img)).size
    video = cv2.VideoWriter(
        str(dir_tracks).removesuffix("_-_tracks") + ".mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        15,
        size,
    )
    for nm in table["Name"]:
        p_img = Path(dir_tracks / nm).with_suffix(".png")
        video.write(cv2.imread(str(p_img)))
