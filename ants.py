from numbers import Number
import os
from pathlib import Path
import shlex
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple, Union
from zipfile import ZipFile

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame
from PIL import Image, ImageDraw, ImageFont
from pathos.multiprocessing import ProcessPool
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

    Returns a list of tuples, (frame, frame to use for initialization), in the order
    in which the registrations should be done.  If there should not be an
    initialization for that frame, the second element of the tuple is None.

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


def plot_roi(img: Union[str, Path, Image.Image], vertices, center, name=None):
    """Plot ROI pts on an image

    Note that if an image object is provided, it will be modified.

    """
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    artist = ImageDraw.Draw(img)
    artist.polygon([(x, y) for x, y, *_ in vertices], fill=None, outline="white")
    artist.line(((center[0] - 3, center[1]), (center[0] + 3, center[1])), fill="white")
    artist.line(((center[0], center[1] - 3), (center[0], center[1] + 3)), fill="white")
    if name is not None:
        fnt = ImageFont.truetype("Arial.ttf", 16)
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
        for k in roi_names:
            roi = rois[k]
            p_affine = root / rois_table.loc[i, f"{k} affine"]
            A = read_affine(p_affine)
            img = plot_roi(img, *transformed_roi(roi, A))
        img.save(dir_out / nm_img)


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
    frames: List[str],
    roi_pts,
    workdir,
    cmd,
    id_,
    exclusion_mask: Optional[Union[str, Path]] = None,
    reference_frame: Optional[str] = None,
    **kwargs,
):
    """Track an ROI through a list of frames

    ROIs should be drawn in the reference frame.  If no reference frame is provided,
    it is assumed to be the first frame in `frames`.

    kwargs are passed directly to `register()`.

    """
    archive = Path(archive)
    frames = [Path(f).name for f in frames]
    if reference_frame is None:
        reference_frame = frames[0]
    workdir = Path(workdir)
    size = get_frame_size(archive)
    mask = mask_from_xy(roi_pts, size)
    if exclusion_mask is not None:
        exclusion_mask = Image.open(exclusion_mask).convert("L")
        mask.paste(Image.new("L", size, 0), mask=exclusion_mask)
    p_mask = workdir / f"{id_}_-_mask.tiff"
    mask.save(p_mask)
    # Copy the images
    if archive.is_dir():
        # Re-use the existing image directory
        dir_images = archive
    elif archive.suffix == ".zip":
        # Extract the needed images
        dir_images = clean_dir(workdir / f"{id_}_-_images")
        with ZipFile(archive) as a:
            for frame in frames:
                with open(dir_images / frame, "wb") as f:
                    f.write(a.read(frame))
    else:
        raise ValueError("Image archive must be a zip file or a directory.")
    dir_affines = clean_dir(workdir / f"{id_}_-_affines")
    dir_tracks = clean_dir(workdir / f"{id_}_-_tracks")

    def process_frame(frame, initial_affine: Optional[Union[Path, str]], logf):
        p_def = dir_images / frame
        p_ref = dir_images / reference_frame
        if frame == reference_frame:
            affine = np.eye(3)
            nm = str(Path(reference_frame).with_suffix("").name)
            p_affine = dir_affines / f"{nm}_to_{nm}_0GenericAffine.mat"
            write_affine(affine, p_affine)
        else:
            p_affine, cmd_out = register(
                p_ref, p_def, p_mask, dir_affines, cmd, initial_affine, **kwargs
            )
            logf.write(" ".join([shlex.quote(c) for c in cmd_out]) + "\n")
            affine = read_affine(p_affine)
        vertices, center = transformed_roi(roi_pts, affine=affine)
        img = plot_roi(p_def, vertices, center)
        img.save(dir_tracks / frame)
        info = {
            "Name": frame,
            "Image": os.path.relpath(dir_images / frame, workdir),
            "Affine": str(Path(p_affine).relative_to(workdir)),
            "ROI centroid": center,
        }
        return info

    # Add all registration data to frame info table
    info_table = []
    p_affine = {}  # frame → path of affine
    plan = plan_registration(frames, reference_frame)
    p_log = workdir / f"{id_}_-_commands.log"
    with open(p_log, "w", encoding="utf-8", buffering=1) as logf:
        for frame, initializer in plan:
            if initializer is not None:
                # The registration plan is assumed to always do the initializer
                # frame's registration before its affine needs to be used to
                # initialize another frame's registration.
                initial_affine = p_affine[initializer]
            else:
                initial_affine = None
            row = process_frame(frame, initial_affine, logf)
            info_table.append(row)
            p_affine[frame] = workdir / row["Affine"]
    return DataFrame(info_table).sort_index()


def track_ROIs(
    archive,
    frames: List[str],
    rois: Dict[str, List[Tuple]],
    workdir,
    cmd,
    sid,
    exclusion_mask: Optional[Union[str, Path]] = None,
    reference_frame: Optional[str] = None,
    nproc: Optional[int] = None,
    **kwargs,
):
    """Track multiple ROIs through a list of frames

    kwargs are passed directly to register().

    """
    archive = Path(archive)
    if reference_frame is None:
        reference_frame = frames[0]
    # Extract images from zip archive once up-front, if necessary
    if archive.suffix == ".zip":
        # Images are in a .zip archive
        dir_images = clean_dir(workdir / f"{sid}_-_images")
        with ZipFile(archive) as a:
            for frame in frames:
                with open(dir_images / frame, "wb") as f:
                    f.write(a.read(frame))
    else:
        # Assume images are already in a plain directory (other archive types are not
        # supported)
        dir_images = archive

    def process_roi(roi):
        nm, pts = roi
        info = track_ROI(
            dir_images,
            frames,
            pts,
            workdir,
            cmd,
            f"{sid}_-_ROI={nm}",
            exclusion_mask=exclusion_mask,
            reference_frame=reference_frame,
            **kwargs,
        )
        info = info.rename({"ROI centroid": f"{nm} centroid"}, axis=1)
        info = info.rename({"Affine": f"{nm} affine"}, axis=1)
        return info

    # Plot all ROIs, with labels, in reference frame
    roi_label_img = Image.open(dir_images / reference_frame)
    for nm, pts in rois.items():
        plot_roi(roi_label_img, pts, np.mean(pts, axis=0), nm)
    ref_name = Path(reference_frame).with_suffix("").name
    roi_label_img.save(workdir / f"{sid}_-_all_ROIs_-_{ref_name}.png")
    # Track all ROIs in turn
    if nproc is None:
        tracks = list(map(process_roi, rois.items()))
    else:
        with ProcessPool(nproc) as pool:
            tracks = pool.map(process_roi, rois.items())
    # Tabulate all ROI tracks together
    all_tracks = tracks[0].set_index("Name")
    for t in tracks[1:]:
        col = [c for c in t.columns if (c.endswith("centroid") or c.endswith("affine"))]
        all_tracks = all_tracks.join(t.set_index("Name")[col])
    all_tracks.reset_index(inplace=True)
    # Plot all ROI tracks together
    dir_all_ROIs = clean_dir(workdir / f"{sid}_-_all_ROIs_tracks")
    plot_rois(rois, all_tracks, workdir, dir_all_ROIs)
    return all_tracks


def transformed_roi(boundary: Iterable[Tuple[Number, Number]], affine=None):
    if affine is None:
        affine = np.eye(3)
    pts = [(affine @ [x, y, 1])[:2] for x, y in boundary]
    center = np.mean(pts, axis=0)
    return pts, center
