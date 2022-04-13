import io
from numbers import Number
import os
from pathlib import Path
import shlex
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple, Union
from zipfile import ZipFile

import numpy as np
from pandas import DataFrame
from PIL import Image, ImageDraw
from pathos.multiprocessing import ProcessPool
from scipy.io import loadmat

from .util import clean_dir, ensure_dir


def make_mask(roi, shape):
    mask = Image.new("L", shape, 0)
    artist = ImageDraw.Draw(mask)
    artist.polygon([(x, y) for x, y in roi], fill=255, outline=None)
    return mask


def read_affine(pth: Union[str, Path]):
    ants_affine = loadmat(pth)
    a = ants_affine["AffineTransform_double_2_2"]
    affine = np.array(
        [[a[0][0], a[1][0], a[-2][0]], [a[2][0], a[3][0], a[-1][0]], [0, 0, 1]]
    )
    affine = np.linalg.inv(affine)
    return affine


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
    dir_out = Path(dir_out)
    if not dir_out.exists():
        dir_out.mkdir()
    # Get frame IDs
    ref_name = str(Path(fixed).with_suffix("").name)
    def_name = str(Path(moving).with_suffix("").name)
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
    cmd += ["-o", f"{dir_out}/{ref_name}_to_{def_name}_"]
    if verbose:
        cmd += ["--verbose"]
    subprocess.run(cmd, env=env, cwd=os.getcwd(), check=True)
    p = f"{dir_out}/{ref_name}_to_{def_name}_0GenericAffine.mat"
    return p, cmd


def plot_roi(img: Union[str, Path, Image.Image], vertices, center):
    """Plot ROI pts on an image

    Note that if an image object is provided, it will be modified.

    """
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    artist = ImageDraw.Draw(img)
    artist.polygon([(x, y) for x, y, *_ in vertices], fill=None, outline="white")
    artist.line(((center[0] - 3, center[1]), (center[0] + 3, center[1])), fill="white")
    artist.line(((center[0], center[1] - 3), (center[0], center[1] + 3)), fill="white")
    return img


def track_ROI(archive, frames: List[str], roi_pts, workdir, cmd, id_, **kwargs):
    """Track an ROI through a list of frames

    kwargs are passed directly to `register()`.

    """
    workdir = Path(workdir)
    # Open one image so we can read image size
    with ZipFile(archive) as a:
        size = Image.open(io.BytesIO(a.read(frames[0]))).size
    roi_mask = make_mask(roi_pts, size)
    p_mask = workdir / f"{id_}_-_mask.tiff"
    roi_mask.save(p_mask)
    # Copy the images
    dir_images = clean_dir(workdir / f"{id_}_-_images")
    dir_affines = clean_dir(workdir / f"{id_}_-_affines")
    dir_tracks = clean_dir(workdir / f"{id_}_-_tracks")
    p_log = workdir / f"{id_}_-_commands.log"
    # Run the registrations
    p_ref = dir_images / frames[0]
    with ZipFile(archive) as a:
        with open(p_ref, "wb") as f:
            f.write(a.read(frames[0]))

    def process_frame(frame, logf, affine):
        p_def = dir_images / frame
        with ZipFile(archive) as a:
            with open(p_def, "wb") as f:
                f.write(a.read(frame))
        p_affine, cmd_out = register(p_ref, p_def, p_mask, dir_affines, cmd, affine, **kwargs)
        logf.write(" ".join([shlex.quote(c) for c in cmd_out]) + "\n")
        pts_affine = np.linalg.inv(read_affine(p_affine))
        vertices, center = transformed_roi(roi_pts, affine=pts_affine)
        img = plot_roi(p_def, vertices, center)
        img.save(dir_tracks / frame)
        info = {
            "Name": frame,
            "Image": (dir_images / frame).relative_to(workdir),
            "Affine": Path(p_affine).relative_to(workdir),
            "ROI centroid": center,
        }
        return info

    info = []
    with open(p_log, "w", encoding="utf-8", buffering=1) as logf:
        affine = None
        for frame in frames[1:]:
            row = process_frame(frame, logf, affine)
            affine = workdir / row["Affine"]
            info.append(row)
    info = DataFrame(info)
    return info


def track_ROIs(
    archive,
    frames: List[str],
    rois: Dict[str, List[Tuple]],
    workdir,
    cmd,
    sid,
    nproc: Optional[int] = None,
    **kwargs,
):
    """Track multiple ROIs through a list of frames

    kwargs are passed directly to register().

    """
    def process_roi(roi):
        nm, pts = roi
        info = track_ROI(archive, frames, pts, workdir, cmd, f"{sid}_-_ROI={nm}", **kwargs)
        info = info.rename({"ROI centroid": f"{nm} centroid"}, axis=1)
        return info
    if nproc is None:
        tracks = list(map(process_roi, rois.items()))
    else:
        with ProcessPool(nproc) as pool:
            tracks = pool.map(process_roi, rois.items())
    # Tabulate all ROI tracks together
    all_tracks = tracks[0].set_index("Name")
    for t in tracks[1:]:
        col = [c for c in t.columns if c.endswith("centroid")][0]
        all_tracks = all_tracks.join(t.set_index("Name")[[col]])
    all_tracks.reset_index(inplace=True)
    # Plot all ROI tracks together
    dir_tracks = clean_dir(workdir / f"{sid}_-_all_ROI_tracks")
    for i in range(len(all_tracks)):
        with ZipFile(archive) as a:
            img = Image.open(io.BytesIO(a.read(all_tracks["Name"].iloc[i])))
        for k, roi in rois.items():
            p_affine = workdir / all_tracks["Affine"].iloc[i]
            if p_affine is None:
                affine = None
            else:
                affine = read_affine(p_affine)
            img = plot_roi(img, *transformed_roi(roi, affine))
        img.save(dir_tracks / all_tracks["Name"].iloc[i])
    return all_tracks


def transformed_roi(boundary: Iterable[Tuple[Number, Number]], affine=None):
    if affine is None:
        affine = np.eye(3)
    pts = [(affine @ [x, y, 1])[:2] for x, y in boundary]
    center = np.mean(pts, axis=0)
    return pts, center
