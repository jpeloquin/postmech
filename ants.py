import io
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
from PIL import Image, ImageDraw
from pathos.multiprocessing import ProcessPool
from scipy.io import loadmat, savemat

from .images import get_frame_size
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


def write_affine(affine: NDArray, pth: Union[str, Path]):
    serialized = np.hstack([affine[:-1, :-1].ravel(), affine[:-1, -1]])
    mat = {
        "AffineTransform_double_2_2": np.atleast_2d(serialized).T,
        "fixed": np.zeros((affine.shape[0] - 1, 1)),
    }
    savemat(pth, mat)


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


def plot_roi_tracks(
    images: Iterable[Union[str, Path, Image.Image]],
    rois: Dict,
    affines: Dict,
    dir_out: Union[str, Path],
):
    """Plot tracks of one or more ROIs onto images"""
    dir_out = ensure_dir(Path(dir_out))
    for i, img in enumerate(images):
        if isinstance(img, Image.Image):
            imname = f"{i}.tiff"
        else:
            imname = Path(img).name
            img = Image.open(img)
        for k, roi in rois.items():
            A = np.linalg.inv(read_affine(affines[k][i]))
            img = plot_roi(img, *transformed_roi(roi, A))
        img.save(dir_out / imname)


def read_roi_tracks(p):
    """Read ROI tracks .csv with ROI centroid positions as Numpy arrays"""
    tab = pd.read_csv(p, index_col=0)
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
    **kwargs,
):
    """Track an ROI through a list of frames

    kwargs are passed directly to `register()`.

    """
    archive = Path(archive)
    workdir = Path(workdir)
    size = get_frame_size(archive)
    mask = make_mask(roi_pts, size)
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
    p_log = workdir / f"{id_}_-_commands.log"
    # Run the registrations
    p_ref = dir_images / frames[0]

    def process_frame(frame, logf, affine):
        p_def = dir_images / frame
        p_affine, cmd_out = register(
            p_ref, p_def, p_mask, dir_affines, cmd, affine, **kwargs
        )
        logf.write(" ".join([shlex.quote(c) for c in cmd_out]) + "\n")
        affine = np.linalg.inv(read_affine(p_affine))
        vertices, center = transformed_roi(roi_pts, affine=affine)
        img = plot_roi(p_def, vertices, center)
        img.save(dir_tracks / frame)
        info = {
            "Name": frame,
            "Image": str((dir_images / frame).relative_to(workdir)),
            "Affine": str(Path(p_affine).relative_to(workdir)),
            "ROI centroid": center,
        }
        return info

    # Add reference frame entry to frame info table
    affine = np.eye(3)
    fixed_name = str(Path(frames[0]).with_suffix("").name)
    p_affine = dir_affines / f"{fixed_name}_to_{fixed_name}_0GenericAffine.mat"
    write_affine(affine, p_affine)
    verts, center = transformed_roi(roi_pts, affine)
    info = [
        {
            "Name": frames[0],
            "Image": str((dir_images / frames[0]).relative_to(workdir)),
            "Affine": str(p_affine.relative_to(workdir)),
            "ROI centroid": center,
        }
    ]
    # Add all registration data to frame info table
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
    exclusion_mask: Optional[Union[str, Path]] = None,
    nproc: Optional[int] = None,
    **kwargs,
):
    """Track multiple ROIs through a list of frames

    kwargs are passed directly to register().

    """
    archive = Path(archive)
    # Extract images from zip archive once up-front, if necessary
    if archive.suffix == ".zip":
        dir_images = clean_dir(workdir / f"{sid}_-_images")
        with ZipFile(archive) as a:
            for frame in frames:
                with open(dir_images / frame, "wb") as f:
                    f.write(a.read(frame))
    else:
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
            exclusion_mask,
            **kwargs,
        )
        info = info.rename({"ROI centroid": f"{nm} centroid"}, axis=1)
        info = info.rename({"Affine": f"{nm} affine"}, axis=1)
        return info

    if nproc is None:
        tracks = list(map(process_roi, rois.items()))
    else:
        with ProcessPool(nproc) as pool:
            tracks = pool.map(process_roi, rois.items())
    # Tabulate all ROI tracks together
    all_tracks = tracks[0].set_index("Name")
    for t in tracks[1:]:
        col = [
            c for c in t.columns if (c.endswith("centroid") or c.endswith("affine"))
        ]
        all_tracks = all_tracks.join(t.set_index("Name")[col])
    all_tracks.reset_index(inplace=True)
    # Plot all ROI tracks together
    affines = {
        c.removesuffix(" affine"): [workdir / p for p in all_tracks[c].tolist()]
        for c in all_tracks.columns
        if c.endswith("affine")
    }
    dir_tracks = clean_dir(workdir / f"{sid}_-_all_ROIs_tracks")
    plot_roi_tracks([dir_images / frame for frame in frames], rois, affines, dir_tracks)
    return all_tracks


def transformed_roi(boundary: Iterable[Tuple[Number, Number]], affine=None):
    if affine is None:
        affine = np.eye(3)
    pts = [(affine @ [x, y, 1])[:2] for x, y in boundary]
    center = np.mean(pts, axis=0)
    return pts, center
