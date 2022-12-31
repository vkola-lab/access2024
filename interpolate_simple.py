"""
Code for interpolating missing slices from brains
"""

import os
from typing import Literal

import nibabel as nib
import numpy as np
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator

BASEDIR = "/Users/mromano/research/data/rgan_data/"


def load_brain(fname: str, type_="Z") -> nib.Nifti1Image:
    """
    Given a filename and the type of folder, load nibabel image

    Args:
        fname (str): filename without extension, i.e. ".nii"
        type_ (str, optional): Type of brain to load. Defaults to "Z".

    Returns:
        nib.Nifti1Image: brain image
    """
    return nib.load(os.path.join(BASEDIR, type_, fname + ".nii"))


def load_missing_slices(fname: str, dataset: Literal["ADNI", "NACC"]) -> list:
    """
    Load missing slices file corresponding to a given filename. Available for ADNI and NACC

    Args:
        fname (str): filename without extension
        dataset (Literal[ADNI, NACC]): dataset from which we load our images

    Returns:
        list: list of slices from the x / first dimension to remove
    """
    with open(
        os.path.join(BASEDIR, "slice_list", dataset, fname + ".txt"),
        "r",
        encoding="utf-8",
    ) as file_:
        return eval(file_.read())


def test_missing_slice_values(fname: str, ds_="ADNI") -> None:
    """
    Tests that each of the missing slices corresponds to a slice with values 0\
        from our masked image

    Args:
        fname (str): filename to test; assumes ADNI dataset
    """
    missing_slices = load_missing_slices(fname, ds_)
    rel_path = "Z" if ds_ == "ADNI" else "Z_E"
    brain = load_brain(fname, type_=rel_path)
    data = np.asarray(brain.get_fdata())
    assert np.allclose(data[missing_slices, :, :], 0)


def missing_slice_values(fname: str, type_: str = "Z") -> tuple[np.ndarray, list]:
    """
    Loads missing slices and corresponding brain, assuming slices come\
        from the ADNI dataset

    Args:
        fname (str): filename to load missing slices and brain from
        type_ (str, optional): type of brain to load. Defaults to "Z", \
            or non-interpolated brain

    Returns:
        tuple[np.ndarray, list]: returns the brain as ndarray and slices removed from the brain
    """
    ds_ = "NACC" if type_[-2:] == "_E" else "ADNI"
    missing_slices = load_missing_slices(fname, ds_)
    brain = load_brain(fname, type_=type_)
    data = np.asarray(brain.get_fdata())
    return data, missing_slices


def bool_mask(fname: str) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Masks a brain and returns the original brain, boolean mask, and slices\
        removed from the brain

    Args:
        fname (str): file to load as slices/original brain/mask

    Returns:
        tuple[np.ndarray, np.ndarray, list]: brain, masked_brain, and slices to use for mask
    """
    data, missing_slices = missing_slice_values(fname)
    data_mask = np.full(data.shape, True)
    data_mask[missing_slices, :, :] = False
    return data, data_mask, missing_slices


def interp_generator(
    fname: str, ds_: Literal["ADNI", "NACC"]
) -> tuple[RegularGridInterpolator, list, np.ndarray]:
    """
    Creates an interpolator given a filename. Loads the initial array and missing\
        slices, constructs the cartesian coordinates of each value that we have in\
        the array, and interpolates

    Args:
        fname (str): filename to use to load missing slices and brain

    Returns:
        tuple[RegularGridInterpolator, list, np.ndarray]: RegularGridInterpolator object,\
            list of missing slices, and the initial brain w/ slices removed
    """
    type_ = "Z" if ds_ == "ADNI" else "Z_E"
    data, missing_slices = missing_slice_values(fname, type_=type_)
    shape = data.shape
    y = np.arange(0, shape[1], 1)
    z = np.arange(0, shape[2], 1)
    x = np.setdiff1d(np.arange(0, shape[0], 1), missing_slices)
    interp = RegularGridInterpolator(
        (x, y, z), data[x, :, :], bounds_error=False, fill_value=0, method="linear"
    )
    return interp, missing_slices, data


def interp_slices(fname: str, ds_: Literal["ADNI", "NACC"], dev=False) -> np.ndarray:
    """
    Retrieves an interpolator object for the data missing from the fname brain,\
        and then re-interpolates the entire brain on the initial grid

    Args:
        fname (str): file to use to load the interpolator and brain with missing slices

    Returns:
        np.ndarray: interpolated brain
    """
    interp, _, data = interp_generator(fname, ds_)
    shape = data.shape
    y = np.arange(0, shape[1], 1).reshape(1, -1, 1)
    z = np.arange(0, shape[2], 1).reshape(1, 1, -1)
    x = np.arange(0, shape[0], 1).reshape(-1, 1, 1)
    output = interp((x, y, z))
    if dev:
        xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")
        assert np.array_equal(output, interp((xg, yg, zg)))
    return output


def interpolate_and_save_brain(
    fname: str, ds_: Literal["ADNI", "NACC"], dev: bool = False
) -> None:
    """
    Loads a brain from filename fname, interpolates the missing slices using\
        a linear interpolator, and saves the new, interpolated image using
        the same affine coordinates

    Args:
        fname (str): filendfame to use for loading brain and interpolated brain
    """
    rel_path = "Z_E" if ds_ == "NACC" else "Z"
    brain = load_brain(fname, type_=rel_path)
    interpolated_brain = interp_slices(fname, ds_=ds_, dev=dev)
    affine = brain.affine
    img = nib.Nifti1Image(interpolated_brain, affine)
    output_dir = os.path.join(BASEDIR, "linear_interpolation", fname + ".nii")
    dump_brain(img, output_dir)


def dump_brain(img: nib.Nifti1Image, output_dir: str) -> None:
    if os.path.exists(output_dir):
        img = nib.load(output_dir)

    nib.save(img, output_dir)


def interp_all(external: bool = False, dev: bool = True) -> None:
    """
    Iterates through the missing slice brain directory and generates linearly interpolated brains
    """
    rel_path = "Z_E" if external else "Z"
    ds_ = "NACC" if external else "ADNI"
    orig_dir = os.path.join(BASEDIR, rel_path)
    for file_ in tqdm(os.listdir(orig_dir)):
        if dev:
            test_missing_slice_values(file_[:-4], ds_=ds_)
        interpolate_and_save_brain(file_[:-4], ds_=ds_, dev=dev)


if __name__ == "__main__":
    interp_all(False, True)
