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

def load_brain(fname: str, type_='Z') -> nib.Nifti1Image:
    return nib.load(
            os.path.join(
                BASEDIR, type_, fname + '.nii'
        )
    )

def load_missing_slices(fname: str, dataset: Literal['ADNI','NACC']) -> list:
    with open(os.path.join(
            BASEDIR, 'slice_list', dataset, fname + '.txt'
            ), 'r', encoding='utf-8') as file_:
        return eval(file_.read())

def test_missing_slice_values(fname: str) -> None:
    missing_slices = load_missing_slices(fname, 'ADNI')
    brain = load_brain(fname, type_='M')
    data = np.asarray(brain.get_fdata())
    assert np.allclose(data[missing_slices[0],:,:],0)

def missing_slice_values(fname: str, type_='Z')-> tuple[np.ndarray, list]:
    missing_slices = load_missing_slices(fname, 'ADNI')
    brain = load_brain(fname, type_=type_)
    data = np.asarray(brain.get_fdata())
    return data, missing_slices

def bool_mask(fname: str) -> tuple[np.ndarray, np.ndarray]:
    data, missing_slices = missing_slice_values(fname)
    data_mask = np.full(data.shape, True)
    data_mask[missing_slices, :, :] = False
    return data, data_mask, missing_slices

def interp_generator(fname: str) -> tuple[RegularGridInterpolator, list, np.ndarray]:
    data, missing_slices = missing_slice_values(fname)
    shape = data.shape
    y = np.arange(0,shape[1], 1)
    z = np.arange(0, shape[2], 1)
    x = np.setdiff1d(
        np.arange(0,shape[0], 1), missing_slices
    )
    xg, yg, zg = np.meshgrid(x,y,z, indexing='ij')
    interp = RegularGridInterpolator(
        (x, y, z),
        data[xg, yg, zg],
        bounds_error=False,
        fill_value=0)
    return interp, missing_slices, data

def interp_slices(fname: str) -> np.ndarray:
    interp, _, data = interp_generator(fname)
    shape = data.shape
    y = np.arange(0, shape[1], 1)
    z = np.arange(0, shape[2], 1)
    x = np.arange(0, shape[0], 1)
    xg, yg, zg = np.meshgrid(x,y,z, indexing='ij')
    return interp((xg, yg, zg))

def interpolate_and_save_brain(fname: str) -> None:
    brain = load_brain(fname)
    interpolated_brain = interp_slices(fname)
    affine = brain.affine
    img = nib.Nifti1Image(interpolated_brain, affine)
    os.makedirs(
        os.path.join(BASEDIR, 'linear_interpolation'), exist_ok=True
    )
    nib.save(img, os.path.join(BASEDIR, 'linear_interpolation', fname + '.nii'))

def interp_all() -> None:
    orig_dir = os.path.join(BASEDIR, 'Z')
    for fi in tqdm(os.listdir(orig_dir)):
        interpolate_and_save_brain(fi[:-4])