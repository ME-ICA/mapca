"""Tests for mapca configuration."""

import json
import os
from urllib.request import urlopen, urlretrieve

import pytest


def fetch_file(osf_id, path, filename):
    """
    Fetch file located on OSF and downloads to `path`/`filename`1.

    Parameters
    ----------
    osf_id : str
        Unique OSF ID for file to be downloaded. Will be inserted into relevant
        location in URL: https://osf.io/{osf_id}/download
    path : str
        Path to which `filename` should be downloaded. Ideally a temporary
        directory
    filename : str
        Name of file to be downloaded (does not necessarily have to match name
        of file on OSF)

    Returns
    -------
    full_path : str
        Full path to downloaded `filename`
    """
    # Use OSF API v2 to get the download URL
    api_url = f"https://api.osf.io/v2/files/{osf_id}/"
    full_path = os.path.join(path, filename)

    if not os.path.isfile(full_path):
        # Fetch metadata to get download link
        with urlopen(api_url) as response:
            metadata = json.load(response)
            download_url = metadata["data"]["links"]["download"]

        # Download the actual file
        urlretrieve(download_url, full_path)

    return full_path


@pytest.fixture(scope="session")
def testpath(tmp_path_factory):
    """Test path that will be used to download all files."""
    return tmp_path_factory.getbasetemp()


@pytest.fixture
def test_img(testpath):
    """Fetch data file."""
    return fetch_file("jw43h", testpath, "data.nii.gz")


@pytest.fixture
def test_mask(testpath):
    """Fetch mask file."""
    return fetch_file("9u2m5", testpath, "mask.nii.gz")


@pytest.fixture
def test_ts(testpath):
    """Fetch comp_ts file."""
    return fetch_file("gz2hb", testpath, "comp_ts.npy")


@pytest.fixture
def test_varex(testpath):
    """Fetch varex file."""
    return fetch_file("7xj5k", testpath, "varex.npy")


@pytest.fixture
def test_varex_norm(testpath):
    """Fetch varex_norm file."""
    return fetch_file("jrd9c", testpath, "varex_norm.npy")


@pytest.fixture
def test_weights(testpath):
    """Fetch weights file."""
    return fetch_file("t94m8", testpath, "voxel_comp_weights.npy")
