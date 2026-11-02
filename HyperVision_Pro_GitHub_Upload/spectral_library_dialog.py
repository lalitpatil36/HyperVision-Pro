# -*- coding: utf-8 -*-
from spectral_library import (
    SpectralLibrary,
    SpectralLibraryEntry,
    SpectralLibraryDialog,
)

def get_spectral_library(name="HyperVision Master Library"):
    return SpectralLibrary(name)
