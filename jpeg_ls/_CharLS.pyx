# cython: language_level=3

import logging
import math
from typing import Dict
import sys

import numpy as np
cimport numpy as cnp


LOGGER = logging.getLogger("jpeg_ls._CharLS")


cdef extern from "define_charls_dll.h":
    pass


cdef extern from "charls/public_types.h":
    cdef enum JLS_ERROR "charls::jpegls_errc":
        pass

    cdef enum c_interleave_mode "charls::interleave_mode":
        CHARLS_INTERLEAVE_MODE_NONE = 0
        CHARLS_INTERLEAVE_MODE_LINE = 1
        CHARLS_INTERLEAVE_MODE_SAMPLE = 2

    cdef enum color_transformation "charls::color_transformation":
        CHARLS_COLOR_TRANSFORMATION_NONE = 0
        CHARLS_COLOR_TRANSFORMATION_HP1 = 1
        CHARLS_COLOR_TRANSFORMATION_HP2 = 2
        CHARLS_COLOR_TRANSFORMATION_HP3 = 3

    cdef struct JpegLSPresetCodingParameters:
        long MaximumSampleValue
        long Threshold1
        long Threshold2
        long Threshold3
        long ResetValue

    cdef struct JfifParameters:
        long version
        long units
        long Xdensity
        long Ydensity
        long Xthumbnail
        long Ythumbnail
        void* thumbnail

    cdef struct JlsParameters:
        # Width in pixels (number of samples per line)
        long width
        # Height in pixels (number of lines)
        long height
        # Bits per sample (sample precision (2, 16))
        long bitsPerSample
        # Number of bytes from one row of pixels to the next in memory
        long stride
        # Number of components, 1 for monochrome, 3 for RGB (1, 255)
        long components
        # The allowed lossy error, 0 for lossless
        long allowedLossyError
        # The order of color components in the compressed stream
        c_interleave_mode interleaveMode
        color_transformation colorTransformation
        char outputBgr
        JpegLSPresetCodingParameters custom
        JfifParameters jfif


cdef extern from "charls/charls_jpegls_decoder.h":
    cdef JLS_ERROR JpegLsReadHeader(
        void* source,
        size_t source_length,
        JlsParameters* params,
        char* error_message
    )

    cdef JLS_ERROR JpegLsDecode(
        void* dst,
        size_t dst_length,
        void * src,
        size_t src_length,
        JlsParameters* info,
        char* error_message
    )


cdef extern from "charls/charls_jpegls_encoder.h":
    cdef JLS_ERROR JpegLsEncode(
        void* dst,
        size_t dst_length,
        size_t* bytes_written,
        void* src,
        size_t src_length,
        JlsParameters* info,
        char* error_message
    )


cdef JlsParameters build_parameters():

    cdef JpegLSPresetCodingParameters info_custom
    info_custom.MaximumSampleValue = 0
    info_custom.Threshold1 = 0
    info_custom.Threshold2 = 0
    info_custom.Threshold3 = 0
    info_custom.ResetValue = 0

    cdef JfifParameters jfif
    jfif.version = 0
    jfif.units = 0
    jfif.Xdensity = 0
    jfif.Ydensity = 0
    jfif.Xthumbnail = 0
    jfif.Ythumbnail = 0

    cdef JlsParameters info

    info.width = 0
    info.height = 0
    info.bitsPerSample = 0
    info.stride = 0
    # number of components (RGB = 3, RGBA = 4, monochrome = 1;
    info.components = 1
    #  0 means lossless
    info.allowedLossyError = 0
    # For monochrome images, always use ILV_NONE.
    info.interleaveMode = <c_interleave_mode> 0
    # 0 means no color transform
    info.colorTransformation = <color_transformation> 0
    info.outputBgr = 0  # when set to true, CharLS will reverse the normal RGB
    info.custom = info_custom
    info.jfif = jfif

    return info


cdef JlsParameters _read_header(bytes buffer):
    """Decode grey-scale image via JPEG-LS using CharLS implementation.

    Returns
    -------
    JlsParameters
        The JPEG-LS stream header information.
    """
    # Size of input image, point to buffer.
    cdef int size_buffer = len(buffer)

    # Setup parameter structure.
    cdef JlsParameters info = build_parameters()

    # Pointers to input and output data.
    cdef char* data_buffer_ptr = <char*>buffer
    cdef JlsParameters* info_ptr = &info

    # Error strings are defined in jpegls_error.cpp
    # As of v2.4.2 the longest string is ~114 chars, so give it a 256 buffer
    err_msg = bytearray(b"\x00" * 256)
    cdef char *error_message = <char*>err_msg

    # Read the header.
    cdef JLS_ERROR err
    err = JpegLsReadHeader(
        data_buffer_ptr,
        size_buffer,
        info_ptr,
        error_message
    )

    if <int> err != 0:
        error = err_msg.decode("ascii").strip("\0")
        raise RuntimeError(f"Decoding error: {error}")

    return info


def read_header(src: bytes | bytearray) -> Dict[str, int]:
    """Return a dict containing information about the JPEG-LS file."""
    # info: JlsParameters
    info = _read_header(bytes(src))
    return {
        "width": info.width,
        "height": info.height,
        "bits_per_sample": info.bitsPerSample,
        "stride": info.stride,
        "components": info.components,
        "allowed_lossy_error": info.allowedLossyError,
        "interleave_mode": info.interleaveMode,
        "colour_transformation": info.colorTransformation,
    }


def _decode(src: bytes | bytearray) -> bytearray:
    """Decode the JPEG-LS codestream `src` to a bytearray

    Parameters
    ----------
    src : bytes | bytearray
        The JPEG-LS codestream to be decoded.

    Returns
    -------
    bytearray
        The decoded little-endian ordered image data.
    """
    if isinstance(src, bytearray):
        src = bytes(src)

    info = _read_header(src)

    bytes_per_pixel = math.ceil(info.bitsPerSample / 8)
    dst_length = info.width * info.height * info.components * bytes_per_pixel
    dst = bytearray(b"\x00" * dst_length)

    # Error strings are defined in jpegls_error.cpp
    # As of v2.4.2 the longest string is ~114 chars, so give it a 256 buffer
    error_message = bytearray(b"\x00" * 256)

    # Decode compressed data.
    cdef JLS_ERROR err
    err = JpegLsDecode(
        <char *>dst,
        dst_length,
        <char *>src,
        len(src),
        &info,
        <char *>error_message
    )

    if <int> err != 0:
        msg = error_message.decode("ascii").strip("\0")
        raise RuntimeError(f"Decoding error: {msg}")

    # Need to byte-swap the output on big endian systems
    if sys.byteorder == "big" and bytes_per_pixel > 1:
        dst[::2], dst[1::2] = dst[1::2], dst[::2]

    return dst


def decode_from_buffer(src: bytes | bytearray) -> bytearray:
    """Decode the JPEG-LS codestream `src` to a bytearray

    Parameters
    ----------
    src : bytes | bytearray
        The JPEG-LS codestream to be decoded.

    Returns
    -------
    bytearray
        The decoded little-endian ordered image data.
    """
    return _decode(src)


def decode(cnp.ndarray[cnp.uint8_t, ndim=1] data_buffer):
    """Decode the JPEG-LS codestream in the ndarray `data_buffer`

    Parameters
    ----------
    data_buffer : numpy.ndarray
        The JPEG-LS codestream to be decoded as 1 dimensional ndarray of uint8.

    Returns
    -------
    numpy.ndarray
        The decoded image.
    """
    src = data_buffer.tobytes()

    info = read_header(src)
    bytes_per_pixel = math.ceil(info["bits_per_sample"] / 8)
    arr = np.frombuffer(_decode(src), dtype=f"<u{bytes_per_pixel}")
    rows = info["height"]
    columns = info["width"]
    samples_per_pixel = info["components"]

    if info["components"] == 3:
        if info["interleave_mode"] == 0:
            # ILV 0 is colour-by-plane, needs to be reshaped then transposed
            #   to colour-by-pixel instead
            arr = arr.reshape((samples_per_pixel, rows, columns))
            return arr.transpose(1, 2, 0)

        # Colour-by-pixel, just needs to be reshaped
        return arr.reshape((rows, columns, samples_per_pixel))

    return arr.reshape((rows, columns))


def _encode(
    src: bytes,
    lossy_error: int,
    interleave_mode: int,
    rows: int,
    columns: int,
    samples_per_pixel: int,
    bits_stored: int,
) -> bytearray:
    """Return the image data in `src` as a JPEG-LS encoded bytearray.

    Parameters
    ----------
    src : bytes
        The image data to be JPEG-LS encoded.
    lossy_error : int
        The absolute value of the allowable error when encoding using
        near-lossless, default ``0`` (lossless). For example, if using 8-bit
        pixel data then the allowable error for a lossy image may be in the
        range (1, 255).
    interleave_mode : int
        Required for multi-sample (i.e. non-greyscale) image data, the
        interleaving mode of `src`. One of:

        * ``0``: the pixels in `src` are ordered R1R2...RnG1G2...GnB1B2...Bn,
          otherwise known as colour-by-plane
        * ``1``: the pixels in `src` are ordered R1...RwG1...GwB1...BwRw+1...
          where w is the width of the image, otherwise known as colour-by-line
        * ``2``: the pixels in `src` are ordered R1G1B1R2G2B2...RnGnBn,
          otherwise known as colour-by-pixel
    rows : int
        The number of rows of pixels in the image.
    columns : int
        The number of columns of pixels in the image.
    samples_per_pixel : int
        The number of samples per pixel in the image, otherwise knows as the
        number of components or channels. A greyscale image has 1 sample per
        pixel while an RGB image will have 3.
    bits_stored : int
        The bit-depth per pixel, must be in the range (1, 16).

    Returns
    -------
    bytearray
        The encoded JPEG-LS codestream.
    """
    cdef JlsParameters info = build_parameters()
    info.height = rows
    info.width = columns
    info.components = samples_per_pixel
    info.interleaveMode = <c_interleave_mode><int>interleave_mode
    info.allowedLossyError = lossy_error

    info.stride = info.width * math.ceil(bits_stored / 8)
    if interleave_mode != 0:
        info.stride = info.stride * info.components

    info.bitsPerSample = 2 if bits_stored < 2 else bits_stored

    LOGGER.debug(
        "Encoding paramers are:\n"
        f"\tWidth: {info.width} px\n"
        f"\tHeight: {info.height} px\n"
        f"\tComponents: {info.components}\n"
        f"\tBits per sample: {info.bitsPerSample}\n"
        f"\tStride: {info.stride} bytes\n"
        f"\tInterleave mode: {<int>info.interleaveMode}\n"
        f"\tAllowed lossy error: {info.allowedLossyError}\n"
    )

    # Destination for the compressed data - start out twice the length of raw
    src_length = len(src)
    dst = bytearray(b"\x00" * src_length * 2)

    # Number of bytes of compressed data
    cdef size_t compressed_length = 0

    # Error strings are defined in jpegls_error.cpp
    # As of v2.4.2 the longest string is ~114 chars, so give it a 256 buffer
    error_message = bytearray(b"\x00" * 256)

    # Need to byte-swap the input on big endian systems
    if sys.byteorder == "big" and bits_stored > 8:
        b = bytearray(src)
        b[::2], b[1::2] = b[1::2], b[::2]
        src = bytes(b)

    cdef JLS_ERROR err
    err = JpegLsEncode(
        <char *>dst,
        len(dst),
        &compressed_length,
        <char *>src,
        src_length,
        &info,
        <char *>error_message
    )

    if <int> err != 0:
        msg = error_message.decode("ascii").strip("\0")
        raise RuntimeError(f"Encoding error: {msg}")

    return dst[:compressed_length]
