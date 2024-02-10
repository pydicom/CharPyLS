
from io import BytesIO
import math
import os
from typing import Union, Any, BinaryIO, Dict, Tuple

import numpy as np

import _CharLS


# Old interface
def read(fname: Union[str, os.PathLike]) -> np.ndarray:
    """Read image data from JPEG-LS file."""
    with open(fname, "rb") as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8)

    return _CharLS.decode(arr)


def write(fname: Union[str, os.PathLike], data_image: np.ndarray) -> None:
    """Write compressed image data to JPEG-LS file."""
    buffer = encode_array(data_image)
    with open(fname, "wb") as f:
        f.write(buffer)


def encode(
    data_image: np.ndarray,
    lossy_error: int = 0,
    interleave_mode: Union[int, None] = None
) -> np.ndarray:
    """Encode grey-scale image via JPEG-LS using CharLS implementation."""
    if data_image.dtype == np.uint16 and np.max(data_image) <= 255:
        data_image = data_image.astype(np.uint8)

    buffer = encode_array(data_image, lossy_error, interleave_mode)
    return np.frombuffer(buffer, dtype="u1")


def decode(data_buffer: np.ndarray) -> np.ndarray:
    """Decode grey-scale image via JPEG-LS using CharLS implementation."""
    b = BytesIO(data_buffer.tobytes())
    return jlsread(b)


# New interface - encoding functions
def jlswrite(
    arr: np.ndarray,
    dst: Union[str, os.PathLike, BinaryIO],
    *,
    lossy_error: int = 0,
    interleave_mode: Union[int, None] = None,
) -> None:
    """JPEG-LS compress the image data in `arr` and write it to `dst`.

    Parameters
    ----------
    arr : numpy.ndarray
        An ndarray containing the image data.
    dst : str, PathLike[str], BinaryIO
        The destination where the compressed image data will be written, one of:

        * :class:`str` | :class:`os.PathLike`: the file path to write to, or
        * file-like: a `file-like object
          <https://docs.python.org/3/glossary.html#term-file-object>`_ in
          'wb' mode.
    lossy_error : int, optional
        The absolute value of the allowable error when encoding using
        near-lossless, default ``0`` (lossless). For example, if using 8-bit
        pixel data then the allowable error for a lossy image may be in the
        range (1, 255).
    interleave_mode : int, optional
        Required for multi-sample (i.e. non-greyscale) image data, the
        interleaving mode of `arr`. One of:

        * ``0``: `arr` is ordered R1R2...RnG1G2...GnB1B2...Bn, otherwise known
          as colour-by-plane
        * ``1``: `arr` is ordered R1...RwG1...GwB1...BwRw+1...
          where w is the width of the image, otherwise known as colour-by-line
        * ``2``: `arr` is ordered R1G1B1R2G2B2...RnGnBn, otherwise known as
          colour-by-pixel

        Having multi-sample pixel data ordered to match ``interleave_mode=0``
        should result in the greatest compression ratio, however most
        applications expect the pixel order to be ``interleave_mode=2``.
    """
    buffer = encode_array(arr, lossy_error, interleave_mode)
    if hasattr(dst, "read"):
        dst.write(buffer)
    else:
        with open(dst, "wb") as f:
            f.write(buffer)


def encode_array(
    arr: np.ndarray,
    lossy_error: int = 0,
    interleave_mode: Union[int, None] = None,
) -> bytearray:
    """Return the image data in `arr` as a JPEG-LS encoded bytearray.

    Parameters
    ----------
    arr : numpy.ndarray
        An ndarray containing the image data.
    lossy_error : int, optional
        The absolute value of the allowable error when encoding using
        near-lossless, default ``0`` (lossless). For example, if using 8-bit
        pixel data then the allowable error for a lossy image may be in the
        range (1, 255).
    interleave_mode : int, optional
        Required for multi-sample (i.e. non-greyscale) image data, the
        interleaving mode of `arr`. One of:

        * ``0``: `arr` is ordered R1R2...RnG1G2...GnB1B2...Bn, otherwise known
          as colour-by-plane
        * ``1``: `arr` is ordered R1...RwG1...GwB1...BwRw+1...
          where w is the width of the image, otherwise known as colour-by-line
        * ``2``: `arr` is ordered R1G1B1R2G2B2...RnGnBn, otherwise known as
          colour-by-pixel

        Having multi-sample pixel data ordered to match ``interleave_mode=0``
        should result in the greatest compression ratio, however most
        applications expect the pixel order to be ``interleave_mode=2``.

    Returns
    -------
    bytearray
        The encoded JPEG-LS codestream.
    """
    if arr.dtype == np.uint8:
        bytes_per_pixel = 1
    elif arr.dtype == np.uint16:
        bytes_per_pixel = 2
    else:
        raise ValueError(
            f"Invalid ndarray dtype '{arr.dtype}', expecting np.uint8 or np.uint16."
        )

    nr_dims = len(arr.shape)
    if nr_dims not in (2, 3):
        raise ValueError("Invalid shape - image data must be 2D or 3D")

    if nr_dims == 2:
        # Greyscale images should always be interleave mode 0
        interleave_mode = 0
        rows = arr.shape[0]
        columns = arr.shape[1]
        samples_per_pixel = 1
    else:
        # Multi-component images may be interleave mode 0, 1 or 2
        if arr.shape[-1] in (3, 4):
            # Colour-by-pixel (R, C, 3) or (R, C, 4)
            # Mode 1 and 2 are identical apparently
            interleave_mode = 2 if interleave_mode is None else interleave_mode
        elif arr.shape[0] in (3, 4):
            # Colour-by-plane (3, R, C) or (4, R, C)
            interleave_mode = 0 if interleave_mode is None else interleave_mode
        elif interleave_mode is None:
            raise ValueError(
                "Unable to automatically determine an appropriate 'interleave_mode' "
                "value, please set it manually"
            )

        if interleave_mode == 0:
            samples_per_pixel = arr.shape[0]
            rows = arr.shape[1]
            columns = arr.shape[2]
        else:
            rows = arr.shape[0]
            columns = arr.shape[1]
            samples_per_pixel = arr.shape[2]

    return _CharLS._encode(
        arr.tobytes(),  # needs data in the correct interleave mode (not a view)
        lossy_error,
        interleave_mode,
        rows,
        columns,
        samples_per_pixel,
        math.ceil(math.log(arr.max() + 1, 2)),
    )


def encode_buffer(
    src: bytes,
    rows: int,
    columns: int,
    samples_per_pixel: int,
    bits_stored: int,
    lossy_error: int = 0,
    interleave_mode: Union[int, None] = None,
    **kwargs: Any,
) -> bytearray:
    """Return the image data in `src` as a JPEG-LS encoded bytearray.

    Parameters
    ----------
    src : bytes
        The image data to be JPEG-LS encoded.
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
    lossy_error : int, optional
        The absolute value of the allowable error when encoding using
        near-lossless, default ``0`` (lossless). For example, if using 8-bit
        pixel data then the allowable error for a lossy image may be in the
        range (1, 255).
    interleave_mode : int, optional
        Required for multi-sample (i.e. non-greyscale) image data, the
        interleaving mode of `src`. One of:

        * ``0``: the pixels in `src` are ordered R1R2...RnG1G2...GnB1B2...Bn,
          otherwise known as colour-by-plane
        * ``1``: the pixels in `src` are ordered R1...RwG1...GwB1...BwRw+1...
          where w is the width of the image, otherwise known as colour-by-line
        * ``2``: the pixels in `src` are ordered R1G1B1R2G2B2...RnGnBn,
          otherwise known as colour-by-pixel

        Having multi-sample pixel data ordered to match ``interleave_mode=0``
        should result in the greatest compression ratio, however most
        applications expect the pixel order to be ``interleave_mode=2``.

    Returns
    -------
    bytearray
        The encoded JPEG-LS codestream.
    """
    if samples_per_pixel < 1:
        raise ValueError(
            f"Invalid 'samples_per_pixel' value {samples_per_pixel}: must be "
            "greater than 1"
        )

    if not 0 < bits_stored < 17:
        raise ValueError(
            f"Invalid 'bits_stored' value {bits_stored}: must be in the range (1, 16)"
        )

    if samples_per_pixel == 1:
        interleave_mode = 0
    elif interleave_mode not in (0, 1, 2):
        # Multi-sample must have the interleave mode specified
        raise ValueError(
            f"Invalid 'interleave_mode' value {interleave_mode}: must be 0, 1 or 2"
        )

    max_intensity = 2**bits_stored - 1
    if not 0 <= lossy_error <= max_intensity:
        raise ValueError(
            f"Invalid 'lossy_error' value {lossy_error}: must be in the "
            f"range (0, {max_intensity}) for {bits_stored}-bit pixel data"
        )

    if not 1 <= rows <= 65535:
        raise ValueError(
            f"Invalid 'rows' value {rows}: must be in the range (1, 65535)"
        )

    if not 1 <= columns <= 65535:
        raise ValueError(
            f"Invalid 'columns' value {columns}: must be in the range (1, 65535)"
        )

    bytes_per_pixel = math.ceil(bits_stored / 8)

    length_src = len(src)
    length_expected = rows * columns * samples_per_pixel * bytes_per_pixel
    if length_expected != length_src:
        raise ValueError(
            f"The 'src' length of {length_src} bytes does not match the expected "
            "length determined from 'rows', 'columns', 'samples_per_pixel' and "
            "'bits_stored'"
        )

    return _CharLS._encode(
        src,
        lossy_error,
        interleave_mode,
        rows,
        columns,
        samples_per_pixel,
        bits_stored,
    )


def encode_pixel_data(src: bytes, lossy_error: int = 0, **kwargs: Any) -> bytearray:
    """Return JPEG-LS encoded pixel data.

    .. note::

        This function is intended for use with pydicom. If you want to compress
        raw encoded image data then use ``encode_buffer`` instead.

    Parameters
    ----------
    src : bytes
        The image data to be JPEG-LS encoded.
    lossy_error : int, optional
        The absolute value of the allowable error when encoding using
        near-lossless, default ``0`` (lossless). For example, if using 8-bit
        pixel data then the allowable error for a lossy image may be in the
        range (1, 255).
    **kwargs
        Required and optional keyword parameters, at a minimum must include:

        * `rows`
        * `columns`
        * `samples_per_pixel`
        * `bits_stored`

        If `samples_per_pixel` > 1 then also requires:

        * `planar_configuration`

    Returns
    -------
    bytearray
        The encoded JPEG-LS codestream.
    """
    interleave_mode = 0
    samples_per_pixel = kwargs.get("samples_per_pixel")
    if samples_per_pixel > 1:
        planar_configuration = kwargs.get("planar_configuration")
        interleave_mode = 0 if planar_configuration else 2

    return encode_buffer(
        src,
        kwargs.get("rows"),
        kwargs.get("columns"),
        samples_per_pixel,
        kwargs.get("bits_stored"),
        lossy_error,
        interleave_mode,
    )


# New interface - decoding functions
JLSSourceType = Union[str, os.PathLike, BinaryIO, bytes, bytearray]


def jlsread(src: JLSSourceType) -> np.ndarray:
    """Return the JPEG-LS codestream in `src` as an ndarray.

    Parameters
    ----------
    src : str, PathLike[str], BinaryIO, buffer-like
        The source of compressed image data, one of:

        * :class:`str` | :class:`os.PathLike`: the file path to be read, or
        * file-like: a `file-like object
          <https://docs.python.org/3/glossary.html#term-file-object>`_ in
          'rb' mode.
        * buffer-like: a :class:`bytes` or :class:`bytearray` containing the
          compressed JPEG-LS codestream.

    Returns
    -------
    numpy.ndarray
        The decoded image.
    """
    if isinstance(src, (bytes, bytearray)):
        buffer = src
    elif hasattr(src, "read"):
        buffer = src.read()
    else:
        with open(src, "rb") as f:
            buffer = f.read()

    im, info = decode_buffer(buffer)

    bytes_per_pixel = math.ceil(info["bits_per_sample"] / 8)
    arr = np.frombuffer(im, dtype=f"u{bytes_per_pixel}")
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


def decode_buffer(src: Union[bytes, bytearray]) -> Tuple[bytearray, Dict[str, int]]:
    """Decode the JPEG-LS codestream `src` to a bytearray

    Parameters
    ----------
    src : bytes | bytearray
        The JPEG-LS codestream to be decoded.

    Returns
    -------
    tuple[bytearray, dict[str, int]]
        The decoded (image data, image metadata).
    """
    return _CharLS._decode(src), _CharLS.read_header(src)


def decode_pixel_data(src: Union[bytes, bytearray], **kwargs: Any) -> Tuple[bytearray, Dict[str, int]]:
    """Decode the JPEG-LS codestream `src` to a bytearray

    .. note::

        This function is intended to be used with image data from DICOM
        datasets. If you are decoding a stand-alone JPEG-LS codestream then
        it's recommended you use ``decode_buffer`` instead as this will return
        the decoded image data as well as the image metadata necessary for
        interpreting the image.

    Parameters
    ----------
    src : bytes | bytearray
        The JPEG-LS codestream to be decoded.

    Returns
    -------
    tuple[bytearray, dict[str, int]]
        The decoded (image data, image metadata).
    """
    return decode_buffer(src)
