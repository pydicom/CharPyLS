
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import numpy as np

from jpeg_ls import (
    decode,
    encode,
    write,
    jlswrite,
    encode_array,
    encode_buffer,
    encode_pixel_data,
    jlsread,
    decode_buffer,
)


DATA = Path(__file__).parent / "jlsimV100"


@pytest.fixture
def TEST8():
    # 8-bit colour test image
    # p6, 256 x 256 x 3, uint8, RGB
    p = DATA / "TEST8.PPM"
    arr = np.fromfile(p, dtype="u1", offset=15)
    return arr.reshape((256, 256, 3))


@pytest.fixture
def TEST8R():
    # Red component of TEST8
    # p5, 256 x 256 x 1, 255, uint8, greyscale
    p = DATA / "TEST8R.PGM"
    arr = np.fromfile(p, dtype="u1", offset=15)
    return arr.reshape((256, 256))


@pytest.fixture
def TEST8G():
    # Green component of TEST8
    # p5, 256 x 256 x 1, 255, uint8, greyscale
    p = DATA / "TEST8G.PGM"
    arr = np.fromfile(p, dtype="u1", offset=15)
    return arr.reshape((256, 256))


@pytest.fixture
def TEST8B():
    # Blue component of TEST8
    # p5, 256 x 256 x 1, 255, uint8, greyscale
    p = DATA / "TEST8B.PGM"
    arr = np.fromfile(p, dtype="u1", offset=15)
    return arr.reshape((256, 256))


@pytest.fixture
def TEST8BS2():
    # Blue component of TEST8, subsampled 2X in horizontal and vertical
    # p5, 128 x 128 x 1, 255, uint8, greyscale
    p = DATA / "TEST8BS2.PGM"
    arr = np.fromfile(p, dtype="u1", offset=15)
    return arr.reshape((128, 128))


@pytest.fixture
def TEST8GR4():
    # Gren component of TEST8, subsampled 4X in the vertical
    # p5, 256 x 64 x 1, 255, uint8, greyscale
    p = DATA / "TEST8GR4.PGM"
    arr = np.fromfile(p, dtype="u1", offset=14)
    return arr.reshape((64, 256))


@pytest.fixture
def TEST16():
    # 12-bit greyscale, big-endian
    # p5, 256 x 256 x 1, 4095, uint16, greyscale
    p = DATA / "TEST16.PGM"
    arr = np.fromfile(p, dtype=">u2", offset=16)
    return arr.reshape((256, 256)).astype("<u2")


class TestEncodeArray:
    """Tests for encode_array()"""

    def test_invalid_dtype_raises(self):
        msg = "Invalid ndarray dtype 'float64', expecting np.uint8 or np.uint16"
        with pytest.raises(Exception, match=msg):
            encode(np.empty((2, 2), dtype=float))

    def test_invalid_nr_components_raises(self):
        msg = (
            "Encoding error: The component count argument is outside the range"
            "[1, 255]"
        )
        with pytest.raises(Exception, match=msg):
            encode(np.empty((3, 2, 256), dtype="u1"), interleave_mode=1)

    def test_invalid_shape_raises(self):
        msg = "Invalid shape - image data must be 2D or 3D"
        with pytest.raises(Exception, match=msg):
            encode(np.empty((2,), dtype="u1"))

        with pytest.raises(Exception, match=msg):
            encode(np.empty((2, 2, 2, 2), dtype="u1"))

    def test_TEST8_ILV0(self, TEST8):
        # 3 component, ILV 0
        # Convert from colour-by-pixel to colour-by-plane
        arr = TEST8.transpose(2, 0, 1)
        assert arr.shape == (3, 256, 256)
        buffer = encode(arr, interleave_mode=0)
        assert isinstance(buffer, np.ndarray)
        arr = decode(buffer)
        assert np.array_equal(arr, TEST8)

    def test_TEST8_ILV1(self, TEST8):
        # 3 component, ILV 1
        assert TEST8.shape == (256, 256, 3)
        buffer = encode(TEST8, interleave_mode=1)
        assert isinstance(buffer, np.ndarray)
        arr = decode(buffer)
        assert np.array_equal(arr, TEST8)

    def test_TEST8_ILV2(self, TEST8):
        # 3 component, ILV 2
        assert TEST8.shape == (256, 256, 3)
        buffer = encode(TEST8, interleave_mode=2)
        assert isinstance(buffer, np.ndarray)
        arr = decode(buffer)
        assert np.array_equal(arr, TEST8)

    def test_TEST8R(self, TEST8R):
        # 1 component, ILV 0
        buffer = encode(TEST8R)
        assert isinstance(buffer, np.ndarray)
        assert buffer.shape[0] < TEST8R.shape[0] * TEST8R.shape[1]
        arr = decode(buffer)
        assert np.array_equal(arr, TEST8R)

    def test_TEST8G(self, TEST8G):
        buffer = encode(TEST8G)
        assert isinstance(buffer, np.ndarray)
        arr = decode(buffer)
        assert np.array_equal(arr, TEST8G)

    def test_TEST8B(self, TEST8B):
        buffer = encode(TEST8B)
        assert isinstance(buffer, np.ndarray)
        arr = decode(buffer)
        assert np.array_equal(arr, TEST8B)

    def test_TEST8BS2(self, TEST8BS2):
        buffer = encode(TEST8BS2)
        assert isinstance(buffer, np.ndarray)
        arr = decode(buffer)
        assert np.array_equal(arr, TEST8BS2)

    def test_TEST8GR4(self, TEST8GR4):
        buffer = encode(TEST8GR4)
        assert isinstance(buffer, np.ndarray)
        arr = decode(buffer)
        assert np.array_equal(arr, TEST8GR4)

    def test_TEST16(self, TEST16):
        buffer = encode(TEST16)
        assert isinstance(buffer, np.ndarray)
        arr = decode(buffer)
        assert np.array_equal(arr, TEST16)

    def test_TEST8_lossy(self, TEST8):
        buffer = encode_array(TEST8, lossy_error=5)
        assert isinstance(buffer, bytearray)
        data, info = decode_buffer(buffer)
        arr = np.frombuffer(data, dtype="u1")
        arr = arr.reshape(TEST8.shape)
        assert not np.array_equal(arr, TEST8)
        assert np.allclose(arr, TEST8, atol=5)

    def test_TEST8R_lossy(self, TEST8R):
        buffer = encode_array(TEST8R, lossy_error=3)
        assert isinstance(buffer, bytearray)
        data, info = decode_buffer(buffer)
        arr = np.frombuffer(data, dtype="u1")
        arr = arr.reshape(TEST8R.shape)
        assert not np.array_equal(arr, TEST8R)
        assert np.allclose(arr, TEST8R, atol=3)


class TestEncodeBuffer:
    """Tests for encode_buffer()"""

    def test_invalid_lossy_raises(self, TEST8):
        msg = (
            r"Invalid 'lossy_error' value -1: must be in the range \(0, 255\) "
            "for 8-bit pixel data"
        )
        with pytest.raises(ValueError, match=msg):
            encode_buffer(
                TEST8.tobytes(),
                rows=256,
                columns=256,
                samples_per_pixel=3,
                bits_stored=8,
                interleave_mode=0,
                lossy_error=-1,
            )

        msg = (
            r"Invalid 'lossy_error' value 256: must be in the range \(0, 255\) "
            "for 8-bit pixel data"
        )
        with pytest.raises(ValueError, match=msg):
            encode_buffer(
                TEST8.tobytes(),
                rows=256,
                columns=256,
                samples_per_pixel=3,
                bits_stored=8,
                interleave_mode=0,
                lossy_error=256,
            )

    def test_invalid_samples_per_pixel_raises(self, TEST8):
        msg = "Invalid 'samples_per_pixel' value 0: must be greater than 1"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(
                TEST8.tobytes(),
                rows=256,
                columns=256,
                samples_per_pixel=0,
                bits_stored=8,
                interleave_mode=0,
            )

    def test_invalid_bits_stored_raises(self, TEST8):
        msg = r"Invalid 'bits_stored' value 0: must be in the range \(1, 16\)"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(
                TEST8.tobytes(),
                rows=256,
                columns=256,
                samples_per_pixel=3,
                bits_stored=0,
                interleave_mode=0,
            )

        msg = r"Invalid 'bits_stored' value 17: must be in the range \(1, 16\)"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(
                TEST8.tobytes(),
                rows=256,
                columns=256,
                samples_per_pixel=3,
                bits_stored=17,
                interleave_mode=0,
            )

    def test_invalid_interleave_mode_raises(self, TEST8):
        msg = "Invalid 'interleave_mode' value -1: must be 0, 1 or 2"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(
                TEST8.tobytes(),
                rows=256,
                columns=256,
                samples_per_pixel=3,
                bits_stored=8,
                interleave_mode=-1,
            )

        msg = "Invalid 'interleave_mode' value 3: must be 0, 1 or 2"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(
                TEST8.tobytes(),
                rows=256,
                columns=256,
                samples_per_pixel=3,
                bits_stored=8,
                interleave_mode=3,
            )

    def test_invalid_rows_raises(self, TEST8):
        msg = r"Invalid 'rows' value 0: must be in the range \(1, 65535\)"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(
                TEST8.tobytes(),
                rows=0,
                columns=256,
                samples_per_pixel=3,
                bits_stored=8,
                interleave_mode=2,
            )

        msg = r"Invalid 'rows' value 65536: must be in the range \(1, 65535\)"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(
                TEST8.tobytes(),
                rows=65536,
                columns=256,
                samples_per_pixel=3,
                bits_stored=8,
                interleave_mode=2,
            )

    def test_invalid_columns_raises(self, TEST8):
        msg = r"Invalid 'columns' value 0: must be in the range \(1, 65535\)"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(
                TEST8.tobytes(),
                rows=256,
                columns=0,
                samples_per_pixel=3,
                bits_stored=8,
                interleave_mode=2,
            )

        msg = r"Invalid 'columns' value 65536: must be in the range \(1, 65535\)"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(
                TEST8.tobytes(),
                rows=256,
                columns=65536,
                samples_per_pixel=3,
                bits_stored=8,
                interleave_mode=2,
            )

    def test_invalid_src_length_raises(self, TEST8):
        msg = (
            "The 'src' length of 196608 bytes does not match the expected "
            "length determined from 'rows', 'columns', 'samples_per_pixel' and "
            "'bits_stored'"
        )
        with pytest.raises(ValueError, match=msg):
            encode_buffer(
                TEST8.tobytes(),
                rows=256,
                columns=256,
                samples_per_pixel=3,
                bits_stored=16,
                interleave_mode=2,
            )

    def test_TEST8_ILV0(self, TEST8):
        # 3 component, ILV 0
        # Convert from colour-by-pixel to colour-by-plane
        arr = TEST8.transpose(2, 0, 1)
        assert arr.shape == (3, 256, 256)
        enc = encode_buffer(
            arr.tobytes(),
            rows=256,
            columns=256,
            samples_per_pixel=3,
            bits_stored=8,
            interleave_mode=0,
        )

        arr = jlsread(enc)
        assert np.array_equal(arr, TEST8)

    def test_TEST8_ILV1(self, TEST8):
        # 3 component, ILV 1
        assert TEST8.shape == (256, 256, 3)
        enc = encode_buffer(
            TEST8.tobytes(),
            rows=256,
            columns=256,
            samples_per_pixel=3,
            bits_stored=8,
            interleave_mode=1,
        )

        arr = jlsread(enc)
        assert np.array_equal(arr, TEST8)

    def test_TEST8_ILV2(self, TEST8):
        # 3 component, ILV 2
        assert TEST8.shape == (256, 256, 3)
        enc = encode_buffer(
            TEST8.tobytes(),
            rows=256,
            columns=256,
            samples_per_pixel=3,
            bits_stored=8,
            interleave_mode=2,
        )

        arr = jlsread(enc)
        assert np.array_equal(arr, TEST8)

    def test_TEST8R(self, TEST8R):
        # 1 component, ILV 0
        assert TEST8R.shape == (256, 256)
        enc = encode_buffer(
            TEST8R.tobytes(),
            rows=256,
            columns=256,
            samples_per_pixel=1,
            bits_stored=8,
            interleave_mode=0,
        )

        arr = jlsread(enc)
        assert np.array_equal(arr, TEST8R)

    def test_TEST8G(self, TEST8G):
        assert TEST8G.shape == (256, 256)
        enc = encode_buffer(
            TEST8G.tobytes(),
            rows=256,
            columns=256,
            samples_per_pixel=1,
            bits_stored=8,
            interleave_mode=0,
        )

        arr = jlsread(enc)
        assert np.array_equal(arr, TEST8G)

    def test_TEST8B(self, TEST8B):
        assert TEST8B.shape == (256, 256)
        enc = encode_buffer(
            TEST8B.tobytes(),
            rows=256,
            columns=256,
            samples_per_pixel=1,
            bits_stored=8,
            interleave_mode=0,
        )

        arr = jlsread(enc)
        assert np.array_equal(arr, TEST8B)

    def test_TEST8BS2(self, TEST8BS2):
        assert TEST8BS2.shape == (128, 128)
        enc = encode_buffer(
            TEST8BS2.tobytes(),
            rows=128,
            columns=128,
            samples_per_pixel=1,
            bits_stored=8,
            interleave_mode=0,
        )

        arr = jlsread(enc)
        assert np.array_equal(arr, TEST8BS2)

    def test_TEST8GR4(self, TEST8GR4):
        assert TEST8GR4.shape == (64, 256)
        enc = encode_buffer(
            TEST8GR4.tobytes(),
            rows=64,
            columns=256,
            samples_per_pixel=1,
            bits_stored=8,
            interleave_mode=0,
        )

        arr = jlsread(enc)
        assert np.array_equal(arr, TEST8GR4)

    def test_TEST16(self, TEST16):
        assert TEST16.shape == (256, 256)
        enc = encode_buffer(
            TEST16.tobytes(),
            rows=256,
            columns=256,
            samples_per_pixel=1,
            bits_stored=12,
            interleave_mode=0,
        )

        arr = jlsread(enc)
        assert np.array_equal(arr, TEST16)

    def test_TEST8_lossy(self, TEST8):
        enc = encode_buffer(
            TEST8.tobytes(),
            rows=256,
            columns=256,
            samples_per_pixel=3,
            bits_stored=8,
            lossy_error=5,
            interleave_mode=2,
        )

        arr = jlsread(enc)
        assert not np.array_equal(arr, TEST8)
        assert np.allclose(arr, TEST8, atol=5)

    def test_TEST8R_lossy(self, TEST8R):
        enc = encode_buffer(
            TEST8R.tobytes(),
            rows=256,
            columns=256,
            samples_per_pixel=1,
            bits_stored=8,
            lossy_error=3,
            interleave_mode=0,
        )

        arr = jlsread(enc)
        assert not np.array_equal(arr, TEST8R)
        assert np.allclose(arr, TEST8R, atol=3)

    def test_TEST8R_16_08(self, TEST8R):
        """Test 16-bit container for 8-bit samples"""
        arr = TEST8R.astype("u2")
        buffer = arr.tobytes()
        assert len(buffer) == 256 * 256 * 2

        enc = encode_buffer(
            buffer,
            rows=256,
            columns=256,
            samples_per_pixel=1,
            bits_stored=8,
            lossy_error=0,
            interleave_mode=0,
        )

        arr = jlsread(enc)
        assert np.array_equal(arr, TEST8R)


class TestEncodePixelData:
    """Tests for encode_pixel_data()"""

    def test_TEST8_ILV0(self, TEST8):
        # 3 component, ILV 0
        # Convert from colour-by-pixel to colour-by-plane
        arr = TEST8.transpose(2, 0, 1)
        assert arr.shape == (3, 256, 256)

        kwargs = {
            "rows": 256,
            "columns": 256,
            "samples_per_pixel": 3,
            "bits_stored": 8,
            "planar_configuration": 1,
        }
        enc = encode_pixel_data(arr.tobytes(), **kwargs)

        arr = jlsread(enc)
        assert np.array_equal(arr, TEST8)

    def test_TEST8_ILV2(self, TEST8):
        # 3 component, ILV 2
        assert TEST8.shape == (256, 256, 3)
        kwargs = {
            "rows": 256,
            "columns": 256,
            "samples_per_pixel": 3,
            "bits_stored": 8,
            "planar_configuration": 0,
        }
        enc = encode_pixel_data(TEST8.tobytes(), **kwargs)

        arr = jlsread(enc)
        assert np.array_equal(arr, TEST8)

    def test_TEST8R(self, TEST8R):
        # 1 component, ILV 0
        assert TEST8R.shape == (256, 256)

        kwargs = {
            "rows": 256,
            "columns": 256,
            "samples_per_pixel": 1,
            "bits_stored": 8,
        }
        enc = encode_pixel_data(TEST8R.tobytes(), **kwargs)

        arr = jlsread(enc)
        assert np.array_equal(arr, TEST8R)

    def test_TEST8_lossy(self, TEST8):
        kwargs = {
            "rows": 256,
            "columns": 256,
            "samples_per_pixel": 3,
            "bits_stored": 8,
            "planar_configuration": 0,
        }

        enc = encode_pixel_data(TEST8.tobytes(), lossy_error=5, **kwargs)

        arr = jlsread(enc)
        assert not np.array_equal(arr, TEST8)
        assert np.allclose(arr, TEST8, atol=5)

    def test_TEST8R_lossy(self, TEST8R):
        kwargs = {
            "rows": 256,
            "columns": 256,
            "samples_per_pixel": 1,
            "bits_stored": 8,
        }

        enc = encode_pixel_data(TEST8R.tobytes(), lossy_error=3, **kwargs)

        arr = jlsread(enc)
        assert not np.array_equal(arr, TEST8R)
        assert np.allclose(arr, TEST8R, atol=3)


def test_write(TEST8R):
    """Test write()"""
    with TemporaryDirectory() as tdir:
        p = Path(tdir)
        write(p / "test.jls", TEST8R)

        arr = jlsread(p / "test.jls")
        assert np.array_equal(arr, TEST8R)


def test_jlswrite(TEST8R):
    """Test jlswrite()"""
    with TemporaryDirectory() as tdir:
        p = Path(tdir)
        jlswrite(TEST8R, p / "test.jls")

        arr = jlsread(p / "test.jls")
        assert np.array_equal(arr, TEST8R)

    with TemporaryDirectory() as tdir:
        p = Path(tdir) / "test.jls"

        with p.open("wb") as f:
            jlswrite(TEST8R, f)

        arr = jlsread(p)
        assert np.array_equal(arr, TEST8R)
