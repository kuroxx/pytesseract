#!/usr/bin/env python
from __future__ import annotations

import logging
import re
import shlex
import string
import subprocess
import sys
from contextlib import contextmanager
from csv import QUOTE_NONE
from errno import ENOENT
from functools import wraps
from glob import iglob
from io import BytesIO
from os import environ
from os import extsep
from os import linesep
from os import remove
from os.path import normcase
from os.path import normpath
from os.path import realpath
from tempfile import NamedTemporaryFile
from time import sleep

from packaging.version import InvalidVersion
from packaging.version import parse
from packaging.version import Version
from PIL import Image


tesseract_cmd = 'tesseract'

try:
    from numpy import ndarray

    numpy_installed = True
except ModuleNotFoundError:
    numpy_installed = False

try:
    import pandas as pd

    pandas_installed = True
except ModuleNotFoundError:
    pandas_installed = False

LOGGER = logging.getLogger('pytesseract')

DEFAULT_ENCODING = 'utf-8'
LANG_PATTERN = re.compile('^[a-z0-9_]+$')
RGB_MODE = 'RGB'
SUPPORTED_FORMATS = {
    'JPEG',
    'JPEG2000',
    'PNG',
    'PBM',
    'PGM',
    'PPM',
    'TIFF',
    'BMP',
    'GIF',
    'WEBP',
}

OSD_KEYS = {
    'Page number': ('page_num', int),
    'Orientation in degrees': ('orientation', int),
    'Rotate': ('rotate', int),
    'Orientation confidence': ('orientation_conf', float),
    'Script': ('script', str),
    'Script confidence': ('script_conf', float),
}

EXTENTION_TO_CONFIG = {
    'box': 'tessedit_create_boxfile=1 batch.nochop makebox',
    'xml': 'tessedit_create_alto=1',
    'hocr': 'tessedit_create_hocr=1',
    'tsv': 'tessedit_create_tsv=1',
}

TESSERACT_MIN_VERSION = Version('3.05')
TESSERACT_ALTO_VERSION = Version('4.1.0')


class Output:
    BYTES = 'bytes'
    DATAFRAME = 'data.frame'
    DICT = 'dict'
    STRING = 'string'


class PandasNotSupported(EnvironmentError):
    def __init__(self):
        super().__init__('Missing pandas package')


class TesseractError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        self.message = message
        self.args = (status, message)


class TesseractNotFoundError(EnvironmentError):
    def __init__(self):
        super().__init__(
            f"{tesseract_cmd} is not installed or it's not in your PATH."
            f' See README file for more information.',
        )


class TSVNotSupported(EnvironmentError):
    def __init__(self):
        super().__init__(
            'TSV output not supported. Tesseract >= 3.05 required',
        )


class ALTONotSupported(EnvironmentError):
    def __init__(self):
        super().__init__(
            'ALTO output not supported. Tesseract >= 4.1.0 required',
        )


def kill(process, code):
    process.terminate()
    try:
        process.wait(1)
    except TypeError:  # python2 Popen.wait(1) fallback
        sleep(1)
    except Exception:  # python3 subprocess.TimeoutExpired
        pass
    finally:
        process.kill()
        process.returncode = code


@contextmanager
def timeout_manager(proc, seconds=None):
    try:
        if not seconds:
            yield proc.communicate()[1]
            return

        try:
            _, error_string = proc.communicate(timeout=seconds)
            yield error_string
        except subprocess.TimeoutExpired:
            kill(proc, -1)
            raise RuntimeError('Tesseract process timeout')
    finally:
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()


def run_once(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not kwargs.pop('cached', False) or wrapper._result is wrapper:
            wrapper._result = func(*args, **kwargs)
        return wrapper._result

    wrapper._result = wrapper
    return wrapper


def get_errors(error_string):
    # Optimize string processing with minimal allocations
    decoded = error_string.decode(DEFAULT_ENCODING)
    # Use replace which is faster than splitlines + join for this case
    return decoded.replace('\n', ' ').replace('\r', ' ').strip()


def cleanup(temp_name):
    """Tries to remove temp files by filename wildcard path."""
    if not temp_name:
        return
    
    # Use iglob directly for better performance, avoiding list materialization
    for filename in iglob(f'{temp_name}*'):
        try:
            remove(filename)
        except OSError as e:
            if e.errno != ENOENT:
                raise


def prepare(image):
    if numpy_installed and isinstance(image, ndarray):
        image = Image.fromarray(image)

    if not isinstance(image, Image.Image):
        raise TypeError('Unsupported image object')

    extension = 'PNG' if not image.format else image.format
    if extension not in SUPPORTED_FORMATS:
        raise TypeError('Unsupported image format/type')

    # Cache band check for performance and avoid redundant calls
    bands = image.getbands()
    if len(bands) > 3 and 'A' in bands:
        # Only process alpha channel if it actually exists and has transparency
        alpha_channel = image.getchannel('A')
        # Quick check if alpha channel has any transparency
        if alpha_channel.getextrema()[0] < 255:
            # discard and replace the alpha channel with white background
            background = Image.new(RGB_MODE, image.size, (255, 255, 255))
            background.paste(image, (0, 0), alpha_channel)
            image = background
        else:
            # No transparency, just convert without alpha
            image = image.convert(RGB_MODE)

    image.format = extension
    return image, extension


@contextmanager
def save(image):
    try:
        with NamedTemporaryFile(prefix='tess_', delete=False) as f:
            if isinstance(image, str):
                # Cache the path normalization result
                normalized_path = realpath(normpath(normcase(image)))
                yield f.name, normalized_path
                return
            image, extension = prepare(image)
            input_file_name = f'{f.name}_input{extsep}{extension}'
            # Optimize image saving with better compression settings
            save_kwargs = {'format': image.format}
            if extension == 'PNG':
                save_kwargs.update({'optimize': True, 'compress_level': 1})  # Fast compression
            elif extension in ('JPEG', 'JPEG2000'):
                save_kwargs.update({'optimize': True, 'quality': 95})  # High quality, fast
            image.save(input_file_name, **save_kwargs)
            yield f.name, input_file_name
    finally:
        cleanup(f.name)


def subprocess_args(include_stdout=True):
    # See https://github.com/pyinstaller/pyinstaller/wiki/Recipe-subprocess
    # for reference and comments.

    kwargs = {
        'stdin': subprocess.PIPE,
        'stderr': subprocess.PIPE,
        'startupinfo': None,
        'env': environ,
    }

    if hasattr(subprocess, 'STARTUPINFO'):
        kwargs['startupinfo'] = subprocess.STARTUPINFO()
        kwargs['startupinfo'].dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs['startupinfo'].wShowWindow = subprocess.SW_HIDE

    if include_stdout:
        kwargs['stdout'] = subprocess.PIPE
    else:
        kwargs['stdout'] = subprocess.DEVNULL

    return kwargs


def run_tesseract(
    input_filename,
    output_filename_base,
    extension,
    lang,
    config='',
    nice=0,
    timeout=0,
):
    cmd_args = []
    not_windows = not (sys.platform == 'win32')

    if not_windows and nice != 0:
        cmd_args += ('nice', '-n', str(nice))

    cmd_args += (tesseract_cmd, input_filename, output_filename_base)

    if lang is not None:
        cmd_args += ('-l', lang)

    if config:
        cmd_args += shlex.split(config, posix=not_windows)

    for _extension in extension.split():
        if _extension not in {'box', 'osd', 'tsv', 'xml'}:
            cmd_args.append(_extension)
    LOGGER.debug('%r', cmd_args)

    try:
        proc = subprocess.Popen(cmd_args, **subprocess_args())
    except OSError as e:
        if e.errno != ENOENT:
            raise
        else:
            raise TesseractNotFoundError()

    with timeout_manager(proc, timeout) as error_string:
        if proc.returncode:
            raise TesseractError(proc.returncode, get_errors(error_string))


def _read_output(filename: str, return_bytes: bool = False):
    with open(filename, 'rb', buffering=65536) as output_file:
        if return_bytes:
            return output_file.read()
        return output_file.read().decode(DEFAULT_ENCODING)


def run_and_get_multiple_output(
    image,
    extensions: list[str],
    lang: str | None = None,
    nice: int = 0,
    timeout: int = 0,
    return_bytes: bool = False,
):
    # Pre-compute config strings and binary extensions for efficiency
    config_parts = [EXTENTION_TO_CONFIG.get(ext, '') for ext in extensions if ext in EXTENTION_TO_CONFIG]
    config = f'-c {" ".join(config_parts)}' if config_parts else ''
    
    # Pre-determine which extensions need binary reading
    binary_extensions = {'pdf', 'hocr'}
    read_as_binary = [ext in binary_extensions or return_bytes for ext in extensions]

    with save(image) as (temp_name, input_filename):
        kwargs = {
            'input_filename': input_filename,
            'output_filename_base': temp_name,
            'extension': ' '.join(extensions),
            'lang': lang,
            'config': config,
            'nice': nice,
            'timeout': timeout,
        }

        run_tesseract(**kwargs)

        # Read all outputs with optimized binary flags
        base_path = kwargs['output_filename_base']
        return [
            _read_output(f"{base_path}{extsep}{ext}", read_binary)
            for ext, read_binary in zip(extensions, read_as_binary)
        ]


def run_and_get_output(
    image,
    extension='',
    lang=None,
    config='',
    nice=0,
    timeout=0,
    return_bytes=False,
):
    with save(image) as (temp_name, input_filename):
        kwargs = {
            'input_filename': input_filename,
            'output_filename_base': temp_name,
            'extension': extension,
            'lang': lang,
            'config': config,
            'nice': nice,
            'timeout': timeout,
        }

        run_tesseract(**kwargs)
        return _read_output(
            f"{kwargs['output_filename_base']}{extsep}{extension}",
            return_bytes,
        )


def file_to_dict(tsv, cell_delimiter, str_col_idx):
    lines = tsv.strip().split('\n')
    if len(lines) < 2:
        return {}

    # Pre-split all rows for better performance
    rows = [line.split(cell_delimiter) for line in lines]
    header = rows.pop(0)
    length = len(header)
    
    if rows and len(rows[-1]) < length:
        # Fixes bug that occurs when last text string in TSV is null, and
        # last row is missing a final cell in TSV file
        rows[-1].append('')

    if str_col_idx < 0:
        str_col_idx += length

    # Pre-allocate result dictionary with exact size
    num_rows = len(rows)
    result = {head: [None] * num_rows for head in header}
    
    # Process columns in batches for better cache locality
    for row_idx, row in enumerate(rows):
        row_len = len(row)
        for col_idx, head in enumerate(header):
            if col_idx >= row_len:
                continue
                
            cell_value = row[col_idx]
            if col_idx != str_col_idx and cell_value:
                # Fast integer conversion for numeric columns
                if cell_value.isdigit() or (cell_value.startswith('-') and cell_value[1:].isdigit()):
                    result[head][row_idx] = int(cell_value)
                else:
                    try:
                        # Try float conversion only if needed
                        float_val = float(cell_value)
                        result[head][row_idx] = int(float_val) if float_val.is_integer() else float_val
                    except ValueError:
                        result[head][row_idx] = cell_value
            else:
                result[head][row_idx] = cell_value
    
    # Clean up None values from pre-allocated lists
    for head in header:
        result[head] = [val for val in result[head] if val is not None]

    return result


def is_valid(val, _type):
    if _type is int:
        return val.isdigit()

    if _type is float:
        try:
            float(val)
            return True
        except ValueError:
            return False

    return True


def osd_to_dict(osd):
    result = {}
    for line in osd.split('\n'):
        if ': ' not in line:
            continue
        kv = line.split(': ', 1)  # Split only on first occurrence
        if len(kv) == 2 and kv[0] in OSD_KEYS and is_valid(kv[1], OSD_KEYS[kv[0]][1]):
            result[OSD_KEYS[kv[0]][0]] = OSD_KEYS[kv[0]][1](kv[1])
    return result


@run_once
def get_languages(config=''):
    cmd_args = [tesseract_cmd, '--list-langs']
    if config:
        cmd_args += shlex.split(config)

    try:
        result = subprocess.run(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError:
        raise TesseractNotFoundError()

    # tesseract 3.x
    if result.returncode not in (0, 1):
        raise TesseractNotFoundError()

    languages = []
    if result.stdout:
        for line in result.stdout.decode(DEFAULT_ENCODING).split(linesep):
            lang = line.strip()
            if LANG_PATTERN.match(lang):
                languages.append(lang)

    return languages


@run_once
def get_tesseract_version():
    """
    Returns Version object of the Tesseract version
    """
    try:
        output = subprocess.check_output(
            [tesseract_cmd, '--version'],
            stderr=subprocess.STDOUT,
            env=environ,
            stdin=subprocess.DEVNULL,
        )
    except OSError:
        raise TesseractNotFoundError()

    raw_version = output.decode(DEFAULT_ENCODING)
    str_version, *_ = raw_version.lstrip(string.printable[10:]).partition(' ')
    str_version, *_ = str_version.partition('-')

    try:
        version = parse(str_version)
        assert version >= TESSERACT_MIN_VERSION
    except (AssertionError, InvalidVersion):
        raise SystemExit(f'Invalid tesseract version: "{raw_version}"')

    return version


def image_to_string(
    image,
    lang=None,
    config='',
    nice=0,
    output_type=Output.STRING,
    timeout=0,
):
    """
    Returns the result of a Tesseract OCR run on the provided image to string
    """
    args = [image, 'txt', lang, config, nice, timeout]

    return {
        Output.BYTES: lambda: run_and_get_output(*(args + [True])),
        Output.DICT: lambda: {'text': run_and_get_output(*args)},
        Output.STRING: lambda: run_and_get_output(*args),
    }[output_type]()


def image_to_pdf_or_hocr(
    image,
    lang=None,
    config='',
    nice=0,
    extension='pdf',
    timeout=0,
):
    """
    Returns the result of a Tesseract OCR run on the provided image to pdf/hocr
    """

    if extension not in {'pdf', 'hocr'}:
        raise ValueError(f'Unsupported extension: {extension}')

    if extension == 'hocr':
        config = f'-c tessedit_create_hocr=1 {config.strip()}'

    args = [image, extension, lang, config, nice, timeout, True]

    return run_and_get_output(*args)


def image_to_alto_xml(
    image,
    lang=None,
    config='',
    nice=0,
    timeout=0,
):
    """
    Returns the result of a Tesseract OCR run on the provided image to ALTO XML
    """

    if get_tesseract_version(cached=True) < TESSERACT_ALTO_VERSION:
        raise ALTONotSupported()

    config = f'-c tessedit_create_alto=1 {config.strip()}'
    args = [image, 'xml', lang, config, nice, timeout, True]

    return run_and_get_output(*args)


def image_to_boxes(
    image,
    lang=None,
    config='',
    nice=0,
    output_type=Output.STRING,
    timeout=0,
):
    """
    Returns string containing recognized characters and their box boundaries
    """
    config = (
        f'{config.strip()} -c tessedit_create_boxfile=1 batch.nochop makebox'
    )
    args = [image, 'box', lang, config, nice, timeout]

    return {
        Output.BYTES: lambda: run_and_get_output(*(args + [True])),
        Output.DICT: lambda: file_to_dict(
            f'char left bottom right top page\n{run_and_get_output(*args)}',
            ' ',
            0,
        ),
        Output.STRING: lambda: run_and_get_output(*args),
    }[output_type]()


def get_pandas_output(args, config=None):
    if not pandas_installed:
        raise PandasNotSupported()

    kwargs = {'quoting': QUOTE_NONE, 'sep': '\t'}
    try:
        kwargs.update(config)
    except (TypeError, ValueError):
        pass

    return pd.read_csv(BytesIO(run_and_get_output(*args)), **kwargs)


def image_to_data(
    image,
    lang=None,
    config='',
    nice=0,
    output_type=Output.STRING,
    timeout=0,
    pandas_config=None,
):
    """
    Returns string containing box boundaries, confidences,
    and other information. Requires Tesseract 3.05+
    """

    if get_tesseract_version(cached=True) < TESSERACT_MIN_VERSION:
        raise TSVNotSupported()

    config = f'-c tessedit_create_tsv=1 {config.strip()}'
    args = [image, 'tsv', lang, config, nice, timeout]

    return {
        Output.BYTES: lambda: run_and_get_output(*(args + [True])),
        Output.DATAFRAME: lambda: get_pandas_output(
            args + [True],
            pandas_config,
        ),
        Output.DICT: lambda: file_to_dict(run_and_get_output(*args), '\t', -1),
        Output.STRING: lambda: run_and_get_output(*args),
    }[output_type]()


def image_to_osd(
    image,
    lang='osd',
    config='',
    nice=0,
    output_type=Output.STRING,
    timeout=0,
):
    """
    Returns string containing the orientation and script detection (OSD)
    """
    config = f'--psm 0 {config.strip()}'
    args = [image, 'osd', lang, config, nice, timeout]

    return {
        Output.BYTES: lambda: run_and_get_output(*(args + [True])),
        Output.DICT: lambda: osd_to_dict(run_and_get_output(*args)),
        Output.STRING: lambda: run_and_get_output(*args),
    }[output_type]()


def main():
    if len(sys.argv) == 2:
        filename, lang = sys.argv[1], None
    elif len(sys.argv) == 4 and sys.argv[1] == '-l':
        filename, lang = sys.argv[3], sys.argv[2]
    else:
        print('Usage: pytesseract [-l lang] input_file\n', file=sys.stderr)
        return 2

    try:
        with Image.open(filename) as img:
            print(image_to_string(img, lang=lang))
    except TesseractNotFoundError as e:
        print(f'{str(e)}\n', file=sys.stderr)
        return 1
    except OSError as e:
        print(f'{type(e).__name__}: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
