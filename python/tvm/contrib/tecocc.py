# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Utility to invoke tecocc compiler in the system"""
from __future__ import absolute_import as _abs

import subprocess
import os
# import warnings

import tvm._ffi
# from tvm.target import Target

from . import utils
from .._ffi.base import py_str


def compile_sdaa(code, target_format="so", options=None, path_target=None):
    """Compile sdaa code with TECOCC from env.

    Parameters
    ----------
    code : str
        The sdaa code.

    target_format : str
        The target format of tecocc compiler.

    arch : str
        The sdaa architecture.

    options : str or list of str
        The additional options.

    path_target : str, optional
        Output file.

    Return
    ------
    cubin : bytearray
        The bytearray of the cubin
    """
    # if arch is None:
    #     # If None, then it will use `tvm.target.Target.current().arch`.
    #     # Target arch could be a str like "sm_xx", or a list, such as
    #     # [
    #     #   "-gencode", "arch=compute_52,code=sm_52",
    #     #   "-gencode", "arch=compute_70,code=sm_70"
    #     # ]
    #     compute_version = "".join(
    #         get_target_compute_version(Target.current(allow_none=True)).split(".")
    #     )
    #     arch = ["-gencode", f"arch=compute_{compute_version},code=sm_{compute_version}"]

    temp = utils.tempdir()
    if target_format not in ["so", "fatbin"]:
        raise ValueError("target_format must be in so, fatbin")
    temp_code = temp.relpath("my_kernel.swcu")
    temp_target = temp.relpath("my_kernel.%s" % target_format)

    with open(temp_code, "w") as out_file:
        out_file.write(code)

    file_target = path_target if path_target else temp_target
    cmd_o = ["tecocc"]
    inter_file = temp.relpath("my_kernel.o")
    cmd_o += [temp_code]  + ["--sdaa-device-only","-c"]
    # if options:
    #     if isinstance(options, str):
    #         cmd_o += [options]
    #     elif isinstance(options, list):
    #         cmd_o += options
    #     else:
    #         raise ValueError("options must be str or list of str")
    
    cmd_o += ["-o", inter_file]

    # NOTE: ccbin option can be used to tell tecocc where to find the c++ compiler
    # just in case it is not in the path. On Windows it is not in the path by default.
    # However, we cannot use TVM_CXX_COMPILER_PATH because the runtime env.
    # Because it is hard to do runtime compiler detection, we require tecocc is configured
    # correctly by default.
    # if cxx_compiler_path != "":
    #    cmd += ["-ccbin", cxx_compiler_path]

    # proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)


    # (out, _) = proc.communicate()

    proc_o = subprocess.Popen(cmd_o, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out_o, _) = proc_o.communicate()

    if proc_o.returncode != 0:
        msg = code
        msg += "\nCompilation error:\n"
        msg += py_str(out_o)
        raise RuntimeError(msg)

    # try:
    #     subprocess.run(cmd_o, capture_output=True)
    # except subprocess.CalledProcessError as err: 
    #     msg = code
    #     msg += "\nCompilation error:\n"
    #     msg += py_str(err)
    #     raise RuntimeError(msg)
    
    cmd_so = ["tecocc"]
    cmd_so += [inter_file] + ["-device-only", "-fPIC","-shared"]
    cmd_so += ["-o", file_target]

    proc_so = subprocess.Popen(cmd_so, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out_so, _) = proc_so.communicate()

    if proc_so.returncode != 0:
        msg = code
        msg += "\nCompilation error:\n"
        msg += py_str(out_so)
        raise RuntimeError(msg)


    # try:
    #     subprocess.run(cmd_so, capture_output=True)
    # except subprocess.CalledProcessError as err: 
    #     msg = code
    #     msg += "\nCompilation error:\n"
    #     msg += py_str(err)
    #     raise RuntimeError(msg)

    # file_target = '/home/tangxf/zly/sdaa/my_kernel.so'
    with open(file_target, "rb") as f:
        data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        return data
        #return file_target


def find_sdaa_path():
    """Utility function to find sdaa path

    Returns
    -------
    path : str
        Path to sdaa root.
    """
    if "SDAA_PATH" in os.environ:
        return os.environ["SDAA_PATH"]
    cmd = ["which", "tecocc"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    out = py_str(out)
    if proc.returncode == 0:
        return os.path.realpath(os.path.join(str(out).strip(), "../.."))
    sdaa_path = "/usr/local/sdaa"
    if os.path.exists(os.path.join(sdaa_path, "bin/tecocc")):
        return sdaa_path
    raise RuntimeError("Cannot find sdaa path")


def get_sdaa_version(sdaa_path=None):
    """Utility function to get sdaa version

    Parameters
    ----------
    sdaa_path : Optional[str]

        Path to sdaa root.  If None is passed, will use
        `find_sdaa_path()` as default.

    Returns
    -------
    version : float
        The sdaa version

    """
    if sdaa_path is None:
        sdaa_path = find_sdaa_path()

    version_file_path = os.path.join(sdaa_path, "version.txt")
    if not os.path.exists(version_file_path):
        # Debian/Ubuntu repackaged SDAA path
        version_file_path = os.path.join(sdaa_path, "lib", "sdaa", "version.txt")
    try:
        with open(version_file_path) as f:
            version_str = f.read().strip().split()[-1]
            return tuple(int(field) for field in version_str.split("."))
    except FileNotFoundError:
        pass

    cmd = [os.path.join(sdaa_path, "bin", "tecocc"), "--version"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    out = py_str(out)
    if proc.returncode == 0:
        release_line = [l for l in out.split("\n") if "release" in l][0]
        release_fields = [s.strip() for s in release_line.split(",")]
        version_str = [f[1:] for f in release_fields if f.startswith("V")][0]
        return tuple(int(field) for field in version_str.split("."))
    raise RuntimeError("Cannot read sdaa version file")


@tvm._ffi.register_func
def tvm_callback_sdaa_compile(code, fmt, ops):
    """use tecocc to generate fatbin code for better optimization"""
    fatbin = compile_sdaa(code, target_format=fmt, options=ops)
    return fatbin

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

TASK = "test_sdaa"
USE_MANUAL_CODE = False

@tvm._ffi.register_func
def tvm_callback_sdaa_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.swcu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.swcu" % TASK).read()
    return code


# @tvm._ffi.register_func("tvm_callback_libdevice_path")
# def find_libdevice_path(arch):
#     """Utility function to find libdevice

#     Parameters
#     ----------
#     arch : int
#         The compute architecture in int

#     Returns
#     -------
#     path : str
#         Path to libdevice.
#     """
#     sdaa_path = find_sdaa_path()
#     lib_path = os.path.join(sdaa_path, "nvvm/libdevice")
#     if not os.path.exists(lib_path):
#         # Debian/Ubuntu repackaged SDAA path
#         lib_path = os.path.join(sdaa_path, "lib/nvidia-sdaa-toolkit/libdevice")
#     selected_ver = 0
#     selected_path = None
#     sdaa_ver = get_sdaa_version(sdaa_path)
#     major_minor = (sdaa_ver[0], sdaa_ver[1])
#     if major_minor in (
#         (9, 0),
#         (9, 1),
#         (10, 0),
#         (10, 1),
#         (10, 2),
#         (11, 0),
#         (11, 1),
#         (11, 2),
#         (11, 3),
#     ):
#         path = os.path.join(lib_path, "libdevice.10.bc")
#     else:
#         for fn in os.listdir(lib_path):
#             if not fn.startswith("libdevice"):
#                 continue

#             try:
#                 # expected pattern: libdevice.${ARCH}.10.bc
#                 #             e.g., libdevice.compute_20.10.bc
#                 ver = int(fn.split(".")[-3].split("_")[-1])
#                 if selected_ver < ver <= arch:
#                     selected_ver = ver
#                     selected_path = fn
#             except ValueError:
#                 # it can just be `libdevice.10.bc` in SDAA 10
#                 selected_path = fn

#         if selected_path is None:
#             raise RuntimeError("Cannot find libdevice for arch {}".format(arch))
#         path = os.path.join(lib_path, selected_path)
#     return path


# def callback_libdevice_path(arch):
#     try:
#         return find_libdevice_path(arch)
#     except RuntimeError:
#         warnings.warn("Cannot find libdevice path")
#         return ""


# def get_target_compute_version(target=None):
#     """Utility function to get compute capability of compilation target.

#     Looks for the target arch in three different places, first in the target input, then the
#     Target.current() scope, and finally the GPU device (if it exists).

#     Parameters
#     ----------
#     target : tvm.target.Target, optional
#         The compilation target

#     Returns
#     -------
#     compute_version : str
#         compute capability of a GPU (e.g. "8.6")
#     """
#     # 1. input target object
#     # 2. Target.current()
#     target = target or Target.current()
#     if target and target.arch:
#         major, minor = target.arch.split("_")[1]
#         return major + "." + minor

#     # 3. GPU compute version
#     if tvm.sdaa(0).exist:
#         return tvm.sdaa(0).compute_version

#     raise ValueError(
#         "No SDAA architecture was specified or GPU detected."
#         "Try specifying it by adding '-arch=sm_xx' to your target."
#     )


# def parse_compute_version(compute_version):
#     """Parse compute capability string to divide major and minor version

#     Parameters
#     ----------
#     compute_version : str
#         compute capability of a GPU (e.g. "6.0")

#     Returns
#     -------
#     major : int
#         major version number
#     minor : int
#         minor version number
#     """
#     split_ver = compute_version.split(".")
#     try:
#         major = int(split_ver[0])
#         minor = int(split_ver[1])
#         return major, minor
#     except (IndexError, ValueError) as err:
#         # pylint: disable=raise-missing-from
#         raise RuntimeError("Compute version parsing error: " + str(err))


# def have_fp16(compute_version):
#     """Either fp16 support is provided in the compute capability or not

#     Parameters
#     ----------
#     compute_version: str
#         compute capability of a GPU (e.g. "6.0")
#     """
#     major, minor = parse_compute_version(compute_version)
#     # fp 16 support in reference to:
#     # https://docs.nvidia.com/sdaa/sdaa-c-programming-guide/#arithmetic-instructions
#     if major == 5 and minor == 3:
#         return True
#     if major >= 6:
#         return True

#     return False


# def have_int8(compute_version):
#     """Either int8 support is provided in the compute capability or not

#     Parameters
#     ----------
#     compute_version : str
#         compute capability of a GPU (e.g. "6.1")
#     """
#     major, _ = parse_compute_version(compute_version)
#     if major >= 6:
#         return True

#     return False


# def have_tensorcore(compute_version=None, target=None):
#     """Either TensorCore support is provided in the compute capability or not

#     Parameters
#     ----------
#     compute_version : str, optional
#         compute capability of a GPU (e.g. "7.0").

#     target : tvm.target.Target, optional
#         The compilation target, will be used to determine arch if compute_version
#         isn't specified.
#     """
#     if compute_version is None:
#         if tvm.sdaa(0).exist:
#             compute_version = tvm.sdaa(0).compute_version
#         else:
#             if target is None or "arch" not in target.attrs:
#                 warnings.warn(
#                     "Tensorcore will be disabled due to no SDAA architecture specified."
#                     "Try specifying it by adding '-arch=sm_xx' to your target."
#                 )
#                 return False
#             compute_version = target.attrs["arch"]
#             # Compute version will be in the form "sm_{major}{minor}"
#             major, minor = compute_version.split("_")[1]
#             compute_version = major + "." + minor
#     major, _ = parse_compute_version(compute_version)
#     if major >= 7:
#         return True

#     return False


# def have_sdaagraph():
#     """Either SDAA Graph support is provided"""
#     try:
#         sdaa_ver = get_sdaa_version()
#         if sdaa_ver < (10, 0):
#             return False
#         return True
#     except RuntimeError:
#         return False


# def have_bf16(compute_version):
#     """Either bf16 support is provided in the compute capability or not

#     Parameters
#     ----------
#     compute_version : str
#         compute capability of a GPU (e.g. "8.0")
#     """
#     major, _ = parse_compute_version(compute_version)
#     if major >= 8:
#         return True

#     return False
