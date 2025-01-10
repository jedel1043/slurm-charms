#!/usr/bin/env python3
# Copyright 2025 Canonical Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mock source modules available only in production environment."""

import sys
import types
from unittest.mock import Mock

module_name = "apt_pkg"
apt_pkg_mock = types.ModuleType(module_name)
sys.modules[module_name] = apt_pkg_mock
apt_pkg_mock.init = Mock(name=module_name + ".init")
apt_pkg_mock.Cache = Mock(name=module_name + ".Cache")

module_name = "UbuntuDrivers"
drivers_mock = types.ModuleType(module_name)
detect_mock = types.ModuleType(module_name + ".detect")
sys.modules[module_name] = drivers_mock
sys.modules[module_name + ".detect"] = detect_mock
drivers_mock.detect = detect_mock
detect_mock.system_gpgpu_driver_packages = Mock(
    name=module_name + ".detect.system_gpgpu_driver_packages"
)
detect_mock.get_linux_modules_metapackage = Mock(
    name=module_name + ".detect.get_linux_modules_metapackage"
)
