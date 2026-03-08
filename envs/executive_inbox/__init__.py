# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Executive Inbox Environment."""

from .models import ExecutiveInboxAction, ExecutiveInboxObservation

# Allow server-side imports to work even when optional MCP/client deps are missing.
try:
    from .client import ExecutiveInboxEnv
except ModuleNotFoundError:
    ExecutiveInboxEnv = None  # type: ignore[assignment]

__all__ = [
    "ExecutiveInboxAction",
    "ExecutiveInboxObservation",
]

if ExecutiveInboxEnv is not None:
    __all__.append("ExecutiveInboxEnv")
