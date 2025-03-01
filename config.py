#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os


class DefaultConfig:
    """Bot Configuration"""

    PORT = 8000
    APP_ID = os.environ.get("b0a29017-ea3f-4697-aef7-0cb05979d16c", "")
    APP_PASSWORD = os.environ.get("2fc8Q~YUZMbD8E7hEb4.vQoDFortq3Tvt~CLCcEQ", "")
