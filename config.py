#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# ss under the MIT License.

import os


class DefaultConfig:
    """Bot Configuration"""

    PORT = 3978
    APP_ID = os.environ.get("MicrosoftAppId", "1c5524dc-0c09-4925-bca8-d633afc9c070")
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "Kca8Q~mJIubZW7vwUSD2bG1SPQ4_FCWARDezPa31")
