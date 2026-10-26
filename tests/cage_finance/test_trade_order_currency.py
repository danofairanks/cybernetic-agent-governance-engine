# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A trade order's currency is a three-letter, ISO 4217-shaped code.

Follow-up to GHSA-f6xq-23f7-9grm: the policy decides which currencies are
tradable (``tradable_currencies`` in ``trade_governance.rego``); the order model
refuses a spelling that is no currency code at all, so a look-alike never
reaches the broker.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.cage_finance.models.trade_order import TradeOrder

pytestmark = [pytest.mark.unit, pytest.mark.local]


def _order(currency: str) -> TradeOrder:
    return TradeOrder(
        symbol="AAPL", amount=10, currency=currency, side="buy", confidence=0.99
    )


@pytest.mark.parametrize(
    "raw", ["USD", "usd", " Usd "], ids=["canonical", "lower", "padded"]
)
def test_currency_is_normalised(raw: str) -> None:
    assert _order(raw).currency == "USD"


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "US", "USDT", "US D", "U$D", "US\u200bD", "\u0423SD", "840"],
    ids=[
        "empty",
        "blank",
        "short",
        "long",
        "inner-space",
        "punctuation",
        "zero-width",
        "cyrillic-u",
        "numeric",
    ],
)
def test_malformed_currency_is_refused(raw: str) -> None:
    with pytest.raises(ValidationError, match="ISO 4217"):
        _order(raw)
