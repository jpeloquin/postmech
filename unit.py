import re

import pint
from uncertainties import ufloat

# packages must use pint.get_application_registry to integrate properly with user code
ureg = pint.get_application_registry()
ureg.define("pixel = [distance] = px")


def parse(s: str) -> ureg.Quantity:
    """Parse a value with uncertainty and units, e.g. '1 ± 0.5 mm'"""
    m = re.match(
        r"(?P<value>[0-9.]+)\s*(?:±|\+/-)\s*(?P<uncertainty>[0-9.]+)?\s*(?P<units>\S+)?",
        s,
    )
    if m.group("uncertainty") is None:
        return float(m.group("value")) * ureg(m.group("units"))
    else:
        return ufloat(float(m.group("value")), float(m.group("uncertainty"))) * ureg(
            m.group("units")
        )
