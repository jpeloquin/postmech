import re

import pint
from uncertainties import ufloat


def fix_pint_registry(ureg: pint.registry.UnitRegistry):
    # Workaround for https://github.com/hgrecco/pint/issues/1304
    redef_oaction = ureg._on_redefinition
    ureg._on_redefinition = "ignore"
    # Pixels should not have a fixed physical length by default
    # https://github.com/hgrecco/pint/issues/1396
    ureg.define("@alias pixel = px")
    assert len(ureg("px").compatible_units()) == 0
    ureg._on_redefinition = redef_oaction
    return ureg


# packages must use pint.get_application_registry to integrate properly with user code
# Need to use .get() as of 0.18, see https://github.com/hgrecco/pint/issues/1568
ureg = fix_pint_registry(pint.get_application_registry().get())


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
