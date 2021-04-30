#  NanoVNASaver - a python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019.  Rune B. Broberg
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

try:
    import pkg_resources.py2_warn
except ImportError:
    pass
from NanoVNASaver.testscript import main
import logging

if __name__ == '__main__':

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)

    testscriptLogger = logging.getLogger("testscript")
    testscriptLogger.addHandler(consoleHandler)
    testscriptLogger.setLevel(logging.DEBUG)

    testscriptLogger.info('running main')
    main()
