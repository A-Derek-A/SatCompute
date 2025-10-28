# from skyfield.api import EarthSatellite
# from skyfield import vectorlib
# from skyfield.api import utc, load, wgs84, EarthSatellite, Time
# import numpy as np
# from rich.table import Table
# from rich.console import Console
# import math


# GM = 3.98589196e14
# EPH = load("de421.bsp")
# EARTH: vectorlib.VectorSum = EPH["earth"]
# console = Console()

# class SatByTLE:
#     def __init__(self, es: EarthSatellite):
#         self.es_gcrs = es
#         self.es_bcrs: vectorlib.VectorSum = es + EARTH
#         self.epoch: Time = es.epoch
#         self.period = 2 * np.pi / es.model.no_kozai * 60  # seconds
#         self.current_time: Time = timescale.now()
#         self.lat, self.lon, self.height = self.pos_at(self.epoch)

#     def __repr__(self) -> str:
#         table = Table(title="Sat")
#         table.show_lines = True
#         table.add_column("Attr", header_style=None, style="magenta")
#         table.add_column("Value", header_style=None, style="cyan")
#         table.add_row("es", self.es_gcrs.arrow_str())
#         table.add_row("satnum", str(self.es_gcrs.model.satnum))
#         table.add_row("es_bcrs", self.es_bcrs.__str__())
#         table.add_row("epoch (utc_jpl)", self.epoch.utc_jpl())
#         table.add_row(
#             "position (lat, lon, height)",
#             f"{self.lat} degrees, {self.lon} degrees, {self.height} km",
#         )
#         console.print(table, justify="left")
#         return ""

#     def pos_at(self, time: Time) -> tuple[float, float, np.float64]:
#         """calculate (lat, lon, height) of the satellite at a given time (utc)

#         Args:
#             time (Time): skfield.timelib.Time

#         Returns:
#             lat (°), lon (°), height (km)
#         """
#         lat, lon = wgs84.latlon_of(self.es_gcrs.at(time))
#         lat = math.degrees(lat.radians)
#         lon = math.degrees(lon.radians)
#         height = wgs84.height_of(self.es_gcrs.at(time)).km
#         return lat, lon, height

# class SatByConstruct:
#     """
#     用轨道六根数构造一颗 Skyfield 卫星
#     参数:
#         a       : 半长轴 [km]
#         e       : 偏心率
#         i       : 倾角 [deg]
#         raan    : 升交点赤经 [deg]
#         argp    : 近地点幅角 [deg]
#         nu      : 真近点角 [deg]
#         epoch   : datetime（UTC）
#         bstar   : SGP4 阻尼系数（可选，默认 0）
#         ndot    : 平均运动一阶导（可选，默认 0）
#         nddot   : 平均运动二阶导（可选，默认 0）
#     """
#     def __init__(self, a, e, i, raan, argp, nu,
#                  epoch: Time = None,
#                  bstar=0.0, ndot=0.0, nddot=0.0):
#         self.a    = a
#         self.e    = e
#         self.i    = i
#         self.raan = raan
#         self.argp = argp
#         self.nu   = nu
#         self.epoch = epoch or datetime.now(timezone.utc)
#         self.bstar = bstar
#         self.ndot  = ndot
#         self.nddot = nddot

#         # 内部生成 EarthSatellite
#         self._sat = self._make_earth_satellite()
