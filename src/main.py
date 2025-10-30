from pathlib import Path
import argparse
import re
import arrow
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84, EarthSatellite, Time
from skyfield.toposlib import GeographicPosition
from skyfield.vectorlib import VectorSum
from skyfield.positionlib import ICRF
from skyfield.constants import C
from utils.logger import logger
from datetime import timedelta
import numpy as np
from tqdm import tqdm
from sky import SkyfieldManager
from file import InputManager
from datetime import date

work_dir = Path(__file__).parent.parent
tle_data_dir = work_dir / "data" / "TLE"
fig_data_dir = work_dir / "data" / "fig"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--norad_id", type=int, required=True, help="Norad ID of the satellite"
    )
    parser.add_argument(
        "--auto_detect", type=bool, default=False, help="Auto detect the norad id"
    )
    parser.add_argument(
        "--min_delta_time", type=float, required=False, help="Minimum delta time"
    )
    parser.add_argument(
        "--max_delta_time", type=float, required=False, help="Maximum delta time"
    )
    parser.add_argument("--tick", type=float, help="second")
    # 地面站经纬度
    parser.add_argument(
        "--latitude", type=float, default=31.025, help="Observer latitude in degrees"
    )
    parser.add_argument(
        "--longitude", type=float, default=121.437, help="Observer longitude in degrees"
    )
    parser.add_argument("--elevation", type=int, default=30, help="elevation")

    args = parser.parse_args()
    norad_id = args.norad_id
    min_delta_time = args.min_delta_time
    max_delta_time = args.max_delta_time

    # 文件加载
    if type(norad_id) != list[int]:
        norad_id = [norad_id]
    input_manager = InputManager(norad_id)
    input_manager.create_fig_dir()
    sats = input_manager.load_all_tle()

    # 初始化参数
    ts = load.timescale()  # 加载时间尺度
    tick = args.tick  # 时间步长
    lat = args.latitude
    long = args.longitude
    ground_station = wgs84.latlon(lat, long)  # 根据经纬度确定地面站坐标

    skymanager = SkyfieldManager(ground_station, ts, sats)
    sat_records = skymanager.get_sat_by_id_and_date(57582, date=date(2024, 1, 27))
    # logger.info(sat_records)
    
    for item in sat_records:
        for s in item.sats:
            skymanager.print_distance_figure(
                s, item.datadir, duration=timedelta(days=1), delta=tick
            )
