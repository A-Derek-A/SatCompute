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


work_dir = Path(__file__).parent.parent
tle_data_dir = work_dir / "data" / "TLE"


def check1(begin: Time,
           end: Time,
           delta: int,
           vec: VectorSum,
           delta_dis: float)->bool:
    logger.info(delta)
    step = timedelta(microseconds=delta)
    logger.info(f"{step=}")
    current_dt = begin.utc_datetime()
    end_dt = end.utc_datetime()

    while current_dt <= end_dt:
        t_cur = ts.from_datetime(current_dt)
        t_next = ts.from_datetime(current_dt + step)
        rx_pos_cur = vec.at(t_cur)
        rx_pos_next = vec.at(t_next)

        _, _, dis_cur = rx_pos_cur.altaz()
        _, _, dis_next = rx_pos_next.altaz()

        dis_cur_m = dis_cur.m  # 转成米
        dis_next_m = dis_next.m

        if abs(dis_next_m - dis_cur_m) > delta_dis:
            logger.info("False")
            return False

        current_dt += step

    logger.info("True")
    return True

def check(begin: Time,
          end: Time,
          delta: int, # microseconds
          vec: VectorSum,
          delta_dis: float
        ) -> bool:
    logger.info(delta)
    step_days = delta / 1000000 / 86400.0  # 将毫秒转为天数
    logger.info(f"{step_days=}")
    t_cur = begin
    
    logger.info(f"{t_cur=}")
    logger.info(f"{t_cur + step_days}")

    while t_cur.tt <= end.tt:
        t_next = t_cur + step_days

        rx_pos_next = vec.at(t_next)
        rx_pos_cur = vec.at(t_cur)

        alt_next, az_next, dis_next = rx_pos_next.altaz()
        alt_cur, az_cur, dis_cur = rx_pos_cur.altaz()

        if abs(dis_next.m - dis_cur.m) > delta_dis:  # .m 转为米
            logger.info("False")
            return False

        t_cur = t_next
        
    logger.info("True")
    return True

def binary_search(
    vec: VectorSum,
    begin: Time,
    end: Time,
    delta_dis: float,
):
    left = 0 
    right = 1000000 # microseconds
    while(left < right):
        mid = (left + right) / 2
        if check1(begin, end, mid, vec, delta_dis):
            left = mid
        else:
            right = mid - 1
    logger.info(mid)

def print_dis(
    start_time: Time,
    end_time: Time,
    ground_station: GeographicPosition,
    sat: EarthSatellite,
):
    vec: VectorSum = sat - ground_station # 卫星到地面站的向量
    delta = timedelta(milliseconds=100)
    
    current_dt = start_time.utc_datetime()
    end_dt = end_time.utc_datetime()

    times = []
    distances = []

    while(current_dt <= end_dt):
        t_cur = ts.from_datetime(current_dt)
        pos = vec.at(t_cur)
        _, _, dis = pos.altaz()
        times.append(current_dt)
        distances.append(dis.km)
        current_dt += delta

    print(start_time.utc_datetime(), end_time.utc_datetime())
    print(times)
    print(distances)

    plt.figure(figsize=(8, 4))
    plt.plot(times, distances, label='Satellite–Ground Distance (km)')
    plt.xlabel("Time (UTC)")
    plt.ylabel("Distance (km)")
    plt.title("Satellite-Ground Distance Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def func(
    sat: EarthSatellite,
    ground_station: GeographicPosition,
    start_time: Time,
    end_time: Time,
    carry_frequency: float,
    phase: float = 0.5,
):
    delta_dis = 1 / carry_frequency  * phase * C # 固定 距离 delta_dis 搜索最小的 delta_t
    logger.info(f"delta distance: {delta_dis}")
    vec: VectorSum = sat - ground_station # 卫星到地面站的向量
    
    binary_search(vec, start_time, end_time, delta_dis)


    # ticks = np.arange(0, end_time.tt - start_time.tt, delta_t)
    # for _tick in tqdm(ticks):
    #     rx_time: Time = start_time + timedelta(seconds=_tick)
    #     rx_pos: ICRF = vec.at(rx_time)
    #     alt, az, distance = rx_pos.altaz()
    #     print(
    #         f"{rx_time.utc_jpl()} alt: {alt.degrees:.2f}°, az: {az.degrees:.2f}°, distance: {distance.km:.2f} km"
    #     )
    #     rx_range = distance.km


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--norad_id", type=int, required=True, help="Norad ID of the satellite"
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
    print(f"Norad ID: {norad_id}")
    print(f"Minimum delta time: {min_delta_time}")
    print(f"Maximum delta time: {max_delta_time}")

    # ======================================================================================================================================= 文件加载

    sat_dir = tle_data_dir / f"{norad_id}"
    print(f"Path: {sat_dir}")
    if not sat_dir.exists():
        logger.info(f"No TLE file found for Norad ID {norad_id}")
        exit(1)

    # 遍历所有日期文件夹
    dates = []
    tle_files = []
    sats = []  # 创建卫星列表用于存储所有加载的卫星数据

    # 使用 glob 查找所有日期格式的子目录
    for date_dir in sat_dir.glob("*"):
        if date_dir.is_dir():
            # 检查目录名是否为日期格式 (YYYY-MM-DD)
            if re.match(r"\d{4}-\d{2}-\d{2}", date_dir.name):
                dates.append(date_dir.name)
                # 在日期目录下查找所有 .tle 文件
                for tle_file in date_dir.glob("*.tle"):
                    tle_files.append(tle_file)

    # 日期排序

    logger.info(
        f"Found {len(dates)} date directories and {len(tle_files)} TLE files for Norad ID {norad_id}"
    )

    # 遍历所有找到的 TLE 文件并加载卫星数据
    for tle_file in tle_files:
        # logger.info(f'Processing TLE file: {tle_file}')
        # 加载 TLE 文件并将卫星数据添加到 Sats 列表
        try:
            es = load.tle_file(str(tle_file))
            if es:
                sats.append(es)
                # logger.info(f'Loaded {len(es)} satellites from {tle_file}')
            else:
                logger.warning(f"No satellites found in {tle_file}")
        except Exception as e:
            logger.error(f"Error loading {tle_file}: {e}")

    logger.info(f"Total satellites loaded: {len(sats)}")

    # ======================================================================================================================================= 创建计算条件

    ts = load.timescale()  # 加载时间尺度

    tick = args.tick  # 时间步长

    lat = args.latitude
    long = args.longitude
    ground_station = wgs84.latlon(lat, long)  # 根据经纬度确定地面站坐标
    elevation = args.elevation

    test_sat = sats[0][0]
    start_day = test_sat.epoch - timedelta(days=7)
    end_day = test_sat.epoch + timedelta(days=7)
    t, events = test_sat.find_events(
        ground_station, start_day, end_day, altitude_degrees=elevation
    )

    print_dis(t[0], t[2], ground_station, test_sat)
    # func(test_sat, ground_station, start_time=t[0], end_time=t[1], carry_frequency=9600000000, phase=0.5)

    # for item in sats:
    #     for sat in item:
    #         print(type(sat))
    #         epoch = sat.epoch
    #         delta = timedelta(days=7)
    #         start_day = epoch - delta
    #         end_day   = epoch + delta
    #         # print(epoch.utc_jpl())

    #         t, events = sat.find_events(ground_station, start_day, end_day, altitude_degrees=elevation)
    #         event_names = f'rise above {elevation}°', f'culminate', f'set below {elevation}°'
    #         for ti, event in zip(t, events):
    #             name = event_names[event]
    #             # print(ti.utc_strftime('%Y %b %d %H:%M:%S'), name)
    #             if event == 0:
    #                 pass
    #             elif event == 1:
    #                 pass
    #             elif event == 2:
    #                 pass
