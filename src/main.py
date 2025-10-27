from pathlib import Path
import argparse
import re
import arrow
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84, EarthSatellite
from skyfield.constants import C
from utils.logger import logger
from datetime import timedelta

work_dir = Path(__file__).parent.parent
tle_data_dir = work_dir / 'data' / 'TLE'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--norad_id', type=int, required=True, help='Norad ID of the satellite')
    parser.add_argument('--min_delta_time', type=float, required=False, help='Minimum delta time')
    parser.add_argument('--max_delta_time', type=float, required=False, help='Maximum delta time')
    parser.add_argument('--tick', type=float, help='second')
    # 地面站经纬度
    parser.add_argument('--latitude', type=float, default=31.025, help='Observer latitude in degrees')
    parser.add_argument('--longitude', type=float, default=121.437, help='Observer longitude in degrees')
    parser.add_argument('--elevation', type=int, default=30, help='elevation')

    

    args = parser.parse_args()
    norad_id = args.norad_id
    min_delta_time = args.min_delta_time
    max_delta_time = args.max_delta_time
    print(f'Norad ID: {norad_id}')
    print(f'Minimum delta time: {min_delta_time}')
    print(f'Maximum delta time: {max_delta_time}')

    # ======================================================================================================================================= 文件加载

    sat_dir = tle_data_dir / f'{norad_id}'
    print(f'Path: {sat_dir}')
    if not sat_dir.exists():
        logger.info(f'No TLE file found for Norad ID {norad_id}')
        exit(1)
    
    # 遍历所有日期文件夹
    dates = []
    tle_files = []
    sats = []  # 创建卫星列表用于存储所有加载的卫星数据
    
    # 使用 glob 查找所有日期格式的子目录
    for date_dir in sat_dir.glob('*'):
        if date_dir.is_dir():
            # 检查目录名是否为日期格式 (YYYY-MM-DD)
            if re.match(r'\d{4}-\d{2}-\d{2}', date_dir.name):
                dates.append(date_dir.name)
                # 在日期目录下查找所有 .tle 文件
                for tle_file in date_dir.glob('*.tle'):
                    tle_files.append(tle_file)
    
    # 日期排序
    
    logger.info(f'Found {len(dates)} date directories and {len(tle_files)} TLE files for Norad ID {norad_id}')

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
                logger.warning(f'No satellites found in {tle_file}')
        except Exception as e:
            logger.error(f'Error loading {tle_file}: {e}')
    
    logger.info(f'Total satellites loaded: {len(sats)}')

    # ======================================================================================================================================= 创建计算条件

    ts = load.timescale() # 加载时间尺度

    ticks = args.tick # 时间步长

    lat = args.latitude
    long = args.longitude
    bluffton = wgs84.latlon(lat, long) # 根据经纬度确定地面站坐标
    elevation = args.elevation

    for item in sats:
        for sat in item:
            epoch = sat.epoch
            delta = timedelta(days=7)
            start_day = epoch - delta
            end_day   = epoch + delta
            print(epoch.utc_jpl())
            
            t, events = sats[0][0].find_events(bluffton, start_day, end_day, altitude_degrees=elevation)
            event_names = f'rise above {elevation}°', 'culminate', f'set below {elevation}°'
            for ti, event in zip(t, events):
                name = event_names[event]
                print(ti.utc_strftime('%Y %b %d %H:%M:%S'), name)

        
    
    
    

    
        
        
    