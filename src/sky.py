from skyfield.api import Time, EarthSatellite
from datetime import timedelta
from skyfield.toposlib import GeographicPosition
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
from pathlib import Path
from skyfield.vectorlib import VectorSum
from skyfield.constants import C
from utils.logger import logger
from skyfield.api import Angle
import pickle

from dataclasses import dataclass


@dataclass
class SatRecord:
    norad_id: int
    date: date
    datadir: Path
    sats: list[EarthSatellite]
    # events: list[dict[Angle, list[tuple[Time, Time, Time]]]]


class SkyfieldManager:
    def __init__(
        self,
        ground_station: GeographicPosition,
        timescale: Time,
        satellite: dict[int, list[SatRecord]],
    ):
        self.ground_station = ground_station
        self.timescale = timescale
        self.satellites = satellite
    

    def check_elevation(
        self,
        events: list[tuple[Time, Time, Time]],
        origin: list[tuple[Time, Time, Time]],
    )-> list[tuple[Time, Time, Time]]:
        res = [
            ori
            for ori in origin
            if not any(
                ori[0].tt < ev[0].tt and ori[2].tt > ev[2].tt
                for ev in events  # ori.begin > ev && ori.end < ev
            )
        ]
        return res
    
    def devide_events(
        self,
        origin: list[tuple[Time, Time, Time]],
        standard: list[tuple[Time, Time, Time]]
    ) -> list[tuple[Time, Time, Time]]:
        res = [
            sta
            for sta in standard
            if any(
                sta[0].tt < ev[0].tt and sta[0].tt > ev[0].tt
                for ev in origin
            )
        ]
        return res
        

    def get_sat_by_id_and_date(self, sat_id: int, date: date) -> list[SatRecord]:
        res = []
        for item in self.satellites[sat_id]:
            if item.date == date:
                res.append(item)
        return res

    def get_sat_by_id(self, sat_id: int) -> list[SatRecord]:
        return self.satellites[sat_id]
    
    def cal_maximum_ele(
        self,
        sat: EarthSatellite,
        t: Time
    ):
        vec = sat - self.ground_station
        return vec.at(t).altaz()[0]
    
    def print_delta_time(
        self,
        sat: EarthSatellite,
        path: Path,
        duration: timedelta,
        delta: float,
        minimum_elevation: int,
        delta_dis: float,
        elevation: list = [20, 40, 60, 80],
        is_print: bool = False,
    ):  
        all_events = self.generate_events(sat, duration, minimum_elevation)
        events_dict = {}
        max_time = 0
        for event in all_events:
            max_time = max(max_time, event[-1].tt - event[0].tt)
            ele = self.cal_maximum_ele(sat, event[1])
            val = events_dict.get(ele, [])
            val.append(event)
            events_dict[ele] = val

        vec = sat - self.ground_station
        flags = {}
        color_mapping= {}
        last = 0
        elevation.sort()
        # print(elevation)
        cnt = 0
        for d in elevation:
            flags[(last, d)] = True
            color_mapping[(last, d)] = cnt % 4
            cnt += 1
            last = d
        flags[(elevation[-1], 90)] = True
        color_mapping[(elevation[-1], 90)] = cnt % 4
        # print(flags)
        
        count = 0
        step_day = delta / 86400.0  # 100 ms -> 天
        days = np.arange(0, max_time + step_day / 2, step_day) * 86400  # 含端点

        for k, v in events_dict.items():
            for ti in v:
                curve = np.zeros(len(days))
                s = ti[0]
                en = ti[-1]
                end_dt = en.utc_datetime()
                mid = int(
                    (en.utc_datetime() - s.utc_datetime()) / 2
                    / timedelta(microseconds=1)
                )
                s_pt = s.utc_datetime() # 开始的指针
                e_pt = s.utc_datetime() # 之后的指针
                while s_pt <= end_dt and e_pt <= end_dt:
                    sat_e_t = self.timescale.from_datetime(e_pt)
                    sat_s_t = self.timescale.from_datetime(s_pt)
                    dis = vec.at(sat_e_t).altaz()[2].m - vec.at(sat_s_t).altaz()[2].m
                    dis = abs(dis)
                    logger.info(f"{s.utc_datetime()}-{end_dt}-{sat_s_t.utc_datetime()}-{sat_e_t.utc_datetime()}-{delta_dis}-{dis}")
                    if dis < delta_dis:
                        e_pt += timedelta(microseconds=1)
                    else:
                        idx = int(
                            (s_pt - s.utc_datetime())
                            / timedelta(microseconds=1)
                        )
                        idx += len(days) / 2 - mid
                        curve[int(idx)] = (sat_e_t.tt - sat_s_t.tt) * 86400000000 # 微秒
                        s_pt += timedelta(microseconds=1)
                        
                curve_masked = np.where(curve==0, np.nan, curve)
                


    def print_complete_process(
        self,
        sat: EarthSatellite,
        path: Path,
        duration: timedelta,
        delta: float,
        minimum_elevation: int,
        elevation: list = [80, 60, 40, 20],
        is_print: bool = False,
    ): # 画出卫星 大于某个仰角的事件的全过程
        plt.figure(figsize=(8, 4))
        colors = ["red", "green", "blue", "orange"]
        plt.xlabel("Time")
        plt.ylabel("Distance (km)")
        plt.title("Satellite-Ground Distance Over Time")
        plt.grid(True)
        plt.tight_layout()
        vec = sat - self.ground_station
        all_events = self.generate_events(sat, duration, minimum_elevation)
        events_dict = {}
        max_time = 0
        for event in all_events:
            max_time = max(max_time, event[-1].tt - event[0].tt)
            ele = self.cal_maximum_ele(sat, event[1])
            val = events_dict.get(ele, [])
            val.append(event)
            events_dict[ele] = val
        # logger.info(events_dict)
        
        flags = {}
        color_mapping= {}
        last = 0
        elevation.sort()
        # print(elevation)
        cnt = 0
        for d in elevation:
            flags[(last, d)] = True
            color_mapping[(last, d)] = colors[cnt % 4]
            cnt += 1
            last = d
        flags[(elevation[-1], 90)] = True
        color_mapping[(elevation[-1], 90)] = colors[cnt % 4]
        # print(flags)
        
        count = 0
        step_day = delta / 86400.0  # 100 ms -> 天
        days = np.arange(0, max_time + step_day / 2, step_day) * 86400  # 含端点
        days = np.append(days, days[-1] + step_day * 86400)
        for k, v in events_dict.items():
            for ti in v:
                curve = np.zeros(len(days))
                s = ti[0]
                en = ti[-1]
                current_dt = s.utc_datetime()
                end_dt = en.utc_datetime()
                mid = int(
                    (en.utc_datetime() - s.utc_datetime()) / 2
                    / timedelta(milliseconds=100)
                )
                while current_dt <= end_dt:
                    t_cur = self.timescale.from_datetime(current_dt)
                    pos = vec.at(t_cur)
                    _, _, dis = pos.altaz()
                    idx = int(
                        (current_dt - s.utc_datetime())
                        / timedelta(milliseconds=100)
                    )
                    idx += len(days) / 2 - mid
                    curve[int(idx)] = dis.km
                    current_dt += timedelta(milliseconds=100)
                curve_masked = np.where(curve == 0, np.nan, curve)
                for ki, vi in flags.items():
                    # logger.info(f"{ki[0]=}, {ki[1]=}, {k=}, {vi}")
                    if ki[0] < k.degrees and ki[1] >= k.degrees and vi:
                        plt.plot(days, curve_masked, color_mapping[ki], label=f"{ki[0]}-{ki[1]} degree")
                        flags[ki] = False
                    elif ki[0] < k.degrees and ki[1] >= k.degrees and (not vi):

                        plt.plot(days, curve_masked, color_mapping[ki])
        plt.legend()
        plt.savefig(
            path.parent.parent.parent
            / "fig"
            / f"{sat.model.satnum}"
            / f"{path.name}-complete.jpg"
        )

        if is_print:
            plt.show()

                
                # plt.plot(days, curve_masked, colors[count], label=f"{k} degree")
                
                #     plt.plot(days, curve_masked, colors[count])
            

    def print_distance_figure(
        self,
        sat: EarthSatellite,
        path: Path,
        duration: timedelta,
        delta: float,  # 计算每个点之间的时间间隔
        elevation: list = [80, 60, 40, 20],
        is_print: bool = False,
    ): # 只画出 卫星大于某一个仰角的事件开始到结束的片段
        plt.figure(figsize=(8, 4))
        colors = ["red", "green", "blue", "orange"]
        plt.xlabel("Time")
        plt.ylabel("Distance (km)")
        plt.title("Satellite-Ground Distance Over Time")
        plt.grid(True)
        plt.tight_layout()

        vec = sat - self.ground_station  # 卫星到地站的向量
        ele_events = []
        for ele in elevation:
            e = self.generate_events(
                sat, duration, ele
            )  # 生成-duration ~ duration内仰角大于ele的事件
            min_key = min(
                (key for d in ele_events for key in d if key >= ele), default=None
            )  # min_key
            if min_key == None:
                ele_events.append({ele: e})
            else:
                ele_events.append(
                    {
                        ele: self.check_elevation(
                            self.generate_events(sat, duration, min_key), e
                        )
                    }
                )  # 加入去重后的数据
        # logger.info(f"{ele_events=}")
        max_time = 0
        for item in ele_events:
            for k, v in item.items():
                # logger.info(f"{v=}")
                for ti in v:
                    max_time = max(max_time, ti[-1].tt - ti[0].tt)

        step_day = delta / 86400.0  # 100 ms -> 天
        days = np.arange(0, max_time + step_day / 2, step_day) * 86400  # 含端点

        count = 0
        curve_cnt = 0
        for item in ele_events:
            flag_show_legend = True
            for k, v in item.items():
                # logger.info(f"{k=}, {v=}")
                for ti in v:
                    curve = np.zeros(len(days))
                    s = ti[0]
                    en = ti[-1]
                    current_dt = s.utc_datetime()
                    end_dt = en.utc_datetime()
                    while current_dt <= end_dt:
                        t_cur = self.timescale.from_datetime(current_dt)
                        pos = vec.at(t_cur)
                        _, _, dis = pos.altaz()
                        idx = int(
                            (current_dt - s.utc_datetime())
                            / timedelta(milliseconds=100)
                        )
                        curve[idx] = dis.km
                        current_dt += timedelta(milliseconds=100)
                    curve_masked = np.where(curve == 0, np.nan, curve)
                    if flag_show_legend:
                        plt.plot(days, curve_masked, colors[count], label=f"{k} degree")
                        curve_cnt += 1
                        # logger.info(f"now-{curve_cnt=}")
                        flag_show_legend = False
                    else:
                        plt.plot(days, curve_masked, colors[count])
                        # logger.info(f"here-{curve_cnt=}")
                        curve_cnt += 1
            count += 1
        # logger.info(f"{curve_cnt=}")
        plt.legend()
        plt.savefig(
            path.parent.parent.parent
            / "fig"
            / f"{sat.model.satnum}"
            / f"{path.name}.jpg"
        )

        if is_print:
            plt.show()

    def generate_events(
        self,
        sat: EarthSatellite,
        duration: timedelta,
        elvation: int,
    )-> list[tuple[Time, Time, Time]]:
        start_day = sat.epoch - duration
        end_day = sat.epoch + duration

        t, events = sat.find_events(
            self.ground_station, start_day, end_day, altitude_degrees=elvation
        )
        item = []
        all_events = []
        for ti, event in zip(t, events):
            item.append(ti)
            if event == 2:
                all_events.append(item)
                item = []

        return all_events

    def _check(
        self, begin: Time, end: Time, delta: int, vec: VectorSum, delta_dis: float
    ) -> bool:
        logger.info(delta)
        step = timedelta(microseconds=delta)
        logger.info(f"{step=}")
        current_dt = begin.utc_datetime()
        end_dt = end.utc_datetime()

        while current_dt <= end_dt:
            t_cur = self.timescale.from_datetime(current_dt)
            t_next = self.timescale.from_datetime(current_dt + step)
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

    def __binary_search(
        self,
        vec: VectorSum,
        start_time: Time,
        end_time: Time,
        delta_dis: float,
    ):
        left = 0
        right = 1000000  # microseconds
        while left < right:
            mid = int((left + right) / 2)
            if self._check(start_time, end_time, mid, vec, delta_dis):
                left = mid
            else:
                right = mid - 1
        logger.info(mid)

    def search_minimum_delta_time(
        self,
        sat: EarthSatellite,
        start_time: Time,
        end_time: Time,
        carry_frequency: float,
        phase: float = 0.5,
    ):  # 搜索最小的时间间隔
        delta_dis = (
            1 / carry_frequency * phase * C
        )  # 固定 距离 delta_dis 搜索最小的 delta_t
        logger.info(f"delta distance: {delta_dis}")
        vec: VectorSum = sat - self.ground_station  # 卫星到地面站的向量
        self.__binary_search(vec, start_time, end_time, delta_dis)
