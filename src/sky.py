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

from dataclasses import dataclass


@dataclass
class SatRecord:
    norad_id: int
    date: date
    datadir: Path
    sats: list[EarthSatellite]


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
    ):
        res = [
            ori
            for ori in origin
            if not any(
                ori[0].tt < ev[0].tt and ori[2].tt > ev[2].tt
                for ev in events  # ori.begin > ev && ori.end < ev
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

    def print_distance_figure(
        self,
        sat: EarthSatellite,
        path: Path,
        duration: timedelta,
        delta: float,  # 计算每个点之间的时间间隔
        elevation: list = [80, 60, 40, 20],
        is_print: bool = False,
    ):
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
        max_time = 0
        for item in ele_events:
            for k, v in item.items():
                for ti in v:
                    max_time = max(max_time, ti[-1].tt - ti[0].tt)

        step_day = delta / 86400.0  # 100 ms -> 天
        days = np.arange(0, max_time + step_day / 2, step_day) * 86400  # 含端点

        count = 0
        for item in ele_events:
            flag_show_legend = True
            for k, v in item.items():
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
                        flag_show_legend = False
                    else:
                        plt.plot(days, curve_masked, colors[count])
            count += 1
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
    ):
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
