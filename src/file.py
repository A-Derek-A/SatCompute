from pathlib import Path
import re
from skyfield.api import load
from utils.logger import logger
from sky import SatRecord
from datetime import date
import matplotlib.pyplot as plt
from skyfield.api import Angle, Time
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter, FixedLocator


work_dir = Path(__file__).parent.parent

tle_data_dir = work_dir / "data" / "TLE"
fig_data_dir = work_dir / "data" / "fig"
check_data_dir = work_dir / "data" / "checkpoint"
curve_data_dir = work_dir / "data" / "curve"


class InputManager:
    def __init__(
        self, 
        norad_id_list: list[int],
        timescale: Time
    ):
        self.norad_id_list = norad_id_list
        self.timescale = timescale

    def create_fig_dir(self):
        if not fig_data_dir.exists():
            fig_data_dir.mkdir()
            logger.info("create figure directory")
        for id in self.norad_id_list:
            if not (fig_data_dir / f"{id}").exists():
                (fig_data_dir / f"{id}").mkdir()
            if not (check_data_dir / f"{id}").exists():
                (check_data_dir / f"{id}").mkdir()
            if not (curve_data_dir / f"{id}").exists():
                (curve_data_dir / f"{id}").mkdir()

    def load_all_tle(self) -> dict[int, list[SatRecord]]:
        res = {}
        for id in self.norad_id_list:
            res[id] = []
            sat_dir = tle_data_dir / f"{id}"
            if not sat_dir.exists():
                logger.info(f"No TLE file found for Norad ID {id}")
                continue
            # 遍历所有日期文件夹
            for date_dir in sat_dir.glob("*"):
                if date_dir.is_dir():
                    # 检查目录名是否为日期格式 (YYYY-MM-DD)
                    if re.match(r"\d{4}-\d{2}-\d{2}", date_dir.name):
                        # 在日期目录下查找所有 .tle 文件
                        # logger.info(date_dir.name)
                        sr = SatRecord(
                            id, date.fromisoformat(date_dir.name), date_dir, []
                        )  # TODO: date 类型
                        for tle_file in date_dir.glob("*.tle"):
                            # 加载 TLE 文件并将卫星数据添加到 Sats 列表
                            try:
                                es = load.tle_file(str(tle_file))
                                if es:
                                    sr.sats.append(es[0])
                                    # logger.info(f'Loaded {len(es)} satellites from {tle_file}')
                                else:
                                    logger.warning(f"No satellites found in {tle_file}")
                            except Exception as e:
                                logger.error(f"Error loading {tle_file}: {e}")
                        res[id].append(sr)
            logger.info(f"Loaded {len(res[id])} date directories for Norad ID {id}")
        logger.info(f"Total satellites loaded: {len(res)}")
        return res

    def print_figure(
        self,
        xdata: list,
        ydata: dict,
        norad_id: int,
        date_name: str,
        xlabel: str | None,
        ylabel: str | None,
        figure_title: str,
        is_log: bool = True,
        is_grid: bool = True,
        ylim: tuple[float, float] | None = (1, 1e4),
        figure_size: tuple[int, int] = (12, 8),
        is_print: bool = False,
        elevation: list[float] = [20, 40, 60, 80],
    ):
        
        # print(type(xdata), type(ydata), type(norad_id), type(date_name), type(xlabel), type(ylabel), type(figure_title))
        print(type(figure_size))
        print(figure_size)
        colors = ["red", "green", "blue", "orange"]
        flags = {k: True for k in ydata.keys()}
        logger.info(f"{flags=}")
        elevation.append(90)
        elevation.sort()
        plt.figure(figsize=figure_size)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if figure_title is not None:
            plt.title(figure_title)
        
        if is_log:
            ax = plt.gca()
            plt.yscale('log')
            # 主刻度：10⁰, 10¹, 10²...（默认即可）
            # ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
            # 次刻度：每个数量级显示2,3,4...9
            # ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
            custom_ticks = np.concatenate([
                np.arange(2, 10, 2),           # 1-10阶段：2,3,4,5,6,7,8,9
                np.arange(20, 100, 20),     # 10-100阶段：20,30,40,50,60,70,80,90
                [500],                      # 100-1000阶段：仅500
                # 1000以上：不添加任何值
            ])
            # 3. 设置次刻度位置（FixedLocator）
            ax.yaxis.set_minor_locator(FixedLocator(custom_ticks))
            # 4. 格式化标签（显示为整数）
            ax.yaxis.set_minor_formatter(ScalarFormatter())
            # 5. 主刻度保持默认（1,10,100,1000,10000）
            ax.yaxis.set_major_formatter(ScalarFormatter())
            plt.tight_layout()
        if is_grid:
            plt.grid(True, which="both", ls="--", alpha=0.6)
        if ylim is not None:
            plt.ylim(ylim)
        plt.tight_layout()
        for k, v in ydata.items():
            for item in v:
                if flags[k]:
                    logger.info(f"{len(item)=}")
                    logger.info(f"{len(xdata)=}")
                    plt.plot(xdata, item, colors[k], label=f"{elevation[k]}-{elevation[k + 1]} degree")
                    flags[k] = False
                else:
                    plt.plot(xdata, item, colors[k])
        plt.legend()
        plt.savefig(fig_data_dir / f"{norad_id}" / f"{date_name}-{figure_title}.jpg")
        if is_print:
            plt.show()

    def handle_date(
        self,
        data_path: Path,
        norad_id: int,
        elevation: list[float] = [20, 40, 60, 80],
    ):
        
        data = {}
        color_mapping = {}
        last = 0
        elevation.sort()
        cnt = 0
        for d in elevation:
            color_mapping[(last, d)] = cnt % 4
            cnt += 1
            last = d

        color_mapping[(elevation[-1], 90)] = cnt % 4
        max_time = 0
        for item in data_path.glob("*.npz"):
            degree, time = item.name.split("-")
            print(time)
            time, num, _ = time.split(".")
            time = float(time + "." + num)
            degree = float(degree)
            deg = Angle(degrees=degree)
            time = self.timescale.tt_jd(time)
            for ki, vi in color_mapping.items():
                if ki[0] < deg.degrees <= ki[1]:
                    with open(item, "rb") as f:
                        npz = np.load(f)
                        curve_masked = npz["arr_0"]
                        max_time = max(max_time, len(curve_masked))
                        data.setdefault(vi, []).append(curve_masked)
        _, first_value = next(iter(data.items()))
        sample_point = len(first_value[0]) - 1
        logger.info(f"{sample_point=}")
        days = np.arange(0, max_time, max_time / sample_point) * 86400
        days = np.append(days, days[-1] + max_time / sample_point * 86400)

        logger.info(f"handle-{len(days)=}")
        print(type(data))
        print(data_path.parent.name)
        self.print_figure(
            days,
            data,
            norad_id,
            data_path.parent.name,
            "Time (seconds)",
            "Delta Time (us)",
            "Delta-Time-Over-Time",
        )
            
            
if __name__ == "__main__":
    timescale = load.timescale()
    input_manager = InputManager([57582], timescale)
    input_manager.create_fig_dir()
    sats = input_manager.load_all_tle()
    input_manager.handle_date(curve_data_dir / f"{57582}" / "2024-01-27")
