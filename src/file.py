from pathlib import Path
import re
from skyfield.api import load
from utils.logger import logger
from sky import SatRecord
from datetime import date


work_dir = Path(__file__).parent.parent
tle_data_dir = work_dir / "data" / "TLE"
fig_data_dir = work_dir / "data" / "fig"


class InputManager:
    def __init__(self, norad_id_list: list[int]):
        self.norad_id_list = norad_id_list

    def create_fig_dir(self):
        if not fig_data_dir.exists():
            fig_data_dir.mkdir()
            logger.info("create figure directory")
        for id in self.norad_id_list:
            if not (fig_data_dir / f"{id}").exists():
                (fig_data_dir / f"{id}").mkdir()

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
    
    def output_fig():
        pass
