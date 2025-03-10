import os

import rapidjson as json

from base.Base import Base
from module.Cache.CacheItem import CacheItem

class MESSAGEJSON(Base):

    # [
    #     {
    #         "name", "しますか",
    #         "message": "<fgName:pipo-fog004><fgLoopX:1><fgLoopY:1><fgSx:-2><fgSy:0.5>"
    #     },
    #     {
    #         "message": "エンディングを変更しますか？"
    #     },
    #     {
    #         "message": "はい"
    #     },
    # ]

    def __init__(self, config: dict) -> None:
        super().__init__()

        # 初始化
        self.config: dict = config
        self.input_path: str = config.get("input_folder")
        self.output_path: str = config.get("output_folder")
        self.source_language: str = config.get("source_language")
        self.target_language: str = config.get("target_language")

    # 读取
    def read_from_path(self, abs_paths: list[str]) -> list[CacheItem]:
        items = []
        for abs_path in set(abs_paths):
            # 获取相对路径
            rel_path = os.path.relpath(abs_path, self.input_path)

            # 数据处理
            with open(abs_path, "r", encoding = "utf-8-sig") as reader:
                json_data: list[dict] = json.load(reader)

                # 格式校验
                if not isinstance(json_data, list):
                    continue

                for v in json_data:
                    if isinstance(v, dict) and "message" in v:
                        items.append(
                            CacheItem({
                                "src": v.get("message", ""),
                                "dst": v.get("message", ""),
                                "extra_field": v.get("name", None),
                                "row": len(items),
                                "file_type": CacheItem.FileType.MESSAGEJSON,
                                "file_path": rel_path,
                            })
                        )

        return items

    # 写入
    def write_to_path(self, items: list[CacheItem]) -> None:
        target = [
            item for item in items
            if item.get_file_type() == CacheItem.FileType.MESSAGEJSON
        ]

        data: dict[str, list[str]] = {}
        for item in target:
            data.setdefault(item.get_file_path(), []).append(item)

        for rel_path, items in data.items():
            abs_path = os.path.join(self.output_path, rel_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok = True)

            result = []
            for item in items:
                if item.get_extra_field() is None:
                    result.append({
                        "message": item.get_dst(),
                    })
                else:
                    result.append({
                        "name": item.get_extra_field(),
                        "message": item.get_dst(),
                    })

            with open(abs_path, "w", encoding = "utf-8") as writer:
                writer.write(json.dumps(result, indent = 4, ensure_ascii = False))