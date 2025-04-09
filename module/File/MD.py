import os
import re

from base.Base import Base
from module.Cache.CacheItem import CacheItem

class MD(Base):
    
    # 添加用于匹配 Markdown 图片语法的正则表达式
    RE_MD_IMAGE = re.compile(r"!\[.*?\]\(.*?\)")

    # 修正用于匹配数学公式的正则表达式 - 先匹配双美元符号再匹配单美元符号
    #RE_MATH_FORMULA = re.compile(r"\$\$(.*?)\$\$|\$(.*?)\$", re.DOTALL)

    def __init__(self, config: dict) -> None:
        super().__init__()

        # 初始化
        self.config: dict = config
        self.input_path: str = config.get("input_folder")
        self.output_path: str = config.get("output_folder")
        self.source_language: str = config.get("source_language")
        self.target_language: str = config.get("target_language")

    # 在扩展名前插入文本
    def insert_target(self, path: str) -> str:
        root, ext = os.path.splitext(path)
        return f"{root}.{self.target_language.lower()}{ext}"

    # 在扩展名前插入文本
    def insert_source_target(self, path: str) -> str:
        root, ext = os.path.splitext(path)
        return f"{root}.{self.source_language.lower()}.{self.target_language.lower()}{ext}"

    # 读取
    def read_from_path(self, abs_paths: list[str]) -> list[CacheItem]:
        items = []
        
        for abs_path in set(abs_paths):
            # 获取相对路径
            rel_path = os.path.relpath(abs_path, self.input_path)

            # 数据处理
            with open(abs_path, "r", encoding = "utf-8-sig") as reader:
                in_code_block = False  # 跟踪是否在代码块内
                for line in [line.removesuffix("\n") for line in reader.readlines()]:
                    # 检查是否是代码块的开始或结束
                    if line.startswith("```"):
                        in_code_block = not in_code_block
                        
                    # 确定是否应该排除此行（不翻译）
                    should_exclude = (
                        in_code_block or                             # 在代码块内
                        line.startswith("```") or                    # 代码块标记行
                        line.startswith("!(data:image/jpeg;base64") or  # base64图片数据
                        self.RE_MD_IMAGE.search(line) is not None  
                        #self.RE_MATH_FORMULA.search(line) is not None   
                    )
                    
                    status = Base.TranslationStatus.EXCLUDED if should_exclude else Base.TranslationStatus.UNTRANSLATED
                    
                    items.append(
                        CacheItem({
                            "src": line,
                            "dst": line,
                            "row": len(items),
                            "file_type": CacheItem.FileType.MD,
                            "file_path": rel_path,
                            "text_type": CacheItem.TextType.MD,
                            "status": status,  # 设置翻译状态
                        })
                    )

        return items

    # 写入
    def write_to_path(self, items: list[CacheItem]) -> None:
        # 筛选
        target = [
            item for item in items
            if item.get_file_type() == CacheItem.FileType.MD
        ]

        # 按文件路径分组
        data: dict[str, list[str]] = {}
        for item in target:
            data.setdefault(item.get_file_path(), []).append(item)

        # 分别处理每个文件
        for rel_path, items in data.items():
            abs_path = os.path.join(self.output_path, rel_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok = True)
            with open(self.insert_target(abs_path), "w", encoding = "utf-8") as writer:
                writer.write("\n".join([item.get_dst() for item in items]))
