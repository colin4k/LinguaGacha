import re

from base.Base import Base
from module.Cache.CacheItem import CacheItem

class CodeSaver(Base):

    # 通用规则
    RE_NONE: tuple[str] = (
        r"\s",                                                                      # 空白字符
        r"\u3000",                                                                  # 全角空格
        # r"\\n",                                                                   # 换行符 \n
        r"<br>",                                                                    # 换行符 <br>
    )

    # 用于 RenPy 的规则
    RE_RENPY: tuple[str] = (
        r"\{[^{}]*\}",                                                              # {w=2.3}
        r"\[[^\[\]]*\]",                                                            # [renpy.version_only]
    )

    # 用于 WOLF 和 RPGMaker 的规则
    RE_WOLF_RPGMAKER: tuple[str] = (
        r"en\(.{0,8}[vs]\[\d+\].{0,16}\)",                                          # en(!s[982]) en(v[982] >= 1)
        r"if\(.{0,8}[vs]\[\d+\].{0,16}\)",                                          # if(!s[982]) if(v[982] >= 1)
        r"[<【]{0,1}[/\\][a-z]{1,8}[<\[][a-z\d]{0,16}[>\]][>】]{0,1}",              # /c[xy12] \bc[xy12] <\bc[xy12]>【/c[xy12]】
        r"%\d+",                                                                    # %1 %2
        r"@\d+",                                                                    # WOLF - 角色 ID
        r"\\[cus]db\[.+?:.+?:.+?\]",                                                # WOLF - 数据库变量 \cdb[0:1:2] \udb[0:1:2] \sdb[\cself[90]:\cself[10]:17]
        r"\\fr",                                                                    # 重置文本的改变
        r"\\fb",                                                                    # 加粗
        r"\\fi",                                                                    # 倾斜
        r"\\\{",                                                                    # 放大字体 \{
        r"\\\}",                                                                    # 缩小字体 \}
        # r"\\g",                                                                   # 显示货币 \G
        r"\\\$",                                                                    # 打开金币框 \$
        r"\\\.",                                                                    # 等待0.25秒 \.
        r"\\\|",                                                                    # 等待1秒 \|
        r"\\!",                                                                     # 等待按钮按下 \!
        r"\\>",                                                                     # 在同一行显示文字 \>
        r"\\<",                                                                     # 取消显示所有文字 \<
        r"\\\^",                                                                    # 显示文本后不需要等待 \^
        r"[/\\][a-z]{1,8}(?=<.{0,16}>|\[.{0,16}\])",                                # /C<> \FS<> /C[] \FS[] 中 <> [] 前的部分
        r"\\[a-z](?=[^a-z<>\[\]])",                                                 # \n \e \I 等单字母转义符
    )

    # 占位符文本
    PLACEHOLDER: re.Pattern = "{PLACEHOLDER}"

    # 正则表达式
    RE_BLANK: re.Pattern = re.compile(r"[\s\u3000]+", re.IGNORECASE)

    RE_PREFIX_NONE: re.Pattern = re.compile(rf"^(?:{"|".join(RE_NONE)})+", re.IGNORECASE)
    RE_SUFFIX_NONE: re.Pattern = re.compile(rf"(?:{"|".join(RE_NONE)})+$", re.IGNORECASE)

    RE_BASE_RENPY: re.Pattern = re.compile(rf"{"|".join(RE_RENPY)}", re.IGNORECASE)
    RE_CHECK_RENPY: re.Pattern = re.compile(rf"(?:{"|".join(RE_NONE + RE_RENPY)})+", re.IGNORECASE)
    RE_PREFIX_RENPY: re.Pattern = re.compile(rf"^(?:{"|".join(RE_NONE + RE_RENPY)})+", re.IGNORECASE)
    RE_SUFFIX_RENPY: re.Pattern = re.compile(rf"(?:{"|".join(RE_NONE + RE_RENPY)})+$", re.IGNORECASE)

    RE_BASE_WOLF_RPGMAKER: re.Pattern = re.compile(rf"{"|".join(RE_WOLF_RPGMAKER)}", re.IGNORECASE)
    RE_CHECK_WOLF_RPGMAKER: re.Pattern = re.compile(rf"(?:{"|".join(RE_NONE + RE_WOLF_RPGMAKER)})+", re.IGNORECASE)
    RE_PREFIX_WOLF_RPGMAKER: re.Pattern = re.compile(rf"^(?:{"|".join(RE_NONE + RE_WOLF_RPGMAKER)})+", re.IGNORECASE)
    RE_SUFFIX_WOLF_RPGMAKER: re.Pattern = re.compile(rf"(?:{"|".join(RE_NONE + RE_WOLF_RPGMAKER)})+$", re.IGNORECASE)

    def __init__(self) -> None:
        super().__init__()

        # 初始化
        self.placeholders: set[str] = set()
        self.prefix_codes: dict[str, str] = {}
        self.suffix_codes: dict[str, str] = {}

    # 预处理
    def pre_process(self, src_dict: dict[str, str], item_dict: dict[str, CacheItem]) -> tuple[dict[str, str], list[str]]:
        # 通过字典保证去重且有序
        samples: list[str] = []
        for k, item in zip(src_dict.keys(), item_dict.values()):
            if item.get_text_type() == CacheItem.TextType.MD:
                samples_ex: list[str] = self.pre_process_none(k, src_dict)
                samples.extend(samples_ex)
                samples.append("Markdown代码")
            elif item.get_text_type() == CacheItem.TextType.RENPY:
                samples_ex: list[str] = self.pre_process_renpy(k, src_dict)
                samples.extend(samples_ex)
            elif item.get_text_type() in (CacheItem.TextType.WOLF, CacheItem.TextType.RPGMAKER):
                samples_ex: list[str] = self.pre_process_wolf_rpgmaker(k, src_dict)
                samples.extend(samples_ex)
            else:
                samples_ex: list[str] = self.pre_process_none(k, src_dict)
                samples.extend(samples_ex)

        return src_dict, list(set({v.strip() for v in samples if v.strip() != ""}))

    # 预处理 - None
    def pre_process_none(self, k: str, src_dict: dict[str, str]) -> list[str]:
        # 查找与替换前缀代码段
        self.prefix_codes[k] = CodeSaver.RE_PREFIX_NONE.findall(src_dict.get(k))
        src_dict[k] = CodeSaver.RE_PREFIX_NONE.sub("", src_dict.get(k))

        # 查找与替换后缀代码段
        self.suffix_codes[k] = CodeSaver.RE_SUFFIX_NONE.findall(src_dict.get(k))
        src_dict[k] = CodeSaver.RE_SUFFIX_NONE.sub("", src_dict.get(k))

        # 如果处理后的文本为空，则记录 ID，并将文本替换为占位符
        if src_dict[k] == "":
            src_dict[k] = CodeSaver.PLACEHOLDER
            self.placeholders.add(k)

        return CodeSaver.RE_PREFIX_NONE.findall(src_dict.get(k))

    # 预处理 - RenPy
    def pre_process_renpy(self, k: str, src_dict: dict[str, str]) -> list[str]:
        # 查找与替换前缀代码段
        self.prefix_codes[k] = CodeSaver.RE_PREFIX_RENPY.findall(src_dict.get(k))
        src_dict[k] = CodeSaver.RE_PREFIX_RENPY.sub("", src_dict.get(k))

        # 查找与替换后缀代码段
        self.suffix_codes[k] = CodeSaver.RE_SUFFIX_RENPY.findall(src_dict.get(k))
        src_dict[k] = CodeSaver.RE_SUFFIX_RENPY.sub("", src_dict.get(k))

        # 如果处理后的文本为空，则记录 ID，并将文本替换为占位符
        if src_dict[k] == "":
            src_dict[k] = CodeSaver.PLACEHOLDER
            self.placeholders.add(k)

        return CodeSaver.RE_BASE_RENPY.findall(src_dict.get(k))

    # 预处理 - RPGMaker
    def pre_process_wolf_rpgmaker(self, k: str, src_dict: dict[str, str]) -> list[str]:
        # 查找与替换前缀代码段
        self.prefix_codes[k] = CodeSaver.RE_PREFIX_WOLF_RPGMAKER.findall(src_dict.get(k))
        src_dict[k] = CodeSaver.RE_PREFIX_WOLF_RPGMAKER.sub("", src_dict.get(k))

        # 查找与替换后缀代码段
        self.suffix_codes[k] = CodeSaver.RE_SUFFIX_WOLF_RPGMAKER.findall(src_dict.get(k))
        src_dict[k] = CodeSaver.RE_SUFFIX_WOLF_RPGMAKER.sub("", src_dict.get(k))

        # 如果处理后的文本为空，则记录 ID，并将文本替换为占位符
        if src_dict[k] == "":
            src_dict[k] = CodeSaver.PLACEHOLDER
            self.placeholders.add(k)

        return CodeSaver.RE_BASE_WOLF_RPGMAKER.findall(src_dict.get(k))

    # 后处理
    def post_process(self, src_dict: dict[str, str], dst_dict: dict[str, str]) -> dict[str, str]:
        for k in dst_dict.keys():
            # 检查一下返回值的有效性
            if k not in src_dict:
                continue

            # 如果 ID 在占位符集合中，则将文本置为空
            if k in self.placeholders:
                dst_dict[k] = ""

            # 移除模型可能额外添加的头尾空白符
            dst_dict[k] = dst_dict.get(k).strip()

            # 还原前缀代码段
            dst_dict[k] = "".join(self.prefix_codes.get(k)) + dst_dict.get(k)

            # 还原后缀代码段
            dst_dict[k] = dst_dict.get(k) + "".join(self.suffix_codes.get(k))

        return dst_dict

    # 检查代码段
    def check(self, src: str, dst: str, text_type: str) -> bool:
        if text_type == CacheItem.TextType.RENPY:
            x: list[str] = CodeSaver.RE_CHECK_RENPY.findall(src)
            y: list[str] = CodeSaver.RE_CHECK_RENPY.findall(dst)
        elif text_type in (CacheItem.TextType.WOLF, CacheItem.TextType.RPGMAKER):
            x: list[str] = CodeSaver.RE_CHECK_WOLF_RPGMAKER.findall(src)
            y: list[str] = CodeSaver.RE_CHECK_WOLF_RPGMAKER.findall(dst)
        else:
            x: list[str] = []
            y: list[str] = []

        x = [CodeSaver.RE_BLANK.sub("", v) for v in x if CodeSaver.RE_BLANK.sub("", v) != ""]
        y = [CodeSaver.RE_BLANK.sub("", v) for v in y if CodeSaver.RE_BLANK.sub("", v) != ""]

        return x == y