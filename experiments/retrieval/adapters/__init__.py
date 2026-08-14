"""公开三种数据集适配器。"""

from .common import AdapterError
from .hotpotqa import adapt_hotpotqa
from .triviaqa import adapt_triviaqa
from .two_wiki import adapt_two_wiki

__all__ = ["AdapterError", "adapt_hotpotqa", "adapt_triviaqa", "adapt_two_wiki"]
