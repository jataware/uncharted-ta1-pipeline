from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.convert_slow_tokenizer import (
    SLOW_TO_FAST_CONVERTERS,
    RobertaConverter,
)

from .configuration_layoutlmv3 import Layout_LMv3Config
from .modeling_layoutlmv3 import (
    Layout_LMv3ForTokenClassification,
    Layout_LMv3ForQuestionAnswering,
    Layout_LMv3ForSequenceClassification,
    Layout_LMv3Model,
)
from .tokenization_layoutlmv3 import Layout_LMv3Tokenizer
from .tokenization_layoutlmv3_fast import Layout_LMv3TokenizerFast


AutoConfig.register("layout_lmv3", Layout_LMv3Config)
AutoModel.register(Layout_LMv3Config, Layout_LMv3Model)
AutoModelForTokenClassification.register(
    Layout_LMv3Config, Layout_LMv3ForTokenClassification
)
AutoModelForQuestionAnswering.register(
    Layout_LMv3Config, Layout_LMv3ForQuestionAnswering
)
AutoModelForSequenceClassification.register(
    Layout_LMv3Config, Layout_LMv3ForSequenceClassification
)
AutoTokenizer.register(
    Layout_LMv3Config,
    slow_tokenizer_class=Layout_LMv3Tokenizer,
    fast_tokenizer_class=Layout_LMv3TokenizerFast,
)
SLOW_TO_FAST_CONVERTERS.update({"Layout_LMv3Tokenizer": RobertaConverter})
