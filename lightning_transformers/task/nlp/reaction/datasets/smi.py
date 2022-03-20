# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any

# from transformers import PreTrainedTokenizerBase
from transformers import BatchEncoding

from deepchem.feat.smiles_tokenizer import SmilesTokenizer

from lightning_transformers.task.nlp.translation.data import TranslationDataModule


class SMILESTranslationDataModule(TranslationDataModule):
    """返回字符串的特征
    
    """
    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: SmilesTokenizer,
        padding: str,
        max_source_length: int,
        max_target_length: int,
        src_text_column_name: str,
        tgt_text_column_name: str,
    ) -> BatchEncoding:
        translations = examples["translation"]  # Extract translations from dict
        # examples is a dict

        def extract_text(lang):
            """提取对应的列

            :param lang: 语言
            """
            return [text[lang] for text in translations]


        encoded_results = tokenizer.prepare_seq2seq_batch(
            src_texts=extract_text(src_text_column_name),
            tgt_texts=extract_text(tgt_text_column_name),
            max_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
        )
        return encoded_results
