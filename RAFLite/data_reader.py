import itertools
import sys
from typing import List
import json
import torch
from tqdm import tqdm
import operator
import os

import logging
import pickle
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
import argparse
from pathlib import *
from string import punctuation
import numpy.ma as ma
import numpy as np

class dataReader:
    def __init__(self,max_length=198,tokenizer=None, is_debug=False):
        self.tokenized_map = {}
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.bad_directions = 0
        self.truncated = 0
        self.is_debug = is_debug
        self.actual_max = 0
    def load_cache(self, path_to_docmap):
        pickle_in = open(path_to_docmap, "rb")
        cached_obj = pickle.load(pickle_in)
        pickle_in.close()
        return cached_obj
    def save_cache(self, obj, fname):
        with open(fname, 'wb+') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    """
    Takes as input list of knowledge sentences, a qustion a answer choice 
    and returns 1) tokens 2) segment ids 3) fact masks 
    """
    def bert_features_from_qa(self, tokenizer, max_pieces: int, question: str, answer: str, context: str):
        cls_token = "[CLS]"
        sep_token = "[SEP]"

        question_tokens = self.tokenized_map.get(question, tokenizer.tokenize(question))
        self.tokenized_map[question] = question_tokens
        context_tokens=[cls_token]
        start_position = 1
        fact_mask_list=[]
        end_position = 0
        for sentence in context:
            sentence_tokens = self.tokenized_map.get(sentence, tokenizer.tokenize(sentence))
            self.tokenized_map[sentence] = sentence_tokens
            context_tokens+=sentence_tokens + [sep_token]
            end_position = len(context_tokens)
            fact_mask = [0]*max_pieces
            fact_mask[start_position:(end_position-1)] = [1] * len(sentence_tokens)
            assert sentence_tokens == context_tokens[start_position:(end_position - 1)]
            start_position = end_position
            fact_mask_list.append(list(fact_mask))

            
        start_position = len(context_tokens)
        len_segment_1 = len(context_tokens)
        cquestion_tokens = context_tokens+ question_tokens + [sep_token]
        end_position = len(cquestion_tokens) - 1
        question_mask = [0]*max_pieces
        question_mask[start_position:end_position] = [1] * len(question_tokens)
        
        choice_tokens = self.tokenized_map.get(answer, tokenizer.tokenize(answer))
        self.tokenized_map[answer] = choice_tokens
        
        #cquestion_tokens, choice_tokens = self._truncate_tokens(cquestion_tokens, choice_tokens, max_pieces - 3)

        start_position = len(cquestion_tokens)
        tokens = cquestion_tokens + choice_tokens + [sep_token]
        end_position = len(tokens) - 1
        choice_mask = [0]*max_pieces
        choice_mask[start_position:end_position] = [1] * len(choice_tokens)
        
        segment_ids = [0]*len_segment_1 + [1]*(len(tokens)-len_segment_1 + 1)
        self.actual_max = max(self.actual_max, len(tokens))
        return tokens, segment_ids, fact_mask_list, question_mask, choice_mask

    """
    takes a mcq problem and return 
    1) one token id seq per choice
    2) one segment id per choice
    3) fact masks per choice
    4) optional fact1,fact2 mask per choice
    """
    def text_to_instance(self,  # type: ignore
                         tokenizer,  # type: ignore
                         max_seq_length: int,
                         premise_lists: List[List[str]],
                         choices: List[str],
                         question: str,
                         max_number_premises: int = None):

        finput_ids = []
        fsegment_ids = []
        fact_masks = []
        finput_masks=[]
        fquestion_masks=[]
        fchoice_masks=[]
        for premise_list, hypothesis in zip(premise_lists, choices):
            ph_tokens, segment_ids, pc_fact_masks, pc_question_masks, choice_mask \
                = self.bert_features_from_qa(tokenizer, max_seq_length,
                                             question=question, context=premise_list,
                                             answer=hypothesis)
            # tokenize
            input_ids = tokenizer.convert_tokens_to_ids(ph_tokens)
            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            segment_ids += padding
            input_mask = [1] * len(input_ids) + padding

            
            finput_ids.append(input_ids)
            fsegment_ids.append(segment_ids)
            finput_masks.append(input_mask)
            fact_masks.append(pc_fact_masks)
            fquestion_masks.append(pc_question_masks)
            fchoice_masks.append(choice_mask)

        return finput_ids, fsegment_ids, finput_masks ,fact_masks, fquestion_masks, fchoice_masks

    """
    read each json object and converts it in appropriate format
    """
    def read(self, file_path: str, tokenizer, max_seq_len: int, max_number_premises: int = None):
        file_name = file_path.split("/")[-1]
        dir_name = os.path.dirname(file_path)
        cache_file_path = os.path.join(dir_name,
                                       'cached_bert_{}_{}_{}'.format(file_name, max_seq_len, max_number_premises))
        if os.path.exists(cache_file_path):
            logger.info("Loading features from cached file %s", cache_file_path)
            features = self.load_cache(cache_file_path)
            
            all_tokens = features['tokens']
            all_masks = features['masks']
            all_segment_ids = features['segmentids']
            all_fact_masks = features['all_fact_masks']
            all_question_masks=features['question_masks']
            all_choice_masks=features['choice_masks']
            all_labels = features.get('all_labels', None)
            all_f1_labels = features.get('all_f1_labels', None)
            use_f1_labels = features.get('use_f1_labels', None)
            all_f2_labels = features.get('all_f2_labels', None)
            use_f2_labels = features.get('use_f2_labels', None)
            if all_labels is None:
                return TensorDataset(all_tokens, all_masks, all_segment_ids, all_fact_masks,
                                     all_question_masks, all_choice_masks)

            if all_f1_labels is not None:
                return TensorDataset(all_tokens, all_masks, all_segment_ids, all_fact_masks,
                                     all_question_masks, all_choice_masks,
                                     all_labels,
                                     all_f1_labels, use_f1_labels, all_f2_labels, use_f2_labels)

            else:
                return TensorDataset(all_tokens, all_masks, all_segment_ids, all_fact_masks,
                                     all_question_masks, all_choice_masks, all_labels)


        all_tokens = []
        all_segment_ids = []
        all_f1_labels = []
        all_f2_labels = []
        use_f1_labels = []
        use_f2_labels = []
        all_labels = []
        all_fact_masks = []
        all_question_masks = []
        all_choice_masks = []
        flag_anno = True
        with open(file_path, 'r') as te_file:
            logger.info("Reading MCQ instances of the dataset at: %s", file_path)
            for line in tqdm(te_file, desc="preparing dataset:"):
                if line.strip() == '':
                    continue
                example = json.loads(line)
                label = None
                if "answerKey" in example:
                    label = ord(example["answerKey"])-65
                    assert 0<=label and label<=7, label

                premises = [choice["para"] for choice in example["question"]["choices"]]
                choices  = [choice["text"] for choice in example["question"]["choices"]]
                question = example["question"]["stem"]
                call_f1_labels = []
                call_f2_labels = []
                cuse_f1_labels = []
                cuse_f2_labels = []
                for choice in example["question"]["choices"]:
                    if "f1_idx" not in choice:
                        flag_anno = False
                        break
                    call_f1_labels.append(choice["f1_idx"])
                    call_f2_labels.append(choice["f2_idx"])
                    if choice["f1_idx"]==-1:
                        cuse_f1_labels.append(0)
                    else:
                        cuse_f1_labels.append(1)

                    if choice["f2_idx"]==-1:
                        cuse_f2_labels.append(0)
                    else:
                        cuse_f2_labels.append(1)

                all_f1_labels.append(call_f1_labels)
                use_f1_labels.append(cuse_f1_labels)
                all_f2_labels.append(call_f2_labels)
                use_f2_labels.append(cuse_f2_labels)
                
                pp_tokens, pp_segment_ids, pp_input_masks, pp_fact_masks, pp_question_masks, \
                pp_choice_masks = self.text_to_instance(tokenizer, max_seq_len, premises, choices,
                                                                  question, max_number_premises)
#                 print(pp_fact_masks)
                assert len(pp_tokens) == len(pp_segment_ids)
                all_tokens.append(pp_tokens)
                all_segment_ids.append(pp_segment_ids)
                all_labels.append(label)
                all_fact_masks.append(pp_fact_masks)
                all_question_masks.append(pp_question_masks)
                all_choice_masks.append(pp_choice_masks)

        all_tokens = torch.tensor(all_tokens, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        if None in all_labels:
            print("found none ")
            all_labels = None
        else:
            all_labels = torch.tensor(all_labels, dtype=torch.long)

        all_masks = (all_tokens != 0).long().clone().detach()
        all_fact_masks = torch.tensor(all_fact_masks, dtype=torch.long)
        all_question_masks = torch.tensor(all_question_masks, dtype=torch.long)
        all_choice_masks = torch.tensor(all_choice_masks, dtype=torch.long)
        if flag_anno:
            all_f1_labels=torch.tensor(all_f1_labels, dtype=torch.long)
            use_f1_labels=torch.tensor(use_f1_labels, dtype=torch.long)
            all_f2_labels=torch.tensor(all_f2_labels, dtype=torch.long)
            use_f2_labels=torch.tensor(use_f2_labels, dtype=torch.long)

        features = {}
        features['tokens'] = all_tokens
        features['masks'] = all_masks
        features['segmentids'] = all_segment_ids
        features['all_fact_masks']=all_fact_masks
        features['question_masks']=all_question_masks
        features['choice_masks']=all_choice_masks
        if all_labels is not None:
            features['all_labels'] = all_labels
        if flag_anno:
            features['all_f1_labels'] = all_f1_labels
            features['use_f1_labels'] = use_f1_labels
            features['all_f2_labels'] = all_f2_labels
            features['use_f2_labels'] = use_f2_labels
        self.save_cache(features, cache_file_path)
        if all_labels is None:
            return TensorDataset(all_tokens, all_masks, all_segment_ids, all_fact_masks,
                                 all_question_masks, all_choice_masks)
        if flag_anno:
            return TensorDataset(all_tokens, all_masks, all_segment_ids, all_fact_masks,
                                all_question_masks, all_choice_masks,
                                all_labels,
                                 all_f1_labels, use_f1_labels, all_f2_labels, use_f2_labels)
        else:
            return TensorDataset(all_tokens, all_masks, all_segment_ids, all_fact_masks,
                                 all_question_masks, all_choice_masks,
                                 all_labels)
def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    reader = dataReader()

    out = reader.read("dev.jsonl", tokenizer, 156, None)
    print(reader.actual_max)
#     out = reader.read("p_dev.jsonl", tokenizer, 70, None)
#     out = reader.read("p_test.jsonl", tokenizer, 70, None)
#     print(out[0])
#     tokens, segs, masks, labels = out[0]



if __name__ == "__main__":
    main()
