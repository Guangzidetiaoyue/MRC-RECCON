from __future__ import absolute_import, division, print_function

import json
import logging
import math
import os
import random
import warnings
from dataclasses import asdict
from multiprocessing import cpu_count
import scipy.sparse as sp
import numpy as np
import pandas as pd
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from scipy.stats import pearsonr
from sklearn.metrics import (
    confusion_matrix,
    label_ranking_average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
)
import gc
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    BartConfig,
    BartForQuestionAnswering,
    BartTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraTokenizer,
    LongformerConfig,
    LongformerForQuestionAnswering,
    LongformerTokenizer,
    MobileBertConfig,
    MobileBertForQuestionAnswering,
    MobileBertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import QuestionAnsweringArgs
from simpletransformers.config.utils import sweep_config_to_sweep_values
from simpletransformers.custom_models.models import ElectraForQuestionAnswering, XLMRobertaForQuestionAnswering
from simpletransformers.question_answering.question_answering_utils import (
    LazyQuestionAnsweringDataset,
    RawResult,
    RawResultExtended,
    build_examples,
    convert_examples_to_features,
    get_best_predictions,
    get_best_predictions_extended,
    get_examples,
    squad_convert_examples_to_features,
    to_list,
    write_predictions,
    write_predictions_extended,
)
from .my_model import RobertaForQuestionAnswering,BertForQuestionAnswering



try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False


logger = logging.getLogger(__name__)


class QuestionAnsweringModel:
    def __init__(self, model_type, model_name, args=None, use_cuda=True, cuda_device=-1, **kwargs):

        """
        Initializes a QuestionAnsweringModel model.

        Args:
            model_type: The type of model (bert, xlnet, xlm, distilbert)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            args (optional): Default args will be used if this parameter is not provided. If provided,
                it should be a dict containing the args that should be changed in the default args'
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
            "auto": (AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering),
            "bart": (BartConfig, BartForQuestionAnswering, BartTokenizer),
            "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
            "electra": (ElectraConfig, ElectraForQuestionAnswering, ElectraTokenizer),
            "longformer": (LongformerConfig, LongformerForQuestionAnswering, LongformerTokenizer),
            "mobilebert": (MobileBertConfig, MobileBertForQuestionAnswering, MobileBertTokenizer),
            "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
            "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
            "xlmroberta": (XLMRobertaConfig, XLMRobertaForQuestionAnswering, XLMRobertaTokenizer),
            "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
        }
        # MODEL_CLASSES = {
        #     "roberta":(RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer)
        # }

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, QuestionAnsweringArgs):
            self.args = args

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if not use_cuda:
            self.args.fp16 = False

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.config = config_class.from_pretrained(model_name, **self.args.config)

        if not self.args.quantized_model:
            self.model = model_class.from_pretrained(model_name, config=self.config, **kwargs)
        else:
            quantized_weights = torch.load(os.path.join(model_name, "pytorch_model.bin"))
            self.model = model_class.from_pretrained(None, config=self.config, state_dict=quantized_weights)

        if self.args.dynamic_quantize:
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        if self.args.quantized_model:
            self.model.load_state_dict(quantized_weights)
        if self.args.dynamic_quantize:
            self.args.quantized_model = True

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        self.results_0 = {}
        self.results_1 = {}

        if self.args.fp16:
            try:
                from torch.cuda import amp
            except AttributeError:
                raise AttributeError("fp16 requires Pytorch >= 1.6. Please update Pytorch or turn off fp16.")

        self.tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=self.args.do_lower_case, **kwargs)

        self.args.model_name = model_name
        self.args.model_type = model_type

        if self.args.wandb_project and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args.wandb_project = None

    def convert_all_data_to_dataset(self, data_all):
        all_input_ids = torch.tensor(data_all[0], dtype=torch.long)
        all_attention_masks = torch.tensor(data_all[1], dtype=torch.long)
        all_token_type_ids = torch.tensor(data_all[2], dtype=torch.long)
        all_start_positions = torch.tensor(data_all[3], dtype=torch.long)
        all_end_positions = torch.tensor(data_all[4], dtype=torch.long)
        all_cls_index = torch.tensor(data_all[5], dtype=torch.long)
        all_p_mask = torch.tensor(data_all[6], dtype=torch.float)
        all_is_impossible = torch.tensor(data_all[7], dtype=torch.float)
        all_token_idx = torch.tensor(data_all[8], dtype=torch.long)
        all_segment_idx = torch.tensor(data_all[9], dtype=torch.long)
        all_clause_idx = torch.tensor(data_all[10], dtype=torch.long)
        all_conv_len = torch.tensor(data_all[11], dtype=torch.long)
        all_utterance_input =torch.tensor(data_all[12], dtype=torch.long)
        all_q_query_input = torch.tensor(data_all[13], dtype=torch.long)
        all_q_query_mask = torch.tensor(data_all[14], dtype=torch.long)
        all_q_query_token_type = torch.tensor(data_all[15], dtype=torch.long)
        all_answer_cls = torch.tensor(data_all[16], dtype=torch.long)
        
        all_input_ids_1 = torch.tensor(data_all[17], dtype=torch.long)
        all_attention_masks_1 = torch.tensor(data_all[18], dtype=torch.long)
        all_token_type_ids_1 = torch.tensor(data_all[19], dtype=torch.long)
        all_start_positions_1 = torch.tensor(data_all[20], dtype=torch.long)
        all_end_positions_1 = torch.tensor(data_all[21], dtype=torch.long)
        all_cls_index_1 = torch.tensor(data_all[22], dtype=torch.long)
        all_p_mask_1 = torch.tensor(data_all[23], dtype=torch.float)
        all_is_impossible_1 = torch.tensor(data_all[24], dtype=torch.float)
        all_token_idx_1 = torch.tensor(data_all[25], dtype=torch.long)
        all_segment_idx_1 = torch.tensor(data_all[26], dtype=torch.long)
        all_clause_idx_1 = torch.tensor(data_all[27], dtype=torch.long)
        all_conv_len_1 = torch.tensor(data_all[28], dtype=torch.long)
        all_utterance_input_1 =torch.tensor(data_all[29], dtype=torch.long)
        all_q_query_input_1 = torch.tensor(data_all[30], dtype=torch.long)
        all_q_query_mask_1 = torch.tensor(data_all[31], dtype=torch.long)
        all_q_query_token_type_1 = torch.tensor(data_all[32], dtype=torch.long)
        all_answer_cls_1 = torch.tensor(data_all[33], dtype=torch.long)
        
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_start_positions,
            all_end_positions,
            all_cls_index,
            all_p_mask,
            all_is_impossible,
            all_token_idx,
            all_segment_idx,
            all_clause_idx,
            all_conv_len,
            all_utterance_input,
            all_q_query_input,
            all_q_query_mask,
            all_q_query_token_type,
            all_answer_cls,
            all_input_ids_1,
            all_attention_masks_1,
            all_token_type_ids_1,
            all_start_positions_1,
            all_end_positions_1,
            all_cls_index_1,
            all_p_mask_1,
            all_is_impossible_1,
            all_token_idx_1,
            all_segment_idx_1,
            all_clause_idx_1,
            all_conv_len_1,
            all_utterance_input_1,
            all_q_query_input_1,
            all_q_query_mask_1,
            all_q_query_token_type_1,
            all_answer_cls_1
        )
        
        return dataset
    
    def convert_valid_data_to_dataset(self, data_all):
        all_input_ids = torch.tensor(data_all[0], dtype=torch.long)
        all_attention_masks = torch.tensor(data_all[1], dtype=torch.long)
        all_token_type_ids = torch.tensor(data_all[2], dtype=torch.long)
        all_feature_index = torch.tensor(data_all[3], dtype=torch.long)
        all_cls_index = torch.tensor(data_all[4], dtype=torch.long)
        all_p_mask = torch.tensor(data_all[5], dtype=torch.float)
        all_is_impossible = torch.tensor(data_all[6], dtype=torch.float)
        all_token_idx = torch.tensor(data_all[7], dtype=torch.long)
        all_segment_idx = torch.tensor(data_all[8], dtype=torch.long)
        all_clause_idx = torch.tensor(data_all[9], dtype=torch.long)
        all_conv_len = torch.tensor(data_all[10], dtype=torch.long)
        all_utterance_input =torch.tensor(data_all[11], dtype=torch.long)
        all_q_query_input = torch.tensor(data_all[12], dtype=torch.long)
        all_q_query_mask = torch.tensor(data_all[13], dtype=torch.long)
        all_q_query_token_type = torch.tensor(data_all[14], dtype=torch.long)
        all_answer_cls = torch.tensor(data_all[15], dtype=torch.long)
        
        all_input_ids_1 = torch.tensor(data_all[16], dtype=torch.long)
        all_attention_masks_1 = torch.tensor(data_all[17], dtype=torch.long)
        all_token_type_ids_1 = torch.tensor(data_all[18], dtype=torch.long)
        all_feature_index_1 = torch.tensor(data_all[19], dtype=torch.long)
        all_cls_index_1 = torch.tensor(data_all[20], dtype=torch.long)
        all_p_mask_1 = torch.tensor(data_all[21], dtype=torch.float)
        all_is_impossible_1 = torch.tensor(data_all[22], dtype=torch.float)
        all_token_idx_1 = torch.tensor(data_all[23], dtype=torch.long)
        all_segment_idx_1 = torch.tensor(data_all[24], dtype=torch.long)
        all_clause_idx_1 = torch.tensor(data_all[25], dtype=torch.long)
        all_conv_len_1 = torch.tensor(data_all[26], dtype=torch.long)
        all_utterance_input_1 =torch.tensor(data_all[27], dtype=torch.long)
        all_q_query_input_1 = torch.tensor(data_all[28], dtype=torch.long)
        all_q_query_mask_1 = torch.tensor(data_all[29], dtype=torch.long)
        all_q_query_token_type_1 = torch.tensor(data_all[30], dtype=torch.long)
        all_answer_cls_1 = torch.tensor(data_all[31], dtype=torch.long)
        
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_feature_index,
            all_cls_index,
            all_p_mask,
            all_is_impossible,
            all_token_idx,
            all_segment_idx,
            all_clause_idx,
            all_conv_len,
            all_utterance_input,
            all_q_query_input,
            all_q_query_mask,
            all_q_query_token_type,
            all_answer_cls,
            
            all_input_ids_1,
            all_attention_masks_1,
            all_token_type_ids_1,
            all_feature_index_1,
            all_cls_index_1,
            all_p_mask_1,
            all_is_impossible_1,
            all_token_idx_1,
            all_segment_idx_1,
            all_clause_idx_1,
            all_conv_len_1,
            all_utterance_input_1,
            all_q_query_input_1,
            all_q_query_mask_1,
            all_q_query_token_type_1,
            all_answer_cls_1
        )
        
        return dataset
    def load_and_cache_examples(self, examples, m,evaluate=False, no_cache=False, output_examples=False):
        """
        Converts a list of examples to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir+str(m), exist_ok=True)

        examples = get_examples(examples, is_training=not evaluate)

        mode = "dev" if evaluate else "train"
        cached_features_file = os.path.join(
            args.cache_dir, "cached_{}_{}_{}_{}_{}".format(mode, args.model_type, args.max_seq_length, len(examples),m),
        )

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not no_cache) or (mode == "dev" and args.use_cached_eval_features)
        ):
            features = torch.load(cached_features_file)
            logger.info(f" Features loaded from cache at {cached_features_file}")
        else:
            logger.info(" Converting to features started.")

            features, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=not evaluate,
                tqdm_enabled=not args.silent,
                threads=args.process_count,
                args=args,
            )


        if output_examples:
            return dataset, examples, features
        return dataset

    def train_model(
        self, train_data_0,train_data_1, output_dir=False, show_running_loss=True, args=None, eval_data_0=None, eval_data_1=None , verbose=True, **kwargs
    ):
        """
        Trains the model using 'train_data'

        Args:
            train_data: Path to JSON file containing training data OR list of Python dicts in the correct format. The model will be trained on this data.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_data (optional): Path to JSON file containing evaluation data against which evaluation will be performed when evaluate_during_training is enabled.
                Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)

        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and eval_data_0 is None and eval_data_1 is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_data is not specified."
                " Pass eval_data to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome.".format(output_dir)
            )

        self._move_model_to_device()

        if self.args.lazy_loading:
            if isinstance(train_data_0, str):
                train_dataset_0 = LazyQuestionAnsweringDataset(train_data_0, self.tokenizer, self.args)
            else:
                raise ValueError("Input must be given as a path to a file when using lazy loading")
        else:
            if isinstance(train_data_0, str):
                with open(train_data_0, "r", encoding=self.args.encoding) as f:
                    train_examples_0 = json.load(f)
            else:
                train_examples_0 = train_data_0
            train_dataset_0 = self.load_and_cache_examples(train_examples_0,0)
        # train_dataset_0_list = list(train_dataset_0)

        if self.args.lazy_loading:
            if isinstance(train_data_1, str):
                train_dataset_1 = LazyQuestionAnsweringDataset(train_data_1, self.tokenizer, self.args)
            else:
                raise ValueError("Input must be given as a path to a file when using lazy loading")
        else:
            if isinstance(train_data_1, str):
                with open(train_data_1, "r", encoding=self.args.encoding) as f:
                    train_examples_1 = json.load(f)
            else:
                train_examples_1 = train_data_1
            train_dataset_1 = self.load_and_cache_examples(train_examples_1,1)
        # train_dataset_1_list = list(train_dataset_1)

        train_dataset_all_t = train_dataset_0 + train_dataset_1
        # train_dataset_all_t = []
        # for i_i,item in enumerate(train_dataset_0):
        #     train_data_0_temp = list(train_dataset_0[i_i])
        #     train_data_1_temp = list(train_dataset_1[i_i])
        #     train_data_temp = train_data_0_temp+train_data_1_temp
        #     train_dataset_all_t.append(train_data_temp)
        train_dataset_all = self.convert_all_data_to_dataset(train_dataset_all_t)
        del train_dataset_all_t,train_data_0,train_data_1,train_examples_0,train_examples_1
        gc.collect()

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_dataset_all, output_dir, show_running_loss=show_running_loss, eval_data_0=eval_data_0, eval_data_1=eval_data_1,**kwargs
        )

        self.save_model(model=self.model)

        logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_type, output_dir))

        return global_step, training_details

    def bert_batch_preprocessing(self,batch):
        if len(batch[0])==34:
            all_input_ids,all_attention_masks,all_token_type_ids,all_start_positions,\
                all_end_positions,all_cls_index,all_p_mask,all_is_impossible,\
                all_token_idx,all_segment_idx,all_clause_idx,all_conv_len,all_utterance_input,all_q_query_input,all_q_query_mask,all_q_query_token_type,all_answer_cls,\
                all_input_ids_1,all_attention_masks_1,all_token_type_ids_1,all_start_positions_1,\
                all_end_positions_1,all_cls_index_1,all_p_mask_1,all_is_impossible_1,\
                all_token_idx_1,all_segment_idx_1,all_clause_idx_1,all_conv_len_1,all_utterance_input_1,all_q_query_input_1,all_q_query_mask_1,all_q_query_token_type_1,all_answer_cls_1= zip(*batch)
            N = max(all_conv_len).cpu().numpy().tolist()
            adj_b = []
            for doc_len in all_conv_len:
                doc_len = doc_len.cpu().numpy().tolist()
                adj = np.zeros((doc_len, doc_len))
                for i in range(doc_len):
                    for j in range(i+1):
                        adj[i][j] = 1
                adj = sp.coo_matrix(adj)
                adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                                    shape=(N, N), dtype=np.float32)
                adj_b.append(adj.toarray())
            adj_b = torch.tensor(np.array(adj_b), dtype=torch.long)

            N_1 = max(all_conv_len_1).cpu().numpy().tolist()
            adj_b_1 = []
            for doc_len in all_conv_len_1:
                doc_len = doc_len.cpu().numpy().tolist()
                adj = np.zeros((doc_len, doc_len))
                for i in range(doc_len):
                    for j in range(i+1):
                        adj[i][j] = 1
                adj = sp.coo_matrix(adj)
                adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                                    shape=(N_1, N_1), dtype=np.float32)
                adj_b_1.append(adj.toarray())
            adj_b_1 = torch.tensor(np.array(adj_b_1), dtype=torch.long)

            all_input_ids = torch.tensor(np.array([np.array(temp) for temp in all_input_ids]))
            all_attention_masks = torch.tensor(np.array([np.array(temp) for temp in all_attention_masks]))
            all_token_type_ids = torch.tensor(np.array([np.array(temp) for temp in all_token_type_ids]))
            all_start_positions = torch.tensor(all_start_positions)#.unsqueeze(1)
            all_end_positions = torch.tensor(all_end_positions)#.unsqueeze(1)
            all_cls_index = torch.tensor(all_cls_index)#.unsqueeze(1)
            all_p_mask = torch.tensor(np.array([np.array(temp) for temp in all_p_mask]))
            all_is_impossible = torch.tensor(all_is_impossible)#.unsqueeze(1)
            all_token_idx = torch.tensor(np.array([np.array(temp) for temp in all_token_idx]))
            all_segment_idx = torch.tensor(np.array([np.array(temp) for temp in all_segment_idx]))
            all_clause_idx= torch.tensor(np.array([np.array(temp) for temp in all_clause_idx]))
            all_conv_len = torch.tensor(all_conv_len)#.unsqueeze(1)
            all_utterance_input = torch.tensor(np.array([np.array(temp) for temp in all_utterance_input]))
            all_q_query_input = torch.tensor(np.array([np.array(temp) for temp in all_q_query_input]))
            all_q_query_mask = torch.tensor(np.array([np.array(temp) for temp in all_q_query_mask]))
            all_q_query_token_type = torch.tensor(np.array([np.array(temp) for temp in all_q_query_token_type]))
            all_answer_cls = torch.tensor(all_answer_cls)
            
            all_input_ids_1 = torch.tensor(np.array([np.array(temp) for temp in all_input_ids_1]))
            all_attention_masks_1 = torch.tensor(np.array([np.array(temp) for temp in all_attention_masks_1]))
            all_token_type_ids_1 = torch.tensor(np.array([np.array(temp) for temp in all_token_type_ids_1]))
            all_start_positions_1 = torch.tensor(all_start_positions_1)#.unsqueeze(1)
            all_end_positions_1 = torch.tensor(all_end_positions_1)#.unsqueeze(1)
            all_cls_index_1 = torch.tensor(all_cls_index_1)#.unsqueeze(1)
            all_p_mask_1 = torch.tensor(np.array([np.array(temp) for temp in all_p_mask_1]))
            all_is_impossible_1 = torch.tensor(all_is_impossible_1)#.unsqueeze(1)
            all_token_idx_1 = torch.tensor(np.array([np.array(temp) for temp in all_token_idx_1]))
            all_segment_idx_1 = torch.tensor(np.array([np.array(temp) for temp in all_segment_idx_1]))
            all_clause_idx_1 = torch.tensor(np.array([np.array(temp) for temp in all_clause_idx_1]))
            all_conv_len_1 = torch.tensor(all_conv_len_1)#.unsqueeze(1)
            all_utterance_input_1 = torch.tensor(np.array([np.array(temp) for temp in all_utterance_input_1]))
            all_q_query_input_1 = torch.tensor(np.array([np.array(temp) for temp in all_q_query_input_1]))
            all_q_query_mask_1 = torch.tensor(np.array([np.array(temp) for temp in all_q_query_mask_1]))
            all_q_query_token_type_1 = torch.tensor(np.array([np.array(temp) for temp in all_q_query_token_type_1]))
            all_answer_cls_1 = torch.tensor(all_answer_cls_1)

            return all_input_ids,all_attention_masks,all_token_type_ids,all_start_positions,\
                all_end_positions,all_cls_index,all_p_mask,all_is_impossible,\
                all_token_idx,all_segment_idx,all_clause_idx,all_conv_len,adj_b,all_utterance_input,all_q_query_input,all_q_query_mask,all_q_query_token_type,all_answer_cls,\
                all_input_ids_1,all_attention_masks_1,all_token_type_ids_1,all_start_positions_1,all_end_positions_1,all_cls_index_1,all_p_mask_1,all_is_impossible_1,\
                all_token_idx_1,all_segment_idx_1,all_clause_idx_1,all_conv_len_1,adj_b_1,all_utterance_input_1,all_q_query_input_1,all_q_query_mask_1,all_q_query_token_type_1,all_answer_cls_1
        else:
            all_input_ids,all_attention_masks,all_token_type_ids,all_feature_index,all_cls_index,all_p_mask,all_is_impossible,all_token_idx,all_segment_idx,all_clause_idx,all_conv_len,all_utterance_input,all_q_query_input,all_q_query_mask,all_q_query_token_type,all_answer_cls,\
            all_input_ids_1,all_attention_masks_1,all_token_type_ids_1,all_feature_index_1,all_cls_index_1,all_p_mask_1,all_is_impossible_1,all_token_idx_1,all_segment_idx_1,all_clause_idx_1,all_conv_len_1,all_utterance_input_1,all_q_query_input_1,all_q_query_mask_1,all_q_query_token_type_1,all_answer_cls_1 = zip(*batch) #all_start_positions,\all_end_positions,
            N = max(all_conv_len).cpu().numpy().tolist()
            adj_b = []
            for doc_len in all_conv_len:
                doc_len = doc_len.cpu().numpy().tolist()
                adj = np.zeros((doc_len, doc_len))
                for i in range(doc_len):
                    for j in range(i+1):
                        adj[i][j] = 1
                adj = sp.coo_matrix(adj)
                adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                                    shape=(N, N), dtype=np.float32)
                adj_b.append(adj.toarray())
            adj_b = torch.tensor(np.array(adj_b), dtype=torch.long)

            N_1 = max(all_conv_len_1).cpu().numpy().tolist()
            adj_b_1 = []
            for doc_len in all_conv_len_1:
                doc_len = doc_len.cpu().numpy().tolist()
                adj = np.zeros((doc_len, doc_len))
                for i in range(doc_len):
                    for j in range(i+1):
                        adj[i][j] = 1
                adj = sp.coo_matrix(adj)
                adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                                    shape=(N_1, N_1), dtype=np.float32)
                adj_b_1.append(adj.toarray())
            adj_b_1 = torch.tensor(np.array(adj_b_1), dtype=torch.long)

            all_input_ids = torch.tensor(np.array([np.array(temp) for temp in all_input_ids]))
            all_attention_masks = torch.tensor(np.array([np.array(temp) for temp in all_attention_masks]))
            all_token_type_ids = torch.tensor(np.array([np.array(temp) for temp in all_token_type_ids]))
            all_feature_index = torch.tensor(all_feature_index)#.unsqueeze(1)
            all_cls_index = torch.tensor(all_cls_index)#.unsqueeze(1)
            all_p_mask = torch.tensor(np.array([np.array(temp) for temp in all_p_mask]))
            all_is_impossible = torch.tensor(all_is_impossible)#.unsqueeze(1)
            all_token_idx = torch.tensor(np.array([np.array(temp) for temp in all_token_idx]))
            all_segment_idx = torch.tensor(np.array([np.array(temp) for temp in all_segment_idx]))
            all_clause_idx= torch.tensor(np.array([np.array(temp) for temp in all_clause_idx]))
            all_conv_len = torch.tensor(all_conv_len)#.unsqueeze(1)
            all_utterance_input = torch.tensor(np.array([np.array(temp) for temp in all_utterance_input]))
            all_q_query_input = torch.tensor(np.array([np.array(temp) for temp in all_q_query_input]))
            all_q_query_mask = torch.tensor(np.array([np.array(temp) for temp in all_q_query_mask]))
            all_q_query_token_type = torch.tensor(np.array([np.array(temp) for temp in all_q_query_token_type]))
            all_answer_cls = torch.tensor(all_answer_cls)
            
            all_input_ids_1 = torch.tensor(np.array([np.array(temp) for temp in all_input_ids_1]))
            all_attention_masks_1 = torch.tensor(np.array([np.array(temp) for temp in all_attention_masks_1]))
            all_token_type_ids_1 = torch.tensor(np.array([np.array(temp) for temp in all_token_type_ids_1]))
            all_feature_index_1 = torch.tensor(all_feature_index_1)#.unsqueeze(1)
            all_cls_index_1 = torch.tensor(all_cls_index_1)#.unsqueeze(1)
            all_p_mask_1 = torch.tensor(np.array([np.array(temp) for temp in all_p_mask_1]))
            all_is_impossible_1 = torch.tensor(all_is_impossible_1)#.unsqueeze(1)
            all_token_idx_1 = torch.tensor(np.array([np.array(temp) for temp in all_token_idx_1]))
            all_segment_idx_1 = torch.tensor(np.array([np.array(temp) for temp in all_segment_idx_1]))
            all_clause_idx_1 = torch.tensor(np.array([np.array(temp) for temp in all_clause_idx_1]))
            all_conv_len_1 = torch.tensor(all_conv_len_1)#.unsqueeze(1)
            all_utterance_input_1 = torch.tensor(np.array([np.array(temp) for temp in all_utterance_input_1]))
            all_q_query_input_1 = torch.tensor(np.array([np.array(temp) for temp in all_q_query_input_1]))
            all_q_query_mask_1 = torch.tensor(np.array([np.array(temp) for temp in all_q_query_mask_1]))
            all_q_query_token_type_1 = torch.tensor(np.array([np.array(temp) for temp in all_q_query_token_type_1]))
            all_answer_cls_1 = torch.tensor(all_answer_cls_1)
    
            return all_input_ids,all_attention_masks,all_token_type_ids,all_feature_index,all_cls_index,all_p_mask,all_is_impossible,all_token_idx,all_segment_idx,all_clause_idx,all_conv_len,adj_b,all_utterance_input,all_q_query_input,all_q_query_mask,all_q_query_token_type,all_answer_cls, \
            all_input_ids_1,all_attention_masks_1,all_token_type_ids_1,all_feature_index_1,all_cls_index_1,all_p_mask_1,all_is_impossible_1,all_token_idx_1,all_segment_idx_1,all_clause_idx_1,all_conv_len_1,adj_b_1,all_utterance_input_1,all_q_query_input_1,all_q_query_mask_1,all_q_query_token_type_1,all_answer_cls_1 #all_start_positions,\all_end_positions,



    def train(self, train_dataset, output_dir, show_running_loss=True, eval_data_0=None, eval_data_1=None, verbose=True, **kwargs):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        device = self.device
        model = self.model
        args = self.args

        tb_writer = SummaryWriter(logdir=args.tensorboard_dir)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.bert_batch_preprocessing
        )

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [p for n, p in model.named_parameters() if n in params]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        # training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0)
        epoch_number = 0
        best_eval_metric_0 = None
        best_eval_metric_1 = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info("   Will skip the first %d steps in the current epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(**kwargs)
            training_progress_scores_1 = self._create_training_progress_scores_1(**kwargs)
        if args.wandb_project:
            wandb.init(project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs)
            wandb.watch(self.model)

        if args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for _ in train_iterator:
            model.train()
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(f"Epoch {epoch_number + 1} of {args.num_train_epochs}")
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
                position=0, leave=True
            )
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                # for t in batch:
                #     t.to(device)
                batch = tuple(t.to(device) for t in batch)

                inputs,inputs_1 = self._get_inputs_dict(batch)
                if args.fp16:
                    with amp.autocast():
                        outputs_0 = model(**inputs)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        loss_0 = outputs_0[0]
                        # loss_0_sg = loss_0.detach()

                        outputs_1 = model(**inputs_1)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        loss_1 = outputs_1[0]
                        # loss_1_sg = loss_1.detach()

                else:
                    outputs_0 = model(**inputs)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                    loss_0 = outputs_0[0]
                    # loss_0_sg = loss_0.detach()

                    outputs_1 = model(**inputs_1)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                    loss_1 = outputs_1[0]
                    # loss_1_sg = loss_1.detach()
                
                # loss =  loss_0 + 0.1 * loss_1 #weight loss
                # loss = loss_0/loss_0_sg + loss_1/loss_1_sg
                # r = 0.5
                # loss =  loss_0 * loss_1.pow(r)
                
                r = self.args.loss_r
                loss =  ((loss_0.pow(r) + loss_1.pow(r))/2).pow(1/r)
                
                # loss = loss_1
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar(
                            "loss", (tr_loss - logging_loss) / args.logging_steps, global_step,
                        )
                        logging_loss = tr_loss
                        if args.wandb_project or self.is_sweeping:
                            wandb.log(
                                {
                                    "Training loss": current_loss,
                                    "lr": scheduler.get_last_lr()[0],
                                    "global_step": global_step,
                                }
                            )

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        self.save_model(output_dir_current, optimizer, scheduler, model=model)

                    if args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        _, results_1, _ = self.eval_model(eval_data_0,eval_data_1, verbose=False, **kwargs)
                        results_0, _, _, _ = self.eval_model_0(eval_data_0,eval_data_1, verbose=False, **kwargs)

                        for key, value in results_0.items():
                            tb_writer.add_scalar("eval_0_{}".format(key), value, global_step)
                        for key, value in results_1.items():
                            tb_writer.add_scalar("eval_1_{}".format(key), value, global_step)

                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        if args.save_eval_checkpoints:
                            self.save_model(output_dir_current, optimizer, scheduler, model=model, results=results_0)

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results_0:
                            training_progress_scores[key].append(results_0[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(args.output_dir, "training_progress_scores.csv"), index=False,
                        )
                        if args.wandb_project or self.is_sweeping:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        training_progress_scores_1["global_step"].append(global_step)
                        training_progress_scores_1["train_loss"].append(current_loss)
                        for key in results_1:
                            training_progress_scores_1[key].append(results_1[key])
                        report_1 = pd.DataFrame(training_progress_scores_1)
                        report_1.to_csv(
                            os.path.join(args.output_dir, "training_progress_scores_1.csv"), index=False,
                        )
                        if args.wandb_project or self.is_sweeping:
                            wandb.log(self._get_last_metrics(training_progress_scores_1))

                        # if not best_eval_metric_0:
                        #     best_eval_metric_0 = results_0[args.early_stopping_metric]
                        #     self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results_0)
                        if not best_eval_metric_1:
                            best_eval_metric_1 = results_1[args.early_stopping_metric]
                            self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results_1)
                        if best_eval_metric_1 and args.early_stopping_metric_minimize:
                            if results_1[args.early_stopping_metric] - best_eval_metric_1 < args.early_stopping_delta:
                                best_eval_metric_1 = results_1[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir, optimizer, scheduler, model=model, results=results_1
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores_1,
                                        )
                        else:
                            if results_1[args.early_stopping_metric] - best_eval_metric_1 > args.early_stopping_delta:
                                best_eval_metric_1 = results_1[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir, optimizer, scheduler, model=model, results=results_1
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores_1,
                                        )
            # results, _ = self.eval_model(eval_data, verbose=False, **kwargs)
            # results_0, _, results_1, _ = self.eval_model(eval_data_0,eval_data_1, verbose=False, **kwargs)
            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            if args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:
                _, results_1, _ = self.eval_model(eval_data_0,eval_data_1, verbose=False, **kwargs)
                results_0, _, _, _ = self.eval_model_0(eval_data_0,eval_data_1, verbose=False, **kwargs)

                self.save_model(output_dir_current, optimizer, scheduler, results=results_0)
                self.save_model(output_dir_current, optimizer, scheduler, results=results_1)
                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results_0:
                    training_progress_scores[key].append(results_0[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(args.output_dir, "training_progress_scores.csv"), index=False)
                if args.wandb_project or self.is_sweeping:
                    wandb.log(self._get_last_metrics(training_progress_scores))

                training_progress_scores_1["global_step"].append(global_step)
                training_progress_scores_1["train_loss"].append(current_loss)
                for key in results_1:
                    training_progress_scores_1[key].append(results_1[key])
                report_1 = pd.DataFrame(training_progress_scores_1)
                report_1.to_csv(os.path.join(args.output_dir, "training_progress_scores_1.csv"), index=False)
                if args.wandb_project or self.is_sweeping:
                    wandb.log(self._get_last_metrics(training_progress_scores_1))

                if not best_eval_metric_1:
                    best_eval_metric_1 = results_1[args.early_stopping_metric]
                    self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results_1)

                if best_eval_metric_1 and args.early_stopping_metric_minimize:
                    if results_1[args.early_stopping_metric] - best_eval_metric_1 < args.early_stopping_delta:
                        best_eval_metric_1 = results_1[args.early_stopping_metric]
                        self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results_1)
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores_1,
                                )
                else:
                    if results_1[args.early_stopping_metric] - best_eval_metric_1 > args.early_stopping_delta:
                        best_eval_metric_1 = results_1[args.early_stopping_metric]
                        self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results_1)
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores_1,
                                )
        return (
            global_step,
            tr_loss / global_step if not self.args.evaluate_during_training else training_progress_scores_1,
        )

    def eval_model_0(self, eval_data_0,eval_data_1, output_dir=None, verbose=False, verbose_logging=False, **kwargs):

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        result, model_outputs, wrong_preds, labels  = self.evaluate_0(
            eval_data_0,eval_data_1, output_dir, verbose_logging=verbose
        ) #all_predictions_0, all_nbest_json_0, scores_diff_json_0, 
        self.results_0.update(result)
        if verbose:
            logger.info(self.results)

        return result, model_outputs, wrong_preds, labels

    def eval_model(self, eval_data_0,eval_data_1, output_dir=None, verbose=False, verbose_logging=False, **kwargs):
        """
        Evaluates the model on eval_data. Saves results to output_dir.

        Args:
            eval_data: Path to JSON file containing evaluation data OR list of Python dicts in the correct format. The model will be evaluated on this data.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            verbose_logging: Log info related to feature conversion and writing predictions.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (correct, similar, incorrect)
            text: A dictionary containing the 3 dictionaries correct_text, similar_text (the predicted answer is a substring of the correct answer or vise versa), incorrect_text.
        """  # noqa: ignore flake8"

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        all_predictions_1, all_nbest_json_1, scores_diff_json_1, eval_loss, eval_loss_0, eval_loss_1 = self.evaluate(
            eval_data_0,eval_data_1, output_dir, verbose_logging=verbose
        ) #all_predictions_0, all_nbest_json_0, scores_diff_json_0, 

        if isinstance(eval_data_0, str):
            with open(eval_data_0, "r", encoding=self.args.encoding) as f:
                truth = json.load(f)
        else:
            truth_0 = eval_data_0

        if isinstance(eval_data_1, str):
            with open(eval_data_1, "r", encoding=self.args.encoding) as f:
                truth_1 = json.load(f)
        else:
            truth_1 = eval_data_1

        # result_0, texts_0 = self.calculate_results(truth_0, all_predictions_0, **kwargs)
        result_0 = {}
        result_0["eval_loss"] = eval_loss
        result_0["eval_loss_0"] = eval_loss_0
        result_0["eval_loss_1"] = eval_loss_1
        
        result_1, texts_1 = self.calculate_results(truth_1, all_predictions_1, **kwargs)
        result_1["eval_loss"] = eval_loss
        result_1["eval_loss_0"] = eval_loss_0
        result_1["eval_loss_1"] = eval_loss_1

        
        self.results_1.update(result_1)
        if verbose:
            logger.info(self.results_0)
            logger.info(self.results_1)

        return result_0, result_1, texts_1#result_0, texts_0, result_1, texts_1


    def evaluate_0(
        self, eval_data_0,eval_data_1, output_dir, multi_label=False, prefix="", verbose=True, silent=False, wandb_log=True, **kwargs
    ):
        """
        Evaluates the model on eval_df.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
        labels = []
        model = self.model
        args = self.args
        device = self.device
        eval_output_dir = output_dir
        os.makedirs(eval_output_dir, exist_ok=True)
        results = {}
        
        if isinstance(eval_data_0, str):
            with open(eval_data_0, "r", encoding=self.args.encoding) as f:
                eval_examples_0 = json.load(f)
        else:
            eval_examples_0 = eval_data_0

        eval_dataset_0, examples_0, features_0 = self.load_and_cache_examples(
            eval_examples_0, evaluate=True, output_examples=True,m = 0
        )
        
        if isinstance(eval_data_1, str):
            with open(eval_data_1, "r", encoding=self.args.encoding) as f:
                eval_examples_1 = json.load(f)
        else:
            eval_examples_1 = eval_data_1

        eval_dataset_1, examples_1, features_1 = self.load_and_cache_examples(
            eval_examples_1, evaluate=True, output_examples=True,m = 1
        )
        
        eval_dataset_all_t = eval_dataset_0 + eval_dataset_1
        eval_dataset = self.convert_valid_data_to_dataset(eval_dataset_all_t)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,collate_fn=self.bert_batch_preprocessing)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(eval_dataloader)
        preds = np.empty((len(eval_dataset), 2))
        if multi_label:
            out_label_ids = np.empty((len(eval_dataset), 2))
        else:
            out_label_ids = np.empty((len(eval_dataset)))
        model.eval()

        if self.args.fp16:
            from torch.cuda import amp

        for i, batch in enumerate(tqdm(eval_dataloader, disable=args.silent or silent, desc="Running Evaluation", position=0, leave=True)):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "all_token_idx": batch[2],
                    "all_token_idx":batch[7],
                    "all_segment_idx":batch[8],
                    "all_clause_idx":batch[9],
                    "all_conv_len":batch[10],
                    "adj_b": batch[11],
                    "all_utterance_input":batch[12],
                    "all_q_query_input": batch[13],
                    "all_q_query_mask": batch[14],
                    "all_q_query_token_type": batch[15],
                    "all_answer_cls": batch[16]
                }
                labels.extend(inputs['all_answer_cls'].cpu().numpy().tolist())
                example_indices = batch[3]
                example_indices_1 = batch[20]

                if args.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        tmp_eval_loss = outputs[0]
                        logits = outputs[1]
                else:
                    outputs = model(**inputs)
                    tmp_eval_loss = outputs[0]
                    logits = outputs[1]

                if multi_label:
                    logits = logits.sigmoid()
                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            start_index = self.args.eval_batch_size * i
            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else len(eval_dataset)
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = inputs["all_answer_cls"].detach().cpu().numpy()

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds
        if not multi_label:
            preds = np.argmax(preds, axis=1)

        result, wrong = self.compute_metrics(preds, out_label_ids, eval_examples_0, **kwargs)
        result["eval_loss"] = eval_loss
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results0_cls.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        if self.args.wandb_project and wandb_log and not multi_label and not self.args.regression:
            if not wandb.setup().settings.sweep_id:
                logger.info(" Initializing WandB run for evaluation.")
                wandb.init(project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs)
            if not args.labels_map:
                self.args.labels_map = {i: i for i in range(2)}

            labels_list = sorted(list(self.args.labels_map.keys()))
            inverse_labels_map = {value: key for key, value in self.args.labels_map.items()}

            truth = [inverse_labels_map[out] for out in out_label_ids]

            # Confusion Matrix
            wandb.sklearn.plot_confusion_matrix(
                truth, [inverse_labels_map[pred] for pred in preds], labels=labels_list,
            )

            if not self.args.sliding_window:
                # ROC`
                wandb.log({"roc": wandb.plots.ROC(truth, model_outputs, labels_list)})

                # Precision Recall
                wandb.log({"pr": wandb.plots.precision_recall(truth, model_outputs, labels_list)})

        return results, model_outputs, wrong, labels

    def compute_metrics(self, preds, labels, eval_examples=None, multi_label=False, **kwargs):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            preds: Model predictions
            labels: Ground truth labels
            eval_examples: List of examples on which evaluation was performed
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            wrong: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"

        assert len(preds) == len(labels)

        extra_metrics = {}
        # for metric, func in kwargs.items():
        #     extra_metrics[metric] = func(labels, preds)

        mismatched = labels != preds

        if eval_examples:
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        if multi_label:
            label_ranking_score = label_ranking_average_precision_score(labels, preds)
            return {**{"LRAP": label_ranking_score}, **extra_metrics}, wrong
        # elif self.args.regression:
        #     return {**extra_metrics}, wrong

        mcc = matthews_corrcoef(labels, preds)

        if self.model.num_labels == 2:
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            return (
                {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
                wrong,
            )
        else:
            return {**{"mcc": mcc}, **extra_metrics}, wrong

    def evaluate(self, eval_data_0,eval_data_1, output_dir, verbose_logging=False):
        """
        Evaluates the model on eval_data.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        args = self.args

        if isinstance(eval_data_0, str):
            with open(eval_data_0, "r", encoding=self.args.encoding) as f:
                eval_examples_0 = json.load(f)
        else:
            eval_examples_0 = eval_data_0

        eval_dataset_0, examples_0, features_0 = self.load_and_cache_examples(
            eval_examples_0, evaluate=True, output_examples=True,m = 0
        )
        
        if isinstance(eval_data_1, str):
            with open(eval_data_1, "r", encoding=self.args.encoding) as f:
                eval_examples_1 = json.load(f)
        else:
            eval_examples_1 = eval_data_1

        eval_dataset_1, examples_1, features_1 = self.load_and_cache_examples(
            eval_examples_1, evaluate=True, output_examples=True,m = 1
        )
        
        eval_dataset_all_t = eval_dataset_0 + eval_dataset_1
        eval_dataset = self.convert_valid_data_to_dataset(eval_dataset_all_t)

        del eval_dataset_all_t,eval_dataset_0,eval_dataset_1,eval_examples_0,eval_examples_1
        gc.collect()

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,collate_fn=self.bert_batch_preprocessing)

        eval_loss_0 = 0.0
        eval_loss_1 = 0.0
        nb_eval_steps = 0
        model.eval()

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.args.fp16:
            from torch.cuda import amp

        all_results_0 = []
        all_results_1 = []
        for batch in tqdm(eval_dataloader, disable=args.silent, desc="Running Evaluation", position=0, leave=True):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "all_token_idx": batch[2],
                    #batch all_input_ids,
                    # all_attention_masks,
                    # all_token_type_ids,
                    # all_start_positions,
                    # all_end_positions,
                    # all_cls_index,
                    # all_p_mask,
                    # all_is_impossible,
                    "all_token_idx":batch[7],
                    "all_segment_idx":batch[8],
                    "all_clause_idx":batch[9],
                    "all_conv_len":batch[10],
                    "adj_b": batch[11],
                    "all_utterance_input":batch[12],
                    "all_q_query_input": batch[13],
                    "all_q_query_mask": batch[14],
                    "all_q_query_token_type": batch[15],
                    "all_answer_cls": batch[16]
                }
                inputs_1 = {
                    "input_ids": batch[17],
                    "attention_mask": batch[18],
                    "all_token_idx": batch[19],
                    #batch all_input_ids,
                    # all_attention_masks,
                    # all_token_type_ids,
                    # all_start_positions,
                    # all_end_positions,
                    # all_cls_index,
                    # all_p_mask,
                    # all_is_impossible,
                    "all_token_idx":batch[24],
                    "all_segment_idx":batch[25],
                    "all_clause_idx":batch[26],
                    "all_conv_len":batch[27],
                    "adj_b": batch[28],
                    "all_utterance_input":batch[29],
                    "all_q_query_input": batch[30],
                    "all_q_query_mask": batch[31],
                    "all_q_query_token_type": batch[32]
                }
                # if self.args.model_type in [
                #     "xlm",
                #     "roberta",
                #     "distilbert",
                #     "camembert",
                #     "electra",
                #     "xlmroberta",
                #     "bart",
                # ]:
                #     del inputs["token_type_ids"]

                example_indices = batch[3]
                example_indices_1 = batch[20]

                if args.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                    inputs_1.update({"cls_index": batch[21], "p_mask": batch[22]})

                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        eval_loss_0 += outputs[0].mean().item()
                        # eval_loss_0_sg = eval_loss_0.detach()

                        outputs_1 = model(**inputs_1)
                        eval_loss_1 += outputs_1[0].mean().item()
                        # eval_loss_1_sg = eval_loss_1.detach()
                else:
                    outputs = model(**inputs)
                    eval_loss_0 += outputs[0].mean().item()
                    # eval_loss_0_sg = eval_loss_0.detach()

                    outputs_1 = model(**inputs_1)
                    eval_loss_1 += outputs_1[0].mean().item()
                    # eval_loss_1_sg = eval_loss_1.detach()

                eval_loss = eval_loss_0 + eval_loss_1
                # eval_loss = eval_loss_0/eval_loss_0_sg + eval_loss_1/eval_loss_1_sg
                # r = 0.2
                # eval_loss = math.pow((math.pow(eval_loss_0,r) + math.pow(eval_loss_1,r))/2,r)
                for i, example_index in enumerate(example_indices):
                    eval_feature = features_0[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    if args.model_type in ["xlnet", "xlm"]:
                        # XLNet uses a more complex post-processing procedure
                        result = RawResultExtended(
                            unique_id=unique_id,
                            start_top_log_probs=to_list(outputs[0]), #outputs[0][i]
                            start_top_index=to_list(outputs[1]),
                            end_top_log_probs=to_list(outputs[2]),
                            end_top_index=to_list(outputs[3]),
                            cls_logits=to_list(outputs[4]),
                        )
                    else:
                        result = RawResult(
                            unique_id=unique_id,
                            start_logits=to_list(outputs[0]),
                            end_logits=to_list(outputs[1]),
                        )
                    all_results_0.append(result)

                for i, example_index in enumerate(example_indices_1):
                    eval_feature_1 = features_1[example_index.item()]
                    unique_id_1 = int(eval_feature_1.unique_id)
                    if args.model_type in ["xlnet", "xlm"]:
                        # XLNet uses a more complex post-processing procedure
                        result_1 = RawResultExtended(
                            unique_id=unique_id_1,
                            start_top_log_probs=to_list(outputs_1[0][i]),
                            start_top_index=to_list(outputs_1[1][i]),
                            end_top_log_probs=to_list(outputs_1[2][i]),
                            end_top_index=to_list(outputs_1[3][i]),
                            cls_logits=to_list(outputs_1[4][i]),
                        )
                    else:
                        result_1 = RawResult(
                            unique_id=unique_id_1,
                            start_logits=to_list(outputs_1[0][i]),
                            end_logits=to_list(outputs_1[1][i]),
                        )
                    all_results_1.append(result_1)

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_loss_0 = eval_loss_0 / nb_eval_steps
        eval_loss_1 = eval_loss_1 / nb_eval_steps
        prefix = "test"
        os.makedirs(output_dir, exist_ok=True)

        output_prediction_file_0 = os.path.join(output_dir, "predictions_{}.json".format(prefix))
        output_nbest_file_0 = os.path.join(output_dir, "nbest_predictions_{}.json".format(prefix))
        output_null_log_odds_file_0 = os.path.join(output_dir, "null_odds_{}.json".format(prefix))

        output_prediction_file_1 = os.path.join(output_dir, "predictions_1_{}.json".format(prefix))
        output_nbest_file_1 = os.path.join(output_dir, "nbest_predictions_1_{}.json".format(prefix))
        output_null_log_odds_file_1 = os.path.join(output_dir, "null_odds_1_{}.json".format(prefix))

        if args.model_type in ["xlnet", "xlm"]:
            # XLNet uses a more complex post-processing procedure
            # (all_predictions_0, all_nbest_json_0, scores_diff_json_0,) = write_predictions_extended(
            #     examples_0,
            #     features_0,
            #     all_results_0,
            #     args.n_best_size,
            #     args.max_answer_length,
            #     output_prediction_file_0,
            #     output_nbest_file_0,
            #     output_null_log_odds_file_0,
            #     eval_data_0,
            #     model.config.start_n_top,
            #     model.config.end_n_top,
            #     True,
            #     tokenizer,
            #     verbose_logging,
            # )

            (all_predictions_1, all_nbest_json_1, scores_diff_json_1,) = write_predictions_extended(
                examples_1,
                features_1,
                all_results_1,
                args.n_best_size,
                args.max_answer_length,
                output_prediction_file_1,
                output_nbest_file_1,
                output_null_log_odds_file_1,
                eval_data_1,
                model.config.start_n_top,
                model.config.end_n_top,
                True,
                tokenizer,
                verbose_logging,
            )
        else:
            # all_predictions_0, all_nbest_json_0, scores_diff_json_0 = write_predictions(
            #     examples_0,
            #     features_0,
            #     all_results_0,
            #     args.n_best_size,
            #     args.max_answer_length,
            #     False,
            #     output_prediction_file_0,
            #     output_nbest_file_0,
            #     output_null_log_odds_file_0,
            #     verbose_logging,
            #     True,
            #     args.null_score_diff_threshold,
            # )
            all_predictions_1, all_nbest_json_1, scores_diff_json_1 = write_predictions(
                examples_1,
                features_1,
                all_results_1,
                args.n_best_size,
                args.max_answer_length,
                False,
                output_prediction_file_1,
                output_nbest_file_1,
                output_null_log_odds_file_1,
                verbose_logging,
                True,
                args.null_score_diff_threshold,
            )

        return  all_predictions_1, all_nbest_json_1, scores_diff_json_1, eval_loss, eval_loss_0, eval_loss_1 #all_predictions_0, all_nbest_json_0, scores_diff_json_0,

    def predict(self, to_predict, n_best_size=None):
        """
        Performs predictions on a list of python dicts containing contexts and qas.

        Args:
            to_predict: A python list of python dicts containing contexts and questions to be sent to the model for prediction.
                        E.g: predict([
                            {
                                'context': "Some context as a demo",
                                'qas': [
                                    {'id': '0', 'question': 'What is the context here?'},
                                    {'id': '1', 'question': 'What is this for?'}
                                ]
                            }
                        ])
            n_best_size (Optional): Number of predictions to return. args.n_best_size will be used if not specified.

        Returns:
            list: A python list  of dicts containing the predicted answer/answers, and id for each question in to_predict.
            list: A python list  of dicts containing the predicted probability/probabilities, and id for each question in to_predict.
        """  # noqa: ignore flake8"
        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        args = self.args

        if not n_best_size:
            n_best_size = args.n_best_size

        self._move_model_to_device()

        eval_examples = build_examples(to_predict)
        eval_dataset, examples, features = self.load_and_cache_examples(
            eval_examples, evaluate=True, output_examples=True, no_cache=True
        )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.args.fp16:
            from torch.cuda import amp

        all_results = []
        for batch in tqdm(eval_dataloader, disable=args.silent, desc="Running Prediction", position=0, leave=True):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                if self.args.model_type in [
                    "xlm",
                    "roberta",
                    "distilbert",
                    "camembert",
                    "electra",
                    "xlmroberta",
                    "bart",
                ]:
                    del inputs["token_type_ids"]

                example_indices = batch[3]

                if args.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[4], "p_mask": batch[5]})

                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

                for i, example_index in enumerate(example_indices):
                    eval_feature = features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    if args.model_type in ["xlnet", "xlm"]:
                        # XLNet uses a more complex post-processing procedure
                        result = RawResultExtended(
                            unique_id=unique_id,
                            start_top_log_probs=to_list(outputs[0]),
                            start_top_index=to_list(outputs[1]),
                            end_top_log_probs=to_list(outputs[2]),
                            end_top_index=to_list(outputs[3]),
                            cls_logits=to_list(outputs[4]),
                        )
                    else:
                        result = RawResult(
                            unique_id=unique_id,
                            start_logits=to_list(outputs[0]),
                            end_logits=to_list(outputs[1]),
                        )
                    all_results.append(result)

        if args.model_type in ["xlnet", "xlm"]:
            answers = get_best_predictions_extended(
                examples,
                features,
                all_results,
                n_best_size,
                args.max_answer_length,
                model.config.start_n_top,
                model.config.end_n_top,
                True,
                tokenizer,
                args.null_score_diff_threshold,
            )
        else:
            answers = get_best_predictions(
                examples, features, all_results, n_best_size, args.max_answer_length, False, False, True, False,
            )

        answer_list = [{"id": answer["id"], "answer": answer["answer"][:-1]} for answer in answers]
        probability_list = [{"id": answer["id"], "probability": answer["probability"][:-1]} for answer in answers]

        return answer_list, probability_list

    def calculate_results(self, truth, predictions, **kwargs):
        truth_dict = {}
        questions_dict = {}
        for item in truth:
            for answer in item["qas"]:
                if answer["answers"]:
                    truth_dict[answer["id"]] = answer["answers"][0]["text"]
                else:
                    truth_dict[answer["id"]] = ""
                questions_dict[answer["id"]] = answer["question"]

        correct = 0
        incorrect = 0
        similar = 0
        correct_text = {}
        incorrect_text = {}
        similar_text = {}
        predicted_answers = []
        true_answers = []

        for q_id, answer in truth_dict.items():
            predicted_answers.append(predictions[q_id])
            true_answers.append(answer)
            if predictions[q_id].strip() == answer.strip():
                correct += 1
                correct_text[q_id] = answer
            elif predictions[q_id].strip() in answer.strip() or answer.strip() in predictions[q_id].strip():
                similar += 1
                similar_text[q_id] = {
                    "truth": answer,
                    "predicted": predictions[q_id],
                    "question": questions_dict[q_id],
                }
            else:
                incorrect += 1
                incorrect_text[q_id] = {
                    "truth": answer,
                    "predicted": predictions[q_id],
                    "question": questions_dict[q_id],
                }

        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(true_answers, predicted_answers)

        result = {"correct": correct, "similar": similar, "incorrect": incorrect, **extra_metrics}

        texts = {
            "correct_text": correct_text,
            "similar_text": similar_text,
            "incorrect_text": incorrect_text,
        }

        return result, texts

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _get_inputs_dict(self, batch):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
            #batch all_input_ids,
            # all_attention_masks,
            # all_token_type_ids,
            # all_start_positions,
            # all_end_positions,
            # all_cls_index,
            # all_p_mask,
            # all_is_impossible,
            "all_token_idx":batch[8],
            "all_segment_idx":batch[9],
            "all_clause_idx":batch[10],
            "all_conv_len":batch[11],
            "adj_b": batch[12],
            "all_utterance_input":batch[13],
            "all_q_query_input": batch[14],
            "all_q_query_mask": batch[15],
            "all_q_query_token_type": batch[16],
            "all_answer_cls":batch[17]
        }

        if self.args.model_type in ["xlm", "roberta", "distilbert", "camembert", "electra", "xlmroberta", "bart"]:
            del inputs["token_type_ids"]

        if self.args.model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": batch[5], "p_mask": batch[6]})

        inputs_1 = {
            "input_ids": batch[18],
            "attention_mask": batch[19],
            "token_type_ids": batch[20],
            "start_positions": batch[21],
            "end_positions": batch[22],
            #batch all_input_ids,
            # all_attention_masks,
            # all_token_type_ids,
            # all_start_positions,
            # all_end_positions,
            # all_cls_index,
            # all_p_mask,
            # all_is_impossible,
            "all_token_idx":batch[26],
            "all_segment_idx":batch[27],
            "all_clause_idx":batch[28],
            "all_conv_len":batch[29],
            "adj_b": batch[30],
            "all_utterance_input":batch[31],
            "all_q_query_input": batch[32],
            "all_q_query_mask": batch[33],
            "all_q_query_token_type": batch[34]
        } #"answer_cls":batch[35]

        if self.args.model_type in ["xlm", "roberta", "distilbert", "camembert", "electra", "xlmroberta", "bart"]:
            del inputs_1["token_type_ids"]

        if self.args.model_type in ["xlnet", "xlm"]:
            inputs_1.update({"cls_index": batch[5], "p_mask": batch[6]})

        return inputs, inputs_1
    def _create_training_progress_scores(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
                "global_step": [],
                "tp": [],
                "tn": [],
                "fp": [],
                "fn": [],
                "mcc": [],
                "train_loss": [],
                "eval_loss": [],
                **extra_metrics,
            }
        return training_progress_scores

    def _create_training_progress_scores_1(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores_1 = {
            "global_step": [],
            "correct": [],
            "similar": [],
            "incorrect": [],
            "train_loss": [],
            "eval_loss": [],
            "eval_loss_0": [],
            "eval_loss_1": [],
            **extra_metrics,
        }
        return training_progress_scores_1

    def save_model(self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            self.save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "a+") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = QuestionAnsweringArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
