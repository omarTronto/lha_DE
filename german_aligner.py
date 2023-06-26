"""
Main class for running the german aligner.
"""

import sys
import argparse
import logging
import time
from tqdm import *
import coloredlogs
from annoy import AnnoyIndex
from gensim.models.keyedvectors import KeyedVectors
from preprocessing.file_tools import wc
from align.GermanAligner import GermanAligner
from align.f1_eval import *
from tools.embeddings import EMBED_MODES
from tools.text_similarity import WORD_METRICS, METRICS_IMPLEMENTED

coloredlogs.install()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

parser = argparse.ArgumentParser()

parser.add_argument('-src', help='The source file', required=True)

parser.add_argument('-tgt', help='The target file', required=True)

parser.add_argument('-src_sents', help='The source splitted sents file .eoa', required=True)

parser.add_argument('-tgt_sents', help='The target splitted sents file .eoa', required=True)

parser.add_argument('-gold_src', help='path for the gold aligned complex sentences')

parser.add_argument('-gold_tgt', help='path for the gold aligned simplified sentences')

parser.add_argument('-out_dir', help='path for the outputs directory')

parser.add_argument('-k_best', default=1, help='How many pairs to choose per source/target. '
                                               'More than 1 will extract multiple nearest neighbours.', type=int)

parser.add_argument('-batch_size', default=2000, help='Batch the computation of the alignments. '
                                                      'Set to smaller values for limited amounts of RAM.', type=int)

parser.add_argument('-w2v_model_path', default='news', help='The Word2Vec model path that will be used. Only relevant if youre using a Word2Vec-based embedding method.')

parser.add_argument('-vec_size', default=600, help='The embedding dimensionality of your index/embedding model.'
                                                   'A wrong dimensionality might result in weird results.',type=int)
parser.add_argument('-lower_th', default=0.5, type=float,
                    help='The lower similarity threshold for accepting a pair.')

parser.add_argument('-upper_th', default=1.1, type=float,
                    help='The upper similarity threshold for accepting a pair.')

parser.add_argument('-rolling_thresholds', nargs="+", type=float,
                   help='List of lower threshold values, where results will be obtained for the corresponding highest F1 scores')

args = parser.parse_args()


if __name__ == '__main__':
    logging.warning("Loading the Word2Vec model {} ... This might take a few minutes!".format(
        args.w2v_model_path))
    
    if args.w2v_model_path.endswith(".bin"):
        w2v = KeyedVectors.load_word2vec_format(args.w2v_model_path, binary=True)

    else:
        w2v = KeyedVectors.load_word2vec_format(args.w2v_model_path, binary=False)

    logging.warning("Loaded Word2Vec model {} with embedding dim {} and vocab size {}".format(
        args.w2v_model_path, w2v.vector_size, len(w2v)))
    
    if args.vec_size is None:
        args.vec_size = w2v.vector_size
    
    args_dict = vars(args).copy()
    
    args_dict['emb'] = 'avg'
    args_dict['level'] = 'local'
    args_dict['refine'] = "wmd"
    args_dict['refine_all'] = True
    if args.batch_size == -1:
        args_dict['batch_size'] = None

    src_index = None
    logging.warning("Lazy mode: no src index")
    tgt_index = None
    logging.warning("Lazy mode: no tgt index") 
    args_dict['src_index'] = src_index
    args_dict['tgt_index'] = tgt_index
    args_dict['w2v'] = w2v

    logging.warning("Starting alignment of {} source and {} target documents...".format("LAZY","LAZY"))
    
    logging.warning("Will use the {} metric for refinement.".format(args_dict['refine']))
    
    start_time = time.time()
    batch_time = time.time()

    aligner = None
    
    try:
        total_documents_src = wc(args_dict['src'])
        total_documents_tgt = wc(args_dict['tgt'])
        assert total_documents_src == total_documents_tgt
    except AssertionError as e:
        logging.critical("No. of aligned documents should be equal, got {} docs for src and {} for tgt".format(total_documents_src, total_documents_tgt))
        raise e

    global_pairs = []
    for i in range(total_documents_src):
        global_pairs.append((i,i)) 
    
    args_dict['global_pairs'] = global_pairs
    
    for key in ["w2v_model_path", 'vec_size', 'emb', 'level', 'gold_src', 'gold_tgt', 'rolling_thresholds']:
        del args_dict[key] 
    
    #logging.info("Arguments are: {}".format(args_dict))
    
    if args.rolling_thresholds:
        logging.info("Starting Local aligner with rolling thresholds = {}".format(args.rolling_thresholds))
        best_f1 = -1
        best_th = -1
        for th in tqdm(args.rolling_thresholds, desc='Rolling Thresholds Experiments'):
            args_dict['lower_th'] = th
            aligner = GermanAligner(**args_dict)
            out_complex_sents_cln, out_simpl_sents_cln = aligner.predict_write()
            if args.gold_src and args.gold_tgt:
                logging.warning('Running F1 Evaluation for threshold {}'.format(th))
                precision, recall, n_gold, n_out, correct, f1_score = evaluate(args.gold_src, args.gold_tgt, out_complex_sents_cln, out_simpl_sents_cln)
                logging.info(f'Evaluation at threshold {th}: N_Gold_Aligns: {n_gold}, N_System_Aligns: {n_out}, N_Correct_Aligns: {correct}, precision: {round(precision, 2)}, recall: {round(recall, 2)}, F1 Score: {round(f1_score, 2)}')
                if f1_score >= best_f1:
                    best_f1 = f1_score
                    best_th = th
            else:
                best_f1 = None
                best_th = None
        if best_f1:
            logging.info("Expirement is completed with best F1 = {} at a threshold = {}".format(round(best_f1, 2), best_th))
                        
    else:
        logging.info("Starting Local aligner with single threshold = {}".format(args.lower_th))
        aligner = GermanAligner(**args_dict)
        out_complex_sents_cln, out_simpl_sents_cln = aligner.predict_write()
    
        if args.gold_src and args.gold_tgt:
            logging.warning('Running F1 Evaluation.')
            precision, recall, n_gold, n_out, correct, f1_score = evaluate(args.gold_src, args.gold_tgt, out_complex_sents_cln, out_simpl_sents_cln)
            logging.info(f'Evaluation at threshold {args.lower_th}: N_Gold_Aligns: {n_gold}, N_System_Aligns: {n_out}, N_Correct_Aligns: {correct}, precision: {round(precision, 2)}, recall: {round(recall, 2)}, F1 Score: {round(f1_score, 2)}')

    end_time = time.time()
    time_diff = end_time - start_time
    logging.warning("Finished! Total time for alignment %d minutes, %f seconds" \
                    % (int(time_diff / 60), time_diff % 60))

