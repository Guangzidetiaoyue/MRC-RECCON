import numpy as np
import json, os, logging, pickle, argparse
from evaluate_squad import compute_f1
from simpletransformers.question_answering import QuestionAnsweringModel
from sklearn.metrics import classification_report
def lcs(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])

    return lcs_set


def evaluate_results(text):
    partial_match_scores = []
    lcs_all = []
    impos1, impos2, impos3, impos4 = 0, 0, 0, 0
    pos1, pos2, pos3 = 0, 0, 0
    fscores, squad_fscores = [], []
    
    for i, key in enumerate(['correct_text', 'similar_text', 'incorrect_text']):
        for item in text[key]:
            if i==0:
                if 'impossible' in item and text[key][item]['predicted'] == '':
                    impos1 += 1
                elif 'span' in item:
                    pos1 += 1
                    fscores.append(1)
                    squad_fscores.append(1)
                    
            elif i==1:
                if 'impossible' in item:
                    impos2 += 1
                elif 'span' in item:
                    
                    z = text[key][item]
                    if z['predicted'] != '':
                        longest_match = list(lcs(z['truth'], z['predicted']))[0]
                        lcs_all.append(longest_match)
                        partial_match_scores.append(round(len(longest_match.split())/len(z['truth'].split()), 4))
                        pos2 += 1
                        r = len(longest_match.split())/len(z['truth'].split())
                        p = len(longest_match.split())/len(z['predicted'].split())
                        f = 2*p*r/(p+r)
                        fscores.append(f)
                        squad_fscores.append(compute_f1(z['truth'], z['predicted']))
                    else:
                        pos3 += 1
                        impos4 += 1
                        fscores.append(0)
                        squad_fscores.append(0)
                                
                    
            if i==2:
                if 'impossible' in item:
                    impos3 += 1
                elif 'span' in item:
                    if z['predicted'] == '':
                        impos4 += 1
                    pos3 += 1
                    fscores.append(0)
                    squad_fscores.append(0)
                    
    total_pos = pos1 + pos2 + pos3
    imr = impos2/(impos2+impos3)
    imp = impos2/(impos2+impos4)
    imf = 2*imp*imr/(imp+imr)
    
    p1 = 'Postive Samples:'
    p2 = 'Exact Match: {}/{} = {}%'.format(pos1, total_pos, round(100*pos1/total_pos, 2))
    p3 = 'Partial Match: {}/{} = {}%'.format(pos2, total_pos, round(100*pos2/total_pos, 2))
    p4a = 'LCS F1 Score = {}%'.format(round(100*np.mean(fscores), 2))
    p4b = 'SQuAD F1 Score = {}%'.format(round(100*np.mean(squad_fscores), 2))
    p5 = 'No Match: {}/{} = {}%'.format(pos3, total_pos, round(100*pos3/total_pos, 2))
    p6 = '\nNegative Samples'
    p7 = 'Inv F1 Score = {}%'.format(round(100*imf, 2))
    # p7a = 'Inv Recall: {}/{} = {}%'.format(impos2, impos2+impos3, round(100*imr, 2))
    # p7b = 'Inv Precision: {}/{} = {}%'.format(impos2, impos2+impos4, round(100*imp, 2))
    
    p = '\n'.join([p1, p2, p3, p4a, p4b, p5, p6, p7])
    return p


if __name__ == '__main__':

    global args
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='Initial learning rate')
    parser.add_argument('--batch-size', type=int, default=4, metavar='BS', help='batch size') #16
    parser.add_argument('--epochs', type=int, default=12, metavar='E', help='number of epochs')
    parser.add_argument('--model', default='rob', help='which model rob | span') #rob
    parser.add_argument('--fold', type=int, default=1, metavar='F', help='which fold')
    parser.add_argument('--context', action='store_true', default=True, help='use context')  #False
    parser.add_argument('--cuda', type=int, default=0, metavar='C', help='cuda device')
    parser.add_argument('--manual_seed', type=int, default=42, metavar='S', help='manual_seed')
    parser.add_argument('--loss_r', type=int, default=0, metavar='L_r', help='loss_r')
    args = parser.parse_args()

    print(args)
    
    model_family = {'rob': 'roberta', 'span': 'bert'}
    model_id = {'rob': 'roberta-base', 'span': 'spanbert-squad'}
    model_exact_id = {'rob': 'roberta-base', 'span': 'mrm8488/spanbert-finetuned-squadv2'}
    
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    model = args.model
    fold = str(args.fold)
    context = args.context
    cuda = args.cuda
    dataset = 'dailydialog'
    manual_seed= args.manual_seed
    loss_r = args.loss_r
    
    if context == False:
        max_q_length, max_c_length, max_a_length = 400, 400, 160
    elif context == True:
        max_q_length, max_c_length, max_a_length = 512, 512, 200

    if context == False:
        save_dir    = 'outputs/' + model_id[model] + '-dailydialog-qa-without-context-fold' + fold + '/'
        result_file = 'outputs/' + model_id[model] + '-dailydialog-qa-without-context-fold' + fold + '/results.txt'
        dump_file   = 'outputs/' + model_id[model] + '-dailydialog-qa-without-context-fold' + fold + '/test_predictions.pkl'
        x_train = json.load(open('data/my_data_pro/subtask1/fold' + fold + '/dailydialog_qa_train_without_context_new.json'))
        x_valid = json.load(open('data/my_data_pro/subtask1/fold' + fold + '/dailydialog_qa_valid_without_context_new.json'))
        x_test  = json.load(open('data/my_data_pro/subtask1/fold' + fold + '/dailydialog_qa_test_without_context_new.json'))
    else:
        # save_dir    = 'outputs/' + model_id[model] + '-dailydialog-qa-with-context-fold' + fold + '/'
        # result_file = 'outputs/' + model_id[model] + '-dailydialog-qa-with-context-fold' + fold + '/results.txt'
        # dump_file   = 'outputs/' + model_id[model] + '-dailydialog-qa-with-context-fold' + fold + '/test_predictions.pkl'
        # x_train = json.load(open('data/my_data_pro/subtask1/fold' + fold + '/dailydialog_qa_train_with_context_new.json')) #dailydialog_qa_train_with_context_new
        # x_valid = json.load(open('data/my_data_pro/subtask1/fold' + fold + '/dailydialog_qa_valid_with_context_new.json'))
        # x_test  = json.load(open('data/my_data_pro/subtask1/fold' + fold + '/dailydialog_qa_test_with_context_new.json'))
        
        save_dir    = 'outputs01/' + model_id[model] + '-dailydialog-qa-with-context-fold' + fold + '/'
        result_file_0 = 'outputs01/' + model_id[model] + '-dailydialog-qa-with-context-fold' + fold + '/results_0.txt'
        result_file_1 = 'outputs01/' + model_id[model] + '-dailydialog-qa-with-context-fold' + fold + '/results_1.txt'
        dump_file_0   = 'outputs01/' + model_id[model] + '-dailydialog-qa-with-context-fold' + fold + '/test_predictions_0.pkl'
        dump_file_1   = 'outputs01/' + model_id[model] + '-dailydialog-qa-with-context-fold' + fold + '/test_predictions_1.pkl'
        
        x_train_0 = json.load(open('data/my_data_pro/subtask1/fold' + fold + '/dailydialog_qa_train_with_context_new_0_c.json')) #dailydialog_qa_train_with_context_new
        x_valid_0 = json.load(open('data/my_data_pro/subtask1/fold' + fold + '/dailydialog_qa_valid_with_context_new_0_c.json'))
        x_test_0  = json.load(open('data/my_data_pro/subtask1/fold' + fold + '/dailydialog_qa_test_with_context_new_0_c.json'))

        x_train_1 = json.load(open('data/my_data_pro/subtask1/fold' + fold + '/dailydialog_qa_train_with_context_new_1_c.json')) #dailydialog_qa_train_with_context_new
        x_valid_1 = json.load(open('data/my_data_pro/subtask1/fold' + fold + '/dailydialog_qa_valid_with_context_new_1_c.json'))
        x_test_1  = json.load(open('data/my_data_pro/subtask1/fold' + fold + '/dailydialog_qa_test_with_context_new_1_c.json'))

    if fold == '1':
        num_steps = int(27915/batch_size)
    else:
        num_steps = int(25697/batch_size)

    train_args = {
        'fp16': False,
        'overwrite_output_dir': True, 
        'doc_stride': 512, 
        'max_query_length': max_q_length, 
        'max_answer_length': max_a_length,
        "max_seq_length": max_c_length,
        'n_best_size': 20,
        'null_score_diff_threshold': 0.0,
        'learning_rate': lr,
        'sliding_window': False,
        'output_dir': save_dir,
        'best_model_dir': save_dir + 'best_model/',
        'evaluate_during_training': True,
        'evaluate_during_training_steps': num_steps,
        'save_eval_checkpoints': False,
        'save_model_every_epoch': False,
        'save_steps': 500000,
        'train_batch_size': batch_size,
        'num_train_epochs': epochs,
        'roberta_cache_path':'roberta-base',
        'manual_seed':manual_seed,
        'loss_r':loss_r
    }

    qa_model = QuestionAnsweringModel(model_family[model], model_exact_id[model], args=train_args, cuda_device=cuda)
    qa_model.train_model(x_train_0,x_train_1, eval_data_0=x_valid_0, eval_data_1=x_valid_1)

    qa_model = QuestionAnsweringModel(model_family[model], save_dir + 'best_model/', args=train_args, cuda_device=cuda)
    result_0,  result_1, texts_1 = qa_model.eval_model(x_test_0,x_test_1)

    # r_0 = evaluate_results(texts_0)
    # print (r_0)
    rf_0 = open('results01/dailydialog_qa_0.txt', 'a')
    rf_0.write(str(args) + '\n\n')
    # rf_0.write(r_0 + '\n' + '-'*40 + '\n')    
    rf_0.close()
    rf_0 = open(result_file_0, 'a')
    rf_0.write(str(args) + '\n\n')
    # rf_0.write(r_0 + '\n' + '-'*40 + '\n')    
    rf_0.close()
    # pickle.dump(texts_0, open(dump_file_0, 'wb'))

    r_1 = evaluate_results(texts_1)
    print (r_1)
    rf_1 = open('results01/dailydialog_qa_1.txt', 'a')
    rf_1.write(str(args) + '\n\n')
    rf_1.write(r_1 + '\n' + '-'*40 + '\n')    
    rf_1.close()
    rf_1 = open(result_file_1, 'a')
    rf_1.write(str(args) + '\n\n')
    rf_1.write(r_1 + '\n' + '-'*40 + '\n')    
    rf_1.close()
    pickle.dump(texts_1, open(dump_file_1, 'wb'))

    result, model_outputs, wrong_predictions,labels = qa_model.eval_model_0(x_test_0,x_test_1)
    preds = np.argmax(model_outputs, 1)
    r = str(classification_report(labels, preds, digits=4))
    print (r)

    rf = open('results01/evaluation_' + dataset + '_classification.txt', 'a')
    rf.write(str(args) + '\n\n')
    rf.write(r + '\n' + '-'*54 + '\n')    
    rf.close()