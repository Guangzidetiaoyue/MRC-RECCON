
import json

data_test_dd_ori = 'data/original_annotation/dailydialog_test.json'
data_test_ie_ori = 'data/original_annotation/iemocap_test.json'
data_train_ori = 'data/original_annotation/dailydialog_train.json'
data_valid_ori = 'data/original_annotation/dailydialog_valid.json'

data_train_sub1 = 'data/subtask1/fold1/dailydialog_qa_train_with_context.json'
data_valid_sub1 = 'data/subtask1/fold1/dailydialog_qa_valid_with_context.json'
data_test_dd_sub1 = 'data/subtask1/fold1/dailydialog_qa_test_with_context.json'
data_test_ie_sub1 = 'data/subtask1/fold1/iemocap_qa_test_with_context.json'

data_train_res_0 = 'data/my_data_pro/subtask1/fold1/dailydialog_qa_train_with_context_new_0.json'
data_valid_res_0 = 'data/my_data_pro/subtask1/fold1/dailydialog_qa_valid_with_context_new_0.json'
data_test_dd_res_0 = 'data/my_data_pro/subtask1/fold1/dailydialog_qa_test_with_context_new_0.json'
data_test_ie_res_0 = 'data/my_data_pro/subtask1/fold1/iemocap_qa_test_with_context_new_0.json'

# data_train_res_1 = 'data/my_data_pro/subtask1/fold1/dailydialog_qa_train_with_context_new_1.json'
# data_valid_res_1 = 'data/my_data_pro/subtask1/fold1/dailydialog_qa_valid_with_context_new_1.json'
# data_test_dd_res_1 = 'data/my_data_pro/subtask1/fold1/dailydialog_qa_test_with_context_new_1.json'
# data_test_ie_res_1 = 'data/my_data_pro/subtask1/fold1/iemocap_qa_test_with_context_new_1.json'

# with open(data_test_ie_ori, 'r') as f: #data_train_ori,data_valid_ori,data_test_dd_ori,data_test_ie_ori
#     data = json.load(f)
# #     # for row in data:
# #     #     print(row)
# #     #     c_data = data[row]
# #     #     context = c_data['utterance']
# #     #     emotion = c_data['emotion']

# with open(data_test_ie_sub1, 'r') as f1: #data_train_sub1,data_valid_sub1,data_test_dd_sub1,data_test_ie_sub1
#     data_sub1 = json.load(f1)

#     # data_res = {}
#     for r in data_sub1:
#         context_new = []
#         context_sentence = []
#         context = r['context']
#         sub1_qa = r['qas']
#         sub1_id = sub1_qa[0]['id']
#         idx_c = sub1_id.split('_')
#         tr_id_list = []
#         for i in range(1,len(idx_c)):
#             if idx_c[i] == 'utt':
#                 idx = idx_c[i+1]
#                 break
#             else:
#                 tr_id_list.append(idx_c[i])
#         tr_id = '_'.join(tr_id_list)
#         data_pro = data[tr_id]
#         for i in range(len(data_pro[0])):
#             context_sentence.append(data_pro[0][i]['utterance'])
#         for i in range(0,int(idx)):
#             utt = data_pro[0][i]['utterance']
#             context_new.append(utt)

#         assert context == ' '.join(context_new)
#         r['context_utterance']=context_new
#         r['all_utterance']=context_sentence

#         if sub1_qa[0]['id'] == 'dailydialog_tr_9778_utt_14_true_cause_utt_12_span_1':
#             print(tr_id)

#         if "is_impossible" in sub1_qa[0]:
#             is_impossible = sub1_qa[0]["is_impossible"]
#         else:
#             is_impossible = False
#         if not is_impossible:
#             answer = sub1_qa[0]["answers"]
#             answer_text = answer[0]["text"]

#             q_n = 'What is the evidence utterance'
#             q_s = 'The evidence utterance is'
#             q_e = 'What is the causal span'
#             query_idx_s = sub1_qa[0]['question'].find(q_s)
#             query_idx_s_l = query_idx_s+len(q_s)
#             query_idx_e = sub1_qa[0]['question'].find(q_e)
#             query_idx_e_l = query_idx_e+len(q_e)
#             evidence_u = sub1_qa[0]['question'][query_idx_s_l+1:query_idx_e-1]

#             query_new_0 = sub1_qa[0]['question'][:query_idx_s]+ q_n +sub1_qa[0]['question'][query_idx_e_l:]
#             sub1_qa[0]['question'] = query_new_0 #question_new_0

#             # q_1_n = 'The evidence utterance is '+ evidence_u + ' What is the causal span from context that is relevant to the evidence utterance ?'
#             # sub1_qa[0]['question_new_1'] = q_1_n

#             start_evidence_u = context.find(evidence_u)
#             start_position_character = answer[0]["answer_start"]
#             if start_position_character > 450:
#                 # print(start_position_character)
#                 # if len(context_new) > 4:
#                 for i in range(1,len(context_new)):
#                     context6_temp = ' '.join(context_new[-i:])
#                     context6_temp_7 = ' '.join(context_new[-(i+1):])
#                     evidence_a_s  = context6_temp.find(evidence_u)
#                     answer_start_new = context6_temp.find(answer_text)
#                     if answer_start_new != -1 and evidence_a_s > 0 and answer_start_new > 0:
#                         if len(context6_temp_7) > 512 and len(context6_temp) < 512:
#                             # sub1_qa[0]["answers"][0]["answer_start_new"] = answer_start_new
#                             r['context_new'] = context6_temp
#                             r['context_utterance_new'] = context_new[-i:]
#                             sub1_qa[0]['answers'] = [{'text':evidence_u,'answer_start_new':evidence_a_s,'answer_start':answer_start_new}] #answers_new_0
#                             # sub1_qa[0]['answers_new_1'] = [{'text':answer_text,'answer_start':answer_start_new}]
#                             break
#                         elif answer_start_new > 250 and answer_start_new < 512:
#                                 # print(answer_start_new)
#                             # answer_start_new = context6_temp.find(answer_text)
#                             # sub1_qa[0]["answers"][0]["answer_start_new"] = answer_start_new
#                             r['context_new'] = context6_temp
#                             r['context_utterance_new'] = context_new[-i:]
#                             sub1_qa[0]['answers'] = [{'text':evidence_u,'answer_start_new':evidence_a_s,'answer_start':answer_start_new}] #answers_new_0
#                             # sub1_qa[0]['answers_new_1'] = [{'text':answer_text,'answer_start':answer_start_new}]
#                             break
#                         else:
#                             # sub1_qa[0]["answers"][0]["answer_start_new"] = answer_start_new
#                             r['context_new'] = context6_temp
#                             r['context_utterance_new'] = context_new[-i:]
#                             sub1_qa[0]['answers'] = [{'text':evidence_u,'answer_start_new':evidence_a_s,'answer_start':answer_start_new}] #answers_new_0
#                             # sub1_qa[0]['answers_new_1'] = [{'text':answer_text,'answer_start':answer_start_new}]
#                             break
#                     else:
#                         continue
#                 # if sub1_qa[0]["answers"][0]["answer_start_new"] == answer_start_new:
#                 #     continue
#             else:
#                 r['context_utterance_new'] = context_new
#                 r['context_new'] = context
#                 sub1_qa[0]['answers'] = [{'text':evidence_u,'answer_start':start_position_character,'answer_start_new':start_evidence_u}]
#                 # sub1_qa[0]['answers_new_1'] = [{'text':answer_text,'answer_start':start_position_character}]

#         else:
#             answer = sub1_qa[0]["answers"]
#             answer_text = answer[0]["text"]
#             start_position_character = answer[0]["answer_start"]
#             q_n = 'What is the evidence utterance'
#             q_s = 'The evidence utterance is'
#             q_e = 'What is the causal span'
#             query_idx_s = sub1_qa[0]['question'].find(q_s)
#             query_idx_s_l = query_idx_s+len(q_s)
#             query_idx_e = sub1_qa[0]['question'].find(q_e)
#             query_idx_e_l = query_idx_e+len(q_e)

#             evidence_u = sub1_qa[0]['question'][query_idx_s_l+1:query_idx_e-1]
#             query_new_0 = sub1_qa[0]['question'][:query_idx_s]+ q_n +sub1_qa[0]['question'][query_idx_e_l:]
#             start_query_new_0 = context.find(evidence_u)
#             sub1_qa[0]['question'] = query_new_0 #question_new_0
#             sub1_qa[0]['answers'] = [{'text':answer_text,'answer_start':start_position_character,'answer_start_new':start_position_character}] #answers_new_0

#             # q_1_n = 'The evidence utterance is '+ evidence_u + ' What is the causal span from context that is relevant to the evidence utterance ?'
#             # sub1_qa[0]['question_new_1'] = q_1_n
#             r['context_utterance_new'] = context_new
#             r['context_new'] = context

#             # sub1_qa[0]["answers"][0]["answer_start_new"] = sub1_qa[0]["answers"][0]["answer_start"]
#             # sub1_qa[0]['answers_new_1'] = [{'text':answer_text,'answer_start':start_position_character}]

#         # answer_start = sub1_qa[0]['answers'][0]['answer_start']
#     with open(data_test_ie_res_0,'w',encoding='utf-8') as f2: #data_train_res,data_valid_res,data_test_dd_res,data_test_ie_res
#         f2.write(json.dumps(data_sub1, ensure_ascii=True, indent=4, separators=(',', ':')))


with open(data_test_dd_ori, 'r') as f: #data_train_ori,data_valid_ori,data_test_dd_ori,data_test_ie_ori
    data = json.load(f)
    # for row in data:
    #     print(row)
    #     c_data = data[row]
    #     context = c_data['utterance']
    #     emotion = c_data['emotion']

with open(data_test_dd_sub1, 'r') as f1: #data_train_sub1,data_valid_sub1,data_test_dd_sub1,data_test_ie_sub1
    data_sub1 = json.load(f1)

    # data_res = {}
    for r in data_sub1:
        context_new = []
        context_sentence = []
        context = r['context']
        sub1_qa = r['qas']
        sub1_id = sub1_qa[0]['id']
        idx = sub1_id.split('_')[4]
        tr_id = sub1_id.split('_')[1]+'_'+sub1_id.split('_')[2]

        data_pro = data[tr_id]
        for i in range(len(data_pro[0])):
            context_sentence.append(data_pro[0][i]['utterance'])
        for i in range(0,int(idx)):
            utt = data_pro[0][i]['utterance']
            context_new.append(utt)

        assert context == ' '.join(context_new)
        r['context_utterance']=context_new
        r['all_utterance']=context_sentence

        if sub1_qa[0]['id'] == 'dailydialog_tr_9778_utt_14_true_cause_utt_12_span_1':
            print(tr_id)

        if "is_impossible" in sub1_qa[0]:
            is_impossible = sub1_qa[0]["is_impossible"]
        else:
            is_impossible = False
        if not is_impossible:
            answer = sub1_qa[0]["answers"]
            answer_text = answer[0]["text"]

            q_n = 'What is the evidence utterance'
            q_s = 'The evidence utterance is'
            q_e = 'What is the causal span'
            query_idx_s = sub1_qa[0]['question'].find(q_s)
            query_idx_s_l = query_idx_s+len(q_s)
            query_idx_e = sub1_qa[0]['question'].find(q_e)
            query_idx_e_l = query_idx_e+len(q_e)
            evidence_u = sub1_qa[0]['question'][query_idx_s_l+1:query_idx_e-1]

            query_new_0 = sub1_qa[0]['question'][:query_idx_s]+ q_n +sub1_qa[0]['question'][query_idx_e_l:]
            sub1_qa[0]['question'] = query_new_0 #question_new_0

            # q_1_n = 'The evidence utterance is '+ evidence_u + ' What is the causal span from context that is relevant to the evidence utterance ?'
            # sub1_qa[0]['question_new_1'] = q_1_n

            start_evidence_u = context.find(evidence_u)
            start_position_character = answer[0]["answer_start"]
            if start_position_character > 450:
                # print(start_position_character)
                # if len(context_new) > 4:
                for i in range(1,len(context_new)):
                    context6_temp = ' '.join(context_new[-i:])
                    context6_temp_7 = ' '.join(context_new[-(i+1):])
                    evidence_a_s  = context6_temp.find(evidence_u)
                    answer_start_new = context6_temp.find(answer_text)
                    if answer_start_new != -1 and evidence_a_s > 0 and answer_start_new > 0:
                        if len(context6_temp_7) > 512 and len(context6_temp) < 512:
                            # sub1_qa[0]["answers"][0]["answer_start_new"] = answer_start_new
                            r['context_new'] = context6_temp
                            r['context_utterance_new'] = context_new[-i:]
                            sub1_qa[0]['answers'] = [{'text':evidence_u,'answer_start_new':evidence_a_s,'answer_start':answer_start_new}] #answers_new_0
                            # sub1_qa[0]['answers_new_1'] = [{'text':answer_text,'answer_start':answer_start_new}]
                            break
                        elif answer_start_new > 250 and answer_start_new < 512:
                                # print(answer_start_new)
                            # answer_start_new = context6_temp.find(answer_text)
                            # sub1_qa[0]["answers"][0]["answer_start_new"] = answer_start_new
                            r['context_new'] = context6_temp
                            r['context_utterance_new'] = context_new[-i:]
                            sub1_qa[0]['answers'] = [{'text':evidence_u,'answer_start_new':evidence_a_s,'answer_start':answer_start_new}] #answers_new_0
                            # sub1_qa[0]['answers_new_1'] = [{'text':answer_text,'answer_start':answer_start_new}]
                            break
                        else:
                            # sub1_qa[0]["answers"][0]["answer_start_new"] = answer_start_new
                            r['context_new'] = context6_temp
                            r['context_utterance_new'] = context_new[-i:]
                            sub1_qa[0]['answers'] = [{'text':evidence_u,'answer_start_new':evidence_a_s,'answer_start':answer_start_new}] #answers_new_0
                            # sub1_qa[0]['answers_new_1'] = [{'text':answer_text,'answer_start':answer_start_new}]
                            break
                    else:
                        continue
                # if sub1_qa[0]["answers"][0]["answer_start_new"] == answer_start_new:
                #     continue
            else:
                r['context_utterance_new'] = context_new
                r['context_new'] = context
                sub1_qa[0]['answers'] = [{'text':evidence_u,'answer_start':start_position_character,'answer_start_new':start_evidence_u}]
                # sub1_qa[0]['answers_new_1'] = [{'text':answer_text,'answer_start':start_position_character}]

        else:
            answer = sub1_qa[0]["answers"]
            answer_text = answer[0]["text"]
            start_position_character = answer[0]["answer_start"]
            q_n = 'What is the evidence utterance'
            q_s = 'The evidence utterance is'
            q_e = 'What is the causal span'
            query_idx_s = sub1_qa[0]['question'].find(q_s)
            query_idx_s_l = query_idx_s+len(q_s)
            query_idx_e = sub1_qa[0]['question'].find(q_e)
            query_idx_e_l = query_idx_e+len(q_e)

            evidence_u = sub1_qa[0]['question'][query_idx_s_l+1:query_idx_e-1]
            query_new_0 = sub1_qa[0]['question'][:query_idx_s]+ q_n +sub1_qa[0]['question'][query_idx_e_l:]
            start_query_new_0 = context.find(evidence_u)
            sub1_qa[0]['question'] = query_new_0 #question_new_0
            sub1_qa[0]['answers'] = [{'text':answer_text,'answer_start':start_position_character,'answer_start_new':start_position_character}] #answers_new_0

            # q_1_n = 'The evidence utterance is '+ evidence_u + ' What is the causal span from context that is relevant to the evidence utterance ?'
            # sub1_qa[0]['question_new_1'] = q_1_n
            r['context_utterance_new'] = context_new
            r['context_new'] = context

            # sub1_qa[0]["answers"][0]["answer_start_new"] = sub1_qa[0]["answers"][0]["answer_start"]
            # sub1_qa[0]['answers_new_1'] = [{'text':answer_text,'answer_start':start_position_character}]

        # answer_start = sub1_qa[0]['answers'][0]['answer_start']

    with open(data_test_dd_res_0,'w',encoding='utf-8') as f2: #data_train_res,data_valid_res,data_test_dd_res,data_test_ie_res
        f2.write(json.dumps(data_sub1, ensure_ascii=True, indent=4, separators=(',', ':')))