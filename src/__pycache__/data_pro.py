
import json

data_test_dd_ori = 'data/original_annotation/dailydialog_test.json'
data_test_ie_ori = 'data/original_annotation/iemocap_test.json'
data_train_ori = 'data/original_annotation/dailydialog_train.json'
data_valid_ori = 'data/original_annotation/dailydialog_valid.json'

data_train_sub1 = 'data/subtask1/fold1/dailydialog_qa_train_with_context.json'
data_valid_sub1 = 'data/subtask1/fold1/dailydialog_qa_valid_with_context.json'
data_test_dd_sub1 = 'data/subtask1/fold1/dailydialog_qa_test_with_context.json'
data_test_ie_sub1 = 'data/subtask1/fold1/iemocap_qa_test_with_context.json'

data_train_res = 'data/my_data_pro/subtask1/fold1/dailydialog_qa_train_with_context_new.json'
data_valid_res = 'data/my_data_pro/subtask1/fold1/dailydialog_qa_valid_with_context_new.json'
data_test_dd_res = 'data/my_data_pro/subtask1/fold1/dailydialog_qa_test_with_context_new.json'
data_test_ie_res = 'data/my_data_pro/subtask1/fold1/iemocap_qa_test_with_context_new.json'

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

#         if "is_impossible" in sub1_qa[0]:
#             is_impossible = sub1_qa[0]["is_impossible"]
#         else:
#             is_impossible = False
#         if not is_impossible:
#             answer = sub1_qa[0]["answers"]
#             answer_text = answer[0]["text"]
#             start_position_character = answer[0]["answer_start"]
#             if start_position_character > 450:
#                 # print(start_position_character)
#                 # if len(context_new) > 4:
#                 for i in range(1,len(context_new)):
#                     context6_temp = ' '.join(context_new[-i:])
#                     context6_temp_7 = ' '.join(context_new[-(i+1):])
#                     answer_start_new = context6_temp.find(answer_text)
#                     if answer_start_new != -1:
#                         if len(context6_temp_7) > 512 and len(context6_temp) < 512:
#                             sub1_qa[0]["answers"][0]["answer_start_new"] = answer_start_new
#                             r['context_new'] = context6_temp
#                             r['context_utterance_new'] = context_new[-i:]
#                             break
#                         elif answer_start_new > 250 and answer_start_new < 512:
#                                 # print(answer_start_new)
#                             # answer_start_new = context6_temp.find(answer_text)
#                             sub1_qa[0]["answers"][0]["answer_start_new"] = answer_start_new
#                             r['context_new'] = context6_temp
#                             r['context_utterance_new'] = context_new[-i:]
#                             break
#                         else:
#                             sub1_qa[0]["answers"][0]["answer_start_new"] = answer_start_new
#                             r['context_new'] = context6_temp
#                             r['context_utterance_new'] = context_new[-i:]
#                             break
#                     else:
#                         continue
#                 # if sub1_qa[0]["answers"][0]["answer_start_new"] == answer_start_new:
#                 #     continue
#             else:
#                 r['context_utterance_new'] = context_new
#                 r['context_new'] = context
#                 sub1_qa[0]["answers"][0]["answer_start_new"] = start_position_character
#         else:
#             r['context_utterance_new'] = context_new
#             r['context_new'] = context
#             sub1_qa[0]["answers"][0]["answer_start_new"] = sub1_qa[0]["answers"][0]["answer_start"]

#         # answer_start = sub1_qa[0]['answers'][0]['answer_start']
#     with open(data_test_ie_res,'w',encoding='utf-8') as f2: #data_train_res,data_valid_res,data_test_dd_res,data_test_ie_res
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
            start_position_character = answer[0]["answer_start"]
            if start_position_character > 450:
                # print(start_position_character)
                # if len(context_new) > 4:
                for i in range(1,len(context_new)):
                    context6_temp = ' '.join(context_new[-i:])
                    context6_temp_7 = ' '.join(context_new[-(i+1):])
                    answer_start_new = context6_temp.find(answer_text)
                    if answer_start_new != -1:
                        if len(context6_temp_7) > 512 and len(context6_temp) < 512:
                            sub1_qa[0]["answers"][0]["answer_start_new"] = answer_start_new
                            r['context_new'] = context6_temp
                            r['context_utterance_new'] = context_new[-i:]
                            break
                        elif answer_start_new > 250 and answer_start_new < 512:
                                # print(answer_start_new)
                            # answer_start_new = context6_temp.find(answer_text)
                            sub1_qa[0]["answers"][0]["answer_start_new"] = answer_start_new
                            r['context_new'] = context6_temp
                            r['context_utterance_new'] = context_new[-i:]
                            break
                        else:
                            sub1_qa[0]["answers"][0]["answer_start_new"] = answer_start_new
                            r['context_new'] = context6_temp
                            r['context_utterance_new'] = context_new[-i:]
                            break
                    else:
                        continue
                # if sub1_qa[0]["answers"][0]["answer_start_new"] == answer_start_new:
                #     continue
            else:
                r['context_utterance_new'] = context_new
                r['context_new'] = context
                sub1_qa[0]["answers"][0]["answer_start_new"] = start_position_character
        else:
            r['context_utterance_new'] = context_new
            r['context_new'] = context
            sub1_qa[0]["answers"][0]["answer_start_new"] = sub1_qa[0]["answers"][0]["answer_start"]

    with open(data_test_dd_res,'w',encoding='utf-8') as f2: #data_train_res,data_valid_res,data_test_dd_res,data_test_ie_res
        f2.write(json.dumps(data_sub1, ensure_ascii=True, indent=4, separators=(',', ':')))