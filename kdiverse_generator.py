import time

import st_attention_network
import torch
import numpy as np
import data_generator
import metric
import pprint
import os
import csv
import args_kdiverse

np.random.seed(12345)
torch.manual_seed(12345)

import random

random.seed(12345)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pp = pprint.PrettyPrinter(indent=4)


def get_prob(input_seq):
    input_seq_batch = [input_seq]
    input_seq_batch_len = [len(input_seq)]
    input_seq_batch_t = torch.LongTensor(input_seq_batch).to(device)
    input_seq_batch_len_t = torch.LongTensor(input_seq_batch_len).to(device)
    trained_model = st_attention_network.get_lstm_model(load_from_file=True)
    prob, _ = trained_model(input_seq_batch_t, input_seq_batch_len_t)
    prob = torch.reshape(prob, [len(input_seq), -1])
    prob = torch.softmax(prob, dim=-1)
    prob = prob.cpu().detach().numpy()
    return prob[-1].reshape(1, -1)


def get_top_expansions(input_seq, input_seq_prob, end_point, N_min=3, N_max=10):
    probs = get_prob(input_seq)
    probs = (np.log(probs + 1e-15) + input_seq_prob) * ((len(input_seq)) / (len(input_seq) + 1))

    candidate_inputs = []
    for i in range(probs.shape[1]):
        candidate_inputs.append(input_seq + [i])

    probs = list(probs[0])

    seq_prob_tuples = list(zip(candidate_inputs, probs))
    seq_prob_tuples = sorted(seq_prob_tuples, key=lambda item: item[1], reverse=True)

    for i in range(len(seq_prob_tuples)):
        seq_prob_tuples[i] = (seq_prob_tuples[i][0], seq_prob_tuples[i][1] - 0.5 * i)

    candidate_seq_tuples = []
    for i in range(len(seq_prob_tuples)):
        if seq_prob_tuples[i][0][-1] == end_point and N_min <= len(seq_prob_tuples[i][0]) <= N_max:
            # print(seq_prob_tuples[i][0])
            candidate_seq_tuples.append(seq_prob_tuples[i])

    # print(len(seq_prob_tuples))
    # print(len(candidate_seq_tuples))
    # print(len(candidate_inputs[0]))
    # print("\n")
    return seq_prob_tuples, candidate_seq_tuples


def get_top_beams(seq_prob_tuples, no_beams, end_point):
    next_beam_tuples = []
    for i in range(len(seq_prob_tuples)):
        seq = seq_prob_tuples[i][0]
        if seq[-1] != end_point and seq[-1] not in seq[:-1]:
            next_beam_tuples.append(seq_prob_tuples[i])

    next_beam_tuples = sorted(next_beam_tuples, key=lambda item: item[1], reverse=True)

    no_beams = min(no_beams, len(next_beam_tuples))
    return next_beam_tuples[:no_beams]


def get_trajectories(start_point, end_point, no_times, no_beams, NO_OF_DIVERSE_TRAJECTORIES,
                     eligibility_div=0.2, N_min=3, N_max=10):
    candidate_seq_tuples = []

    trajectories = [([start_point], 0)]
    for i in range(no_times):
        all_traj = []
        if len(trajectories[0][0]) >= N_max:
            break

        for j in range(len(trajectories)):
            new_exp, new_cand_traj = get_top_expansions(trajectories[j][0], trajectories[j][1],
                                                        end_point, N_min=N_min, N_max=N_max)
            all_traj = all_traj + new_exp
            candidate_seq_tuples += new_cand_traj
        trajectories = get_top_beams(all_traj, no_beams, end_point)
        if len(trajectories) == 0:
            break

    candidate_seq_tuples = sorted(candidate_seq_tuples, key=lambda item: item[1], reverse=True)

    ans_traj = []
    waiting_trajectories = []

    def get_refined_traj(unrefined_traj):
        refined_traj = []
        for i_ in range(len(unrefined_traj)):
            gotBefore_ = False
            for j in range(len(refined_traj)):
                if refined_traj[j] == unrefined_traj[i_]:
                    gotBefore_ = True
                    break

            if not gotBefore_:
                refined_traj.append(unrefined_traj[i_])

        return refined_traj.copy()

    for i in range(len(candidate_seq_tuples)):
        rtraj = get_refined_traj(candidate_seq_tuples[i][0])
        if N_min <= len(rtraj) <= N_max:
            gotBefore = False
            for j in range(len(ans_traj)):
                if metric.calc_F1(ans_traj[j], rtraj) > 1 - eligibility_div:
                    gotBefore = True
                    break
            if not gotBefore:
                ans_traj.append(rtraj)
                if len(ans_traj) == NO_OF_DIVERSE_TRAJECTORIES:
                    return ans_traj
            else:
                waiting_trajectories.append(rtraj)

    itr = 0
    while len(ans_traj) < NO_OF_DIVERSE_TRAJECTORIES:
        ans_traj.append(waiting_trajectories[itr])
        itr += 1

    return ans_traj


def generate_result(load_from_file, K, N_min, N_max):
    st_attention_network.get_lstm_model(load_from_file)

    total_score_curr_f1 = 0
    total_score_curr_pf1 = 0
    total_score_likability = 0
    total_score_intra_div_f1 = 0

    total_traj_curr = 0
    count = 1

    all_gtset = dict()
    all_gtfreqset = dict()
    all_recset = dict()

    st = time.time()
    for k, v in data_generator.query_dict_trajectory_test.items():
        str_k = str(k).split("-")
        poi_start = int(str_k[0])
        poi_end = int(str_k[1])

        no_times = len(data_generator.vocab_to_int) - 4

        all_traj = get_trajectories(poi_start, poi_end, no_times=no_times, no_beams=4 * K,
                                    NO_OF_DIVERSE_TRAJECTORIES=K, eligibility_div=0.3, N_min=N_min, N_max=N_max)

        print("{}/{}".format(count, len(data_generator.query_dict_trajectory_test)))
        count += 1
        print([data_generator.int_to_vocab[poi_start], data_generator.int_to_vocab[poi_end]])

        k_converted = str(data_generator.int_to_vocab[poi_start]) + '-' + str(data_generator.int_to_vocab[poi_end])

        dict_temp = dict()
        dict_temp[k] = v
        all_gtset[k_converted] = list(data_generator.convert_int_to_vocab(dict_temp).values())[0]

        all_gtfreqset[k_converted] = [data_generator.query_dict_freq_test[k]]

        dict_temp = dict()
        dict_temp[k] = all_traj
        all_recset[k_converted] = list(data_generator.convert_int_to_vocab(dict_temp).values())[0]
        #print(metric.tot_f1_evaluation(v, data_generator.query_dict_freq_test[k], all_traj))

        total_score_likability += metric.likability_score_3(v, data_generator.query_dict_freq_test[k], all_traj)
        total_score_curr_f1 += metric.tot_f1_evaluation(v, data_generator.query_dict_freq_test[k], all_traj)
        total_score_curr_pf1 += metric.tot_pf1_evaluation(v, data_generator.query_dict_freq_test[k], all_traj)
        total_score_intra_div_f1 += metric.intra_div_F1(all_traj)

        total_traj_curr += np.sum(data_generator.query_dict_freq_test[k]) * len(all_traj)

        avg_likability = total_score_likability / (count - 1)
        avg_div = total_score_intra_div_f1 / (count - 1)
        avg_f1 = total_score_curr_f1 / total_traj_curr
        avg_pf1 = total_score_curr_pf1 / total_traj_curr

        print("Avg. upto now: Likability: " + str(avg_likability) + " F1: " + str(avg_f1) + " PF1: " + str(avg_pf1)
              + " Div: " + str(avg_div))

    end = time.time()
    print("Time: {}".format((end - st)/count))

    print("\n")
    print("Final Score - With K = {}".format(K))
    avg_likability = total_score_likability / (count - 1)
    avg_div = total_score_intra_div_f1 / (count - 1)
    avg_f1 = total_score_curr_f1 / total_traj_curr
    avg_pf1 = total_score_curr_pf1 / total_traj_curr

    print("Likability: " + str(avg_likability) + " F1: " + str(avg_f1) + " PF1: " + str(avg_pf1)
          + " Div: " + str(avg_div))

    write_to_file(all_recset, 'recset_nasr', N_min=N_min, N_max=N_max)

    return


def write_to_file(dictionary, directory, N_min, N_max, isFreq=False):
    if not isFreq:
        file_path = os.path.join(directory, str(data_generator.embedding_name)) \
                    + "_index_" + str(args_kdiverse.test_index) \
                    + "_min_" + str(N_min) \
                    + "_max_" + str(N_max) \
                    + "_copy_" + str(args_kdiverse.copy_no) + '.csv'
    else:
        file_path = os.path.join(directory, str(data_generator.embedding_name)) + "_" + str(
            args_kdiverse.test_index) + '_freq.csv'

    write_lines = []

    for k, v in dictionary.items():
        for i in range(len(v)):
            write_lines.append(v[i])
        write_lines.append([-1])

    with open(file_path, mode='w+', newline="") as to_csv_file:
        csv_file_writer = csv.writer(to_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in write_lines:
            csv_file_writer.writerow(row)

    return

#     write_distmat_to_file()
#
#         total_score_curr += metric.f1_evaluation(v, data_generator.test_data_dicts_vi[2][k], all_traj)
#         total_traj_curr += np.sum(data_generator.test_data_dicts_vi[2][k])
#         print("Avg. upto now:" + str(total_score_curr / total_traj_curr))
#
#     print("\n")
#     print("Final Score - With K = {}".format(K))
#     print(total_score_curr / total_traj_curr)
#
#         likability_score_curr = metric.likability_score(v, all_traj)
#         likability_score.append(likability_score_curr)
#         print(np.average(likability_score))
#         # print(likability_score_curr)
#         # print("\n")
#
#     print("Final Score: K - {}".format(K))
#     print(np.average(likability_score))
#
#

#
