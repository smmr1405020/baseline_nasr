import st_attention_network
import torch
import numpy as np
import data_generator
import metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def get_top_expansions(input_seq, input_seq_prob, no_of_expansions, end_point, candidate_seq_tuples):
    probs = get_prob(input_seq)
    probs = np.log(probs + 1e-15) + input_seq_prob

    candidate_inputs = []
    for i in range(probs.shape[1]):
        candidate_inputs.append(input_seq + [i])

    probs = list(probs[0])

    seq_prob_tuples = list(zip(candidate_inputs, probs))
    seq_prob_tuples = sorted(seq_prob_tuples, key=lambda item: item[1], reverse=True)

    for i in range(len(seq_prob_tuples)):
        seq_prob_tuples[i] = (seq_prob_tuples[i][0], seq_prob_tuples[i][1] - 0.5 * i)

    for i in range(len(seq_prob_tuples)):
        if seq_prob_tuples[i][0][-1] == end_point and len(seq_prob_tuples[i][0]) >= 3:
            candidate_seq_tuples.append(seq_prob_tuples[i])

    return seq_prob_tuples[:no_of_expansions], candidate_seq_tuples


def get_top_beams(seq_prob_tuples, no_beams, end_point, candidate_seq_tuples):
    next_beam_tuples = []
    for i in range(len(seq_prob_tuples)):
        seq = seq_prob_tuples[i][0]
        prob = seq_prob_tuples[i][1]
        if seq[-1] != end_point:
            next_beam_tuples.append(seq_prob_tuples[i])

    next_beam_tuples = sorted(next_beam_tuples, key=lambda item: item[1], reverse=True)

    return next_beam_tuples[:no_beams], candidate_seq_tuples


def get_trajectories(start_point, end_point, no_times, no_beams, no_expansions, K):
    candidate_seq_tuples = []

    trajectories = [([start_point], 0)]
    for i in range(no_times):
        all_traj = []
        for j in range(len(trajectories)):
            new_exp, new_cand_traj = get_top_expansions(trajectories[j][0], trajectories[j][1],
                                                        no_expansions, end_point, candidate_seq_tuples)
            all_traj = all_traj + new_exp
            candidate_seq_tuples = new_cand_traj
        trajectories, candidate_seq_tuples = get_top_beams(all_traj, no_beams, end_point, candidate_seq_tuples)

    candidate_seq_tuples = sorted(candidate_seq_tuples, key=lambda item: item[1], reverse=True)

    ans_traj = []

    def get_refined_traj(unrefined_traj):
        refined_traj = []
        for i in range(len(unrefined_traj)):
            gotBefore = False
            for j in range(len(refined_traj)):
                if refined_traj[j] == unrefined_traj[i]:
                    gotBefore = True
                    break

            if gotBefore == False:
                refined_traj.append(unrefined_traj[i])

        return refined_traj.copy()

    def isSame(traj_a, traj_b):
        if (len(traj_a) != len(traj_b)):
            return False
        for i in range(len(traj_a)):
            if (traj_a[i] != traj_b[i]):
                return False

        return True

    for i in range(len(candidate_seq_tuples)):
        rtraj = get_refined_traj(candidate_seq_tuples[i][0])
        if len(rtraj) >= 3:
            gotBefore = False
            for j in range(len(ans_traj)):
                if metric.calc_F1(ans_traj[j], rtraj) > 0.8:
                    gotBefore = True
                    break
            if not gotBefore:
                ans_traj.append(rtraj)
                if len(ans_traj) == K:
                    return ans_traj

    return ans_traj


def generate_result(load_from_file, K):
    st_attention_network.get_lstm_model(load_from_file)

    count = 1
    likability_score = []

    total_score_curr = 0
    total_traj_curr = 0

    total_score_curr_f1 = 0
    total_score_curr_pf1 = 0
    total_score_curr_edt = 0

    all_iou_scores = []

    total_traj_curr = 0
    count = 1

    for k, v in data_generator.test_data_dicts_vi[0].items():
        str_k = str(k).split("-")
        poi_start = int(str_k[0])
        poi_end = int(str_k[1])

        all_traj = get_trajectories(poi_start, poi_end, 13, max(2, K) * K, 10 * K, K)

        print("{}/{}".format(count, len(data_generator.test_data_dicts_vi[0])))
        count += 1
        print([poi_start, poi_end])
        print(v)
        print(all_traj)

        total_score_curr_f1 += metric.tot_f1_evaluation(v, data_generator.test_data_dicts_vi[2][k], all_traj)
        total_score_curr_pf1 += metric.tot_pf1_evaluation(v, data_generator.test_data_dicts_vi[2][k], all_traj)
        total_traj_curr += np.sum(data_generator.test_data_dicts_vi[2][k]) * len(all_traj)

        all_iou_scores.append(metric.coverage_iou(v,all_traj))

        avg_f1 = total_score_curr_f1 / total_traj_curr
        avg_pf1 = total_score_curr_pf1 / total_traj_curr
        avg_iou = np.average(np.array(all_iou_scores))

        print("Avg. upto now: F1: " + str(avg_f1) + " PF1: " + str(avg_pf1) + " IOU: " + str(avg_iou))

    print("\n")
    print("Final Score - With K = {}".format(K))
    avg_f1 = total_score_curr_f1 / total_traj_curr
    avg_pf1 = total_score_curr_pf1 / total_traj_curr
    avg_iou = np.average(np.array(all_iou_scores))

    print("F1: " + str(avg_f1) + " PF1: " + str(avg_pf1) + " IOU: " + str(avg_iou))

    #     total_score_curr += metric.f1_evaluation(v, data_generator.test_data_dicts_vi[2][k], all_traj)
    #     total_traj_curr += np.sum(data_generator.test_data_dicts_vi[2][k])
    #     print("Avg. upto now:" + str(total_score_curr / total_traj_curr))
    #
    # print("\n")
    # print("Final Score - With K = {}".format(K))
    # print(total_score_curr / total_traj_curr)

    #     likability_score_curr = metric.likability_score(v, all_traj)
    #     likability_score.append(likability_score_curr)
    #     print(np.average(likability_score))
    #     # print(likability_score_curr)
    #     # print("\n")
    #
    # print("Final Score: K - {}".format(K))
    # print(np.average(likability_score))
