import queue
from queue import PriorityQueue

import torch

from meta.classes import Job


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def acc(output, target, threshold):
    with torch.no_grad():
        return torch.count_nonzero(torch.abs(target - output) < threshold).item() / target.numel()


def acc_5(output, target):
    with torch.no_grad():
        return acc(output, target, 0.5)


def acc_4(output, target):
    with torch.no_grad():
        return acc(output, target, 0.4)


def acc_3(output, target):
    with torch.no_grad():
        return acc(output, target, 0.3)


def acc_2(output, target):
    with torch.no_grad():
        return acc(output, target, 0.2)


def acc_1(output, target):
    with torch.no_grad():
        return acc(output, target, 0.1)


def acc_05(output, target):
    with torch.no_grad():
        return acc(output, target, 0.05)


def early_acc_5(output, target):
    with torch.no_grad():
        early = target == 0
        if not torch.any(early):
            return 1
        return torch.count_nonzero(torch.abs(target[early] - output[early]) < 0.5).item() / torch.count_nonzero(
            early).item()


def early_acc_4(output, target):
    with torch.no_grad():
        early = target == 0
        if not torch.any(early):
            return 1
        return torch.count_nonzero(torch.abs(target[early] - output[early]) < 0.4).item() / torch.count_nonzero(
            early).item()


def early_acc_3(output, target):
    with torch.no_grad():
        early = target == 0
        if not torch.any(early):
            return 1
        return torch.count_nonzero(torch.abs(target[early] - output[early]) < 0.3).item() / torch.count_nonzero(
            early).item()


def early_acc_2(output, target):
    with torch.no_grad():
        early = target == 0
        if not torch.any(early):
            return 1
        return torch.count_nonzero(torch.abs(target[early] - output[early]) < 0.2).item() / torch.count_nonzero(
            early).item()


def early_acc_1(output, target):
    with torch.no_grad():
        early = target == 0
        if not torch.any(early):
            return 1
        return torch.count_nonzero(torch.abs(target[early] - output[early]) < 0.1).item() / torch.count_nonzero(
            early).item()


def early_acc_05(output, target):
    with torch.no_grad():
        early = target == 0
        if not torch.any(early):
            return 1
        return torch.count_nonzero(torch.abs(target[early] - output[early]) < 0.05).item() / torch.count_nonzero(
            early).item()


def tardy_acc_5(output, target):
    with torch.no_grad():
        tardy = target == 1
        if not torch.any(tardy):
            return 1
        return torch.count_nonzero(torch.abs(target[tardy] - output[tardy]) < 0.5).item() / torch.count_nonzero(
            tardy).item()


def tardy_acc_4(output, target):
    with torch.no_grad():
        tardy = target == 1
        if not torch.any(tardy):
            return 1
        return torch.count_nonzero(torch.abs(target[tardy] - output[tardy]) < 0.4).item() / torch.count_nonzero(
            tardy).item()


def tardy_acc_3(output, target):
    with torch.no_grad():
        tardy = target == 1
        if not torch.any(tardy):
            return 1
        return torch.count_nonzero(torch.abs(target[tardy] - output[tardy]) < 0.3).item() / torch.count_nonzero(
            tardy).item()


def tardy_acc_2(output, target):
    with torch.no_grad():
        tardy = target == 1
        if not torch.any(tardy):
            return 1
        return torch.count_nonzero(torch.abs(target[tardy] - output[tardy]) < 0.2).item() / torch.count_nonzero(
            tardy).item()


def tardy_acc_1(output, target):
    with torch.no_grad():
        tardy = target == 1
        if not torch.any(tardy):
            return 1
        return torch.count_nonzero(torch.abs(target[tardy] - output[tardy]) < 0.1).item() / torch.count_nonzero(
            tardy).item()


def tardy_acc_05(output, target):
    with torch.no_grad():
        tardy = target == 1
        if not torch.any(tardy):
            return 1
        return torch.count_nonzero(torch.abs(target[tardy] - output[tardy]) < 0.05).item() / torch.count_nonzero(
            tardy).item()


def err_5(output, target):
    with torch.no_grad():
        return torch.count_nonzero(torch.round(output) - target).item() / target.numel()


def err_25(output, target):
    with torch.no_grad():
        sure = torch.abs(output - 0.5) > 0.25
        if not torch.any(sure):
            return 1
        return torch.count_nonzero(torch.round(output[sure]) - target[sure]).item() / torch.sum(sure).item()


def err_05(output, target):
    with torch.no_grad():
        sure = torch.abs(output - 0.5) > 0.45
        if not torch.any(sure):
            return 1
        return torch.count_nonzero(torch.round(output[sure]) - target[sure]).item() / torch.sum(sure).item()


def determined_4(output, target):
    with torch.no_grad():
        return torch.count_nonzero(torch.abs(output - 0.5) > 0.1).item() / torch.numel(target)


def determined_3(output, target):
    with torch.no_grad():
        return torch.count_nonzero(torch.abs(output - 0.5) > 0.2).item() / torch.numel(target)


def determined_2(output, target):
    with torch.no_grad():
        return torch.count_nonzero(torch.abs(output - 0.5) > 0.3).item() / torch.numel(target)


def determined_1(output, target):
    with torch.no_grad():
        return torch.count_nonzero(torch.abs(output - 0.5) > 0.4).item() / torch.numel(target)


def determined_05(output, target):
    with torch.no_grad():
        return torch.count_nonzero(torch.abs(output - 0.5) > 0.45).item() / torch.numel(target)


def early_pct(output, target):
    with torch.no_grad():
        return 1 - torch.count_nonzero(target).item() / target.numel()


def tanh_acc(input, output, target, multiplier):
    with torch.no_grad():
        d = input[:, :, 1]
        d_mul = input[:, :, 1] * (1 + multiplier)
        pred_c = torch.atanh(output) * d + d
        late = pred_c > d_mul
        target_c = torch.atanh(target) * d + d
        target_binary = target_c > d
        return torch.sum(late == target_binary).item() / target.numel()


def tanh_acc_n2(input, output, target):
    with torch.no_grad():
        return tanh_acc(input, output, target, -0.2)


def tanh_acc_n15(input, output, target):
    with torch.no_grad():
        return tanh_acc(input, output, target, -0.15)


def tanh_acc_n1(input, output, target):
    with torch.no_grad():
        return tanh_acc(input, output, target, -0.1)


def tanh_acc_n05(input, output, target):
    with torch.no_grad():
        return tanh_acc(input, output, target, -0.05)


def tanh_acc_0(input, output, target):
    with torch.no_grad():
        return tanh_acc(input, output, target, 0)


def tanh_acc_05(input, output, target):
    with torch.no_grad():
        return tanh_acc(input, output, target, 0.05)


def tanh_acc_1(input, output, target):
    with torch.no_grad():
        return tanh_acc(input, output, target, 0.1)


def tanh_acc_15(input, output, target):
    with torch.no_grad():
        return tanh_acc(input, output, target, 0.15)


def tanh_acc_2(input, output, target):
    with torch.no_grad():
        return tanh_acc(input, output, target, 0.2)


def tanh_early_acc(input, output, target):
    with torch.no_grad():
        d = input[:, :, 1]
        pred_c = -(torch.atanh(output) * d - d)
        pred_late = pred_c > d
        target_binary = -(torch.atanh(target) * d - d) > d
        correct = pred_late == target_binary
        correct[~target_binary] = False
        return torch.sum(correct).item() / target.numel()


def tanh_tardy_acc(input, output, target):
    with torch.no_grad():
        d = input[:, :, 1]
        pred_c = torch.atanh(output) * d + d
        pred_late = pred_c > d
        target_binary = (torch.atanh(target) * d + d) > d
        correct = pred_late == target_binary
        correct[target_binary] = False
        return torch.sum(correct).item() / target.numel()


def tanh_early_pct(input, output, target):
    with torch.no_grad():
        d = input[:, :, 1]
        target_binary = (torch.atanh(target) * d + d) > d
        return torch.sum(~target_binary).item() / target.numel()


def tanh_earliest_s_acc(input, output, target):
    with torch.no_grad():
        p = input[:, :, 0]
        d = input[:, :, 1]
        r = input[:, :, 2]
        pred_c = torch.atanh(output) * d + d
        pred_s = pred_c - p
        n = len(p)
        n_jobs = len(p[0])
        pred_binary = torch.zeros((n, n_jobs))
        for i in range(n):
            q = PriorityQueue()
            for idx in range(n_jobs):
                q.put((pred_s[i][idx].item(), idx))
            t = 0
            while not q.empty():
                cur = q.get()
                cur_idx = cur[1]
                t = max(t, r[i][cur_idx].item())
                t += p[i][cur_idx].item()
                if t > d[i][cur_idx].item():
                    pred_binary[i][cur_idx] = 1
        return acc(pred_binary, target, 0.5)


def edd_wrong_acc_5(input, output, target):
    pp = input[:, :, 0]
    dd = input[:, :, 1]
    rr = input[:, :, 2]
    edd_late = torch.zeros(input.shape[:-1])
    for ii in range(len(pp)):
        p = pp[ii]
        r = rr[ii]
        d = dd[ii]
        n = len(p)
        I = [Job(i, p[i], r[i], d[i]) for i in range(n)]
        t = 0
        idx = 0
        ready = queue.PriorityQueue()
        for i in range(n):
            if ready.empty() and idx < n and I[idx].r > t:
                t = I[idx].r
            while idx < n and I[idx].r <= t:
                ready.put((I[idx].d, I[idx].p, I[idx]))
                idx += 1
            cur = ready.get()[2]
            t += cur.p
            if t > cur.d:
                edd_late[ii][cur.id] = 1
    pred_binary = torch.logical_xor(edd_late, torch.round(output)).int()
    return acc(pred_binary, target, 0.5)
