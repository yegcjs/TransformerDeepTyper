import os
import pynvml
import threading
from subprocess import Popen
from time import sleep


def run_task(idx, param):
    print(idx, ":", param)
    log = open(param['savepath'].replace('ckpts', 'logs').replace('.pt', '.txt'), "w")
    cmd = [
        'python', 'transformer.py',
        '--devices', f'0,1',
        '--dataset', param['dataset'],
        '--max_seq_len', param['max_seq_len'],
        '--hidden_dim', param['hidden_dim'],
        '--depth', param['depth'],
        '--num_attention_heads', param['num_attention_heads'],
        '--batch_size', param['batch_size'],
        '--loss_accumulation_steps', param['loss_accumulation_steps'],
        '--num_epochs', param['num_epochs'],
        '--savepath', param['savepath'],
        '--train'
    ]
    print(" ".join(cmd))
    print("="*10)
    proc = Popen(cmd, stdout=log)
    proc.wait()
    if proc.poll() is not None:
        print(idx, "Finish!")
    log.close()


if __name__ == '__main__':
    if not os.path.exists('ckpts'):
        os.mkdir('ckpts')
    if not os.path.exists('logs'):
        os.mkdir('logs')

    pynvml.nvmlInit()
    gpu_info = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(2)]

    idx = 0

    for dataset in ['js', 'py']:

        bsz_by_heads = {
            1: ['pad', 4, 4, 4] if dataset == 'py' else ['pad', 32, 32, 32],
            2: ['pad', 4, 4, 4] if dataset == 'py' else ['pad', 32, 32, 32],
            4: ['pad', 4, 4, 4] if dataset == 'py' else ['pad', 32, 32, 32]
        }
        for depth in [1, 2, 3]:
            for num_attention_heads in [1, 2, 4]:
                params = {
                    'dataset': dataset,
                    'max_seq_len': '1024',
                    'hidden_dim': str(256),
                    'depth': str(depth),
                    'num_attention_heads': str(num_attention_heads),
                    'batch_size': str(bsz_by_heads[num_attention_heads][depth]),
                    'loss_accumulation_steps': str(128 // bsz_by_heads[num_attention_heads][depth]),
                    'num_epochs': str(5) if dataset=='py' else str(100),
                    'savepath': f"ckpts/{dataset}-{depth}-{num_attention_heads}.pt",
                    'train': True
                }
                started = False
                while not started:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_info[idx % 2])
                    if mem.free > 8 * 1024 * 1024 * 1024:
                        run_task(idx, params)
                        thread = threading.Thread(target=run_task, args=(idx, params))
                        thread.start()
                        started = True
                        idx += 1
                    else:
                        sleep(180)
