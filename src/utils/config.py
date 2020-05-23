logk_config = {
    'band_norm': 100,
    'lr_decay': 0.92,
    'test_interval': 3,
    'cpu_num': 1,
    'rand_vars_3': [0.2],
    'batch_size': 64,
    'actor_lr': 0.002,
    'critic_lr': 0.002,
    'capacity': 1000,
    'profile_path': '/home/shen/research/RL/WANStream/logk_profile',
    'cpu_num_test': 1
}

ar_config = {
    'band_norm': 166,  # 161 145
    'lr_decay': 0.92,
    'test_interval': 3,
    'cpu_num': 1,
    'actor_lr': 0.002,
    'critic_lr': 0.002,
    'batch_size': 64,
    'rand_vars_3': [0.3]
}

pd_config = {
    'rand_vars_3': [0.2],
    'band_norm': 140,
    'cpu_num': 4,
    'actor_lr': 0.002,
    'critic_lr': 0.002,
    'batch_size': 64
}


ar_config2 = {
    **ar_config,
    'alpha': 0.2,
    'policy_type': 'not Gaussian',
    'target_update_interval': 1,
    'automatic_entropy_tuning': True,
    'lr': 0.001,
    'hidden_size': 128,
    'use_sac': 1
}
