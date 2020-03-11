from gym.envs.registration import register

register(
    id='blackjack-v2', 
    entry_point='blackjack_env2.env.blackjack_env2:BlackjackEnv',
)