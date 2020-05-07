from gym.envs.registration import register

register(
    id='blackjack-side-v1', 
    entry_point='side_env.env.side_env:SideEnv',
)