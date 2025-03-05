import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time
from tqdm import trange


def sample_rollout(env, 
                   horizon, 
                   policy,
                   noise_stddev=None,
                   dU=None,
                   record_fname=None):
    """Samples a rollout from the agent.

    Arguments: 
        horizon: (int) The length of the rollout to generate from the agent.
        policy: (policy) The policy that the agent will use for actions.
        record_fname: (str/None) The name of the file to which a recording of the rollout
            will be saved. If None, the rollout will not be recorded.

    Returns: (dict) A dictionary containing data from the rollout.
        The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
    """
    video_record = record_fname is not None
    recorder = None if not video_record else VideoRecorder(env, record_fname)

    times, rewards = [], []
    obss, next_obss, actions, dones, reward_sum, done = [], [], [], [], 0, False

    obs = env.reset()
    policy.reset()
    for t in trange(horizon):
        if video_record:
            recorder.capture_frame()
        start = time.time()
        action = policy.act(obs, t)
        times.append(time.time() - start)

        if noise_stddev is None:
            next_obs, reward, done, info = env.step(action)
        else:
            action = action + np.random.normal(loc=0, scale=noise_stddev, size=[dU])
            action = np.minimum(np.maximum(action, env.action_space.low), env.action_space.high)
            next_obs, reward, done, info = env.step(action)
        reward_sum += reward
        
        obss.append(obs)
        next_obss.append(next_obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        obs = next_obs

        if done:
            break

    if video_record:
        recorder.capture_frame()
        recorder.close()

    return {
        "obs": np.array(obss),
        'next_obs': np.array(next_obss),
        "act": np.array(actions),
        "reward_sum": reward_sum,
        "rewards": np.array(rewards),
        'done': np.array(done)
    }