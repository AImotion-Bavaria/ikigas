from gas_environment import SimpleGasEnvironment

def make_env():
    env = SimpleGasEnvironment(in_flow=15, out_flow=10)
    from stable_baselines3.common.env_checker import check_env

    # It will check your custom environment and output additional warnings if needed
    check_env(env)
    return env


def do_nothing(env):
    print("-"*50)
    print("Doing nothing")
    env.reset()
    for i in range(10):
        obs, reward, done, info = env.step(SimpleGasEnvironment.NOTHING)
        print("Got reward ", reward, " and observed ", obs)
        env.render()

def do_up(env):
    print("-"*50)
    print("Doing up")
    env.reset()
    for i in range(10):
        obs, reward, done, info = env.step(SimpleGasEnvironment.UP)
        print("Got reward ", reward, " and observed ", obs)
        env.render()


def do_pid(env):
    obs = env.reset()
    print("-" * 50)
    print("PID")
    for i in range(30):
        error = 50.0 - obs[0]  # difference to line pack
        if error < 0:
            action = SimpleGasEnvironment.DOWN
        else:
            action = SimpleGasEnvironment.UP

        obs, reward, done, info = env.step(action)
        print("Got reward ", reward, " and observed ", obs)
        env.render()

from stable_baselines3 import A2C, DQN
def train_stable_baselines(env):
    env.reset()
    print("Training ....")

    model = A2C("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=30_000)
    model.save("A2C_simple_gas")
    print("-" * 50)
    print("A2C")
    print("-" * 50)
    obs = env.reset()
    for i in range(50):
        act, something = model.predict(obs, deterministic=True)
        act_labels = ["up", "nothing", "down"]
        print("Picking action ... ", act_labels[act])
        obs, reward, done, info = env.step(act)
        env.render()
        print("Got reward ", reward, " and observed ", obs)
        if done:
            print("Game over")
            break



if __name__ == '__main__':
    env = make_env()
    do_nothing(env)
    #do_up(env)
    #do_naive(env)
    do_pid(env)
    train_stable_baselines(env)
    #policy_gradients(env)