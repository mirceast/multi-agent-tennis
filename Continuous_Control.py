

agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)

# Run DDPG
scores = ddpg(
    agent, env, folder="double_400_300_chunk_actor_1e-4_critic_1e-4_batch_128")

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')

plt.show()
