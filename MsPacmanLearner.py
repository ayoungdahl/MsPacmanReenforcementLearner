
import tensorflow as tf
import gym
import numpy as np

# This code was inspired by Andrej Karpathy's Deep Reinforcement Learning: Pong from Pixels 
# http://karpathy.github.io/2016/05/31/rl/

pixels_x = 88
pixels_y = 80

def variable_summaries(var):
    #Copied from https://www.tensorflow.org/versions/master/get_started/summaries_and_tensorboard
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# The next few functions are helpers for use in constructing the conv net.
# The name parameters are provided to allow for passing output names to tensorboard
# in_layer should be the reference to the previous layer, or in other words the input layer
# shape ought to be a 4 element list representing the dimensions of the tensor at this layer
def make_layer(shape, in_layer):
    with tf.name_scope('weights'):      
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        variable_summaries(weights)
    with tf.name_scope('biases'):
        biases = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
        variable_summaries(biases)
    with tf.name_scope('input'):
        tf.summary.histogram('input', in_layer)
    return weights, biases    

def conv_layer(name, shape, stride, in_layer):
    with tf.name_scope(name): 
        weights, biases = make_layer(shape, in_layer)
        with tf.name_scope('activation'):
            conv = tf.nn.elu(tf.nn.conv2d(in_layer, weights, strides=stride, padding='SAME') + biases)
            tf.summary.histogram('activation', conv)
        return conv

def pool_layer(name, reduce_factor, in_layer):
    with tf.name_scope(name):
        with tf.name_scope('input'):
            tf.summary.histogram('input', in_layer)
        with tf.name_scope('pool_reduce'):
            pool = tf.nn.max_pool(in_layer, ksize=reduce_factor, strides=reduce_factor, padding='SAME')
            tf.summary.histogram('pool_reduce', pool)
        return pool

def fc_layer(name, shape, in_layer):
    with tf.name_scope(name):
        weights, biases = make_layer(shape, in_layer)
        with tf.name_scope('activation'):
            fc = tf.nn.elu(tf.matmul(in_layer, weights) + biases)
            tf.summary.histogram('activation', fc)
        return tf.nn.dropout(fc, 0.5)

def out_layer(name, shape, in_layer):
    with tf.name_scope(name):
        weights, biases = make_layer(shape, in_layer)
        with tf.name_scope('pre-activation'):
            z = tf.matmul(in_layer, weights) + biases
            tf.summary.histogram('pre-activation', z)
        with tf.name_scope('softmax'):
            s_max = tf.nn.softmax(z)
            tf.summary.histogram('softmax', z)
        return s_max

# This function is called at every game step.  It's input is the 3D pixel tensor provided by openAI Gym
# Frist axis is the rows, second is columns, and third is for RGB
def norm_and_reduce(raw_image):
    # First downsample by a factor of 2.  Desperately need to keep computations down and this is a good
    # place to start
    image_reduced = raw_image[:pixels_x*2:2, ::2, ::]
    # Norm the input value range from (0,255) to (-1,1)
    image_normed = (image_reduced - 128) / 128
    return image_normed.reshape((-1, pixels_x, pixels_y, 3))


# The next function looks backwards through our rewards, assigns penalties, and applies discounted
# rewards from the future.  The resulting reward away has its mean centered about zero and is then 
# normed by its std dev
# I pass in running_mean as part of my scheme to assign the values of "ghost chomp (loss of life)" penalties
# and hesitation penalites (discourage waffling back and forth) based on the current runnin_mean.  The
# thinking is that as the agent improves, what may have been acceptable play before will now be less
# acceptable.  I'm not sure if this is a reasonable approach, but I thought it worth a try...

def rewards_back_in_time(rewards, running_mean):
    # lol_penalty is applied at the estimated point where MsPacman is devoured
    lol_penalty = -running_mean / 3
    # every game step subtract a gradually increasing amount to encourage efficient play
    jitter_penalty = -np.ceil(running_mean / 300)

    # we know that at the last step MsPacman was eaten so apply the penalty
    rewards[-1] += lol_penalty
    # ... then since the emaulator only gives us a flag when all three lives are lost we must estimate
    # when the first two are lost.  I"ve noticed that there is a beginning of round hesitation in the 
    # game of at least 85 game steps (sometimes its more due to the game step sampling from the emulator)
    # If we see 85 consecutive 0 rewards assume that a life was lost.  It may not have been exactly 
    # 85 game steps ago but the penalty will be carried back (albiet discounted).  Also, its possible
    # that the agent has not lost a MsPacman but has been dithering about for 85 turns without earning 
    # rewards.  While no life has been lost it probably wouldn't be all so bad to apply a penalty to this
    # situation as well, so I won't sweat the potential inaccuracy. 
    score_ixs = np.nonzero(rewards)
    for i in range(1,len(score_ixs) - 1):
        if score_ixs[i] - score_ixs[i-1] >= 85:
            rewards[score_ixs[i]-85] += lol_penalty
    # Now discount are rewards/penalties backwards through record.
    reward_adj = 0
    rewards_back = np.zeros_like(rewards)
    for r in reversed(range(len(rewards))):
        reward_adj = reward_adj * 0.99 + rewards[r] + jitter_penalty
        rewards_back[r] = reward_adj
    # Center and normalize
    rewards_back -= np.mean(rewards_back)
    rewards_back /= np.std(rewards_back)
    return np.array(rewards_back)


# Here's the part where we define the graph
image_in = tf.placeholder(tf.float32, shape=[None, pixels_x,pixels_y,3])

conv1 = conv_layer('conv1', [3,3,3,16], [1,2,2,1], image_in)
pool1 = pool_layer('pool1', [1,2,2,1], conv1)
conv2 = conv_layer('conv2', [3,3,16,32], [1,2,2,1], pool1)
pool2 = pool_layer('pool2', [1,2,2,1], conv2)
pool2_flat = tf.reshape(pool2, [-1, 960])
fc1 = fc_layer('fc1', [960, 512], pool2_flat)
out = out_layer('softmax', [512, 9], fc1)


adam = tf.train.AdamOptimizer(1e-4)

# This next line is where the agent picks at random from the options provided by softmax.  Each 
# option is picked proportionally to its likelihood
choice = tf.multinomial(out, 1)[0]

# The next few lines are where we "fudge" the ground truth.  Make y_true for the softmax node corresponding
# To the choice picked above equal to 1.0.  All others to 0.0 and then we can compute our loss
delta = tf.SparseTensor([choice], [1.0], [9])
all_zeros = tf.zeros_like(out)
adj_truth = all_zeros + tf.sparse_tensor_to_dense(delta)
loss = tf.nn.softmax_cross_entropy_with_logits(labels=adj_truth, logits=out) 

# We can't let tensorflow apply the gradients for us as we need to adjust based off our expected rewards
# function.  We will let tensorflow compute the gradients for us, save the gradients for each layer in a 
# list and then 
# apply the adjusted gradients later.  We must pass the weights and biases for each layer in 
# with the gradients (the way the api works) so each list element is a tuple containing a placehodler
# for the gradients and then the weights or biases for that layer.  
nvs = adam.compute_gradients(loss)
get_nablas = [nabla for nabla, _ in nvs]
nablas_to_put = [tf.placeholder(tf.float32, shape=n.get_shape()) for n in get_nablas]
wsbs = [wb for _, wb in nvs]
grads_and_vars = zip(nablas_to_put, wsbs)
update = adam.apply_gradients(grads_and_vars)

saver = tf.train.Saver()

# Initialize MsPacman
env = gym.make('MsPacman-v4')
atari_state = env.reset()

with tf.Session() as sess:
    
    
    merged = tf.summary.merge_all()
    game_writer = tf.summary.FileWriter('tf_logs/game', sess.graph)
    train_writer = tf.summary.FileWriter('tf_logs/train', sess.graph)

    sess.run(tf.global_variables_initializer())

    #rewards will hold the returned reward from the emulator for a game step
    # epns will hold the gradients returned from tensorflow for that game step
    rewards, epns = [], []
    running_mean, epoch, game_step, tot_reward = 0, 0, 0, 0
    #If we want to restart a saved network, uncomment the following line and replace the "0" for epoch
    # above with the epoch of the save state
    #saver.restore(sess, 'msPacNet' + str(epoch))
    
    while True:
        game_step += 1
        env.render()
        # Feed the adjusted pixels from the emulator into the newtork and decide on an action
        summary, step_nabla, soft_out = sess.run([merged, get_nablas, choice],
                                                 feed_dict={image_in:norm_and_reduce(atari_state)})
        game_writer.add_summary(summary, game_step)
        # Save the list of gradients in another list.  List in a list...
        # So each element of epns is a list of gradients for a game step
        # where each element is that list is a gradient tensor for a partiuclar layer.
        epns.append(step_nabla)
        
        # Have the emulator perform the action, get a new game state
        atari_state, reward, game_over, info = env.step(soft_out)
        # Tally and save rewards
        tot_reward += reward
        rewards.append(reward)
        #^
        #|------- if game is not over keep playing and saveing above
        #|----------- if game is over discount rewards, weight and apply the gradients below
        #v
        if game_over:
            epoch += 1
            rewards_adjusted_and_normed = rewards_back_in_time(rewards, running_mean)
            
            weighted_nablas = {}
            weighted = []
            # for each layer i in the network
            for i in range(len(epns[0])):
                # Init a zero matrix whose shape matches the shape of weight or baises at this layer
                m = np.zeros_like(epns[0][i])
                # for each game step j in the episode
                for j in range(len(epns)):
                    # sum the product of the reward at a game step with the gradient at a game step
                    # for our current layer i 
                    m += np.multiply(rewards_adjusted_and_normed[j], epns[j][i])
                # build a new list (one element per layer) for the means of the weighted gradients
                weighted.append(np.divide(m, len(epns)))
            
            # Now, build our placeholder dict to feed below.   
            for ix, nabla in enumerate(nablas_to_put):
                weighted_nablas[nabla] = weighted[ix]
            
            # Also (embarassingly) I'm not tensorflow savvy enough yet to know how to avoid 
            # doing a feed forward so for now (until I get smarter) place another input into the feed  
            weighted_nablas[image_in] = norm_and_reduce(atari_state)
            # Perform the gradient updates
            summary, _ = sess.run([merged, apply], feed_dict=weighted_nablas)
            train_writer.add_summary(summary, epoch)

            # Our running mean will be 1% latest score, 99% previous mean except for the first
            # which will just be the score
            if running_mean == 0: running_mean = tot_reward
            running_mean = (running_mean * 0.99) + (tot_reward * 0.01) 
                
            print('GAME %d OVER! Score: %f running mean: %f' % (epoch, tot_reward, running_mean))
            
            # Every 100 episodes save a copy of our net
            if epoch % 100 == 0:
               saver.save(sess, 'msPacNet' + str(epoch))
            # Re-init lists and counters and reset the emulator for another game
            rewards, epns = [], []
            tot_reward = 0
            atari_state = env.reset()

