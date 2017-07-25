from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
import functools
import random
import os
import json
import pickle, dill

from .RL_Agent_Base import RLAgent_Base
from helper.general_helpers import safe_list_get

###### Utility functions #######

def weight_variable(shape, wd = None):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    var = tf.Variable(initial)

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('weight_decay', weight_decay)

    return var

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.tanh, wd=None):
    """Reusable code for making a simple neural net layer.
    sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim], wd=wd)
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
    return activations

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.

    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator





class RLAgent_NN(RLAgent_Base):

    def __init__(self, actions, V, T, consume, period_length, limit_base,
                 model=None, samples=None, agent_name='NN_Agent', lim_stepsize=0.1,
                 state_variables=['volume', 'time'], normalized=False, agent_type='NN_Agent',
                 nn_params=None, init_sess=True):
        super().__init__(
            actions=actions,
            lim_stepsize=lim_stepsize,
            V=V,
            T=T,
            consume=consume,
            period_length=period_length,
            samples=samples,
            agent_name=agent_name,
            state_variables=state_variables,
            normalized=normalized,
            limit_base=limit_base,
            agent_type=agent_type)

        if nn_params is None:
            nn_params = {
                'layers':         [len(state_variables)+1, 400, 1],
                'steps':          40,
                'dropout':        0.2,
                'timereg':        0.04,
                'weightdecay':    4e-5,
                'learningrate':   0.01,
                'actreg':         0.05,
                'log_dir':        '/home/axel/tmp/tensorboard',
                'mini_batches':   True,
                'minibatch_size': 32768,
            }
        self.nn_params = nn_params
        for k,v in nn_params.items():
            setattr(self, k, v)

        self.graph = tf.Graph()

        
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
        
            self.inputs   = tf.placeholder(tf.float32, [None, nn_params.get('layers')[0]])
            self.code = tf.placeholder(tf.float32, [None, nn_params.get('layers')[-2]])
            self.targets  = tf.placeholder(tf.float32, [None, nn_params.get('layers')[-1]])
            self.keep_prob = tf.placeholder(tf.float32)
            self.scaler
            self.prediction
            self.encoder
            self.decoder
            self.optimize
            self.evaluate
            self.summary
            
            self.saver = tf.train.Saver()
            
            self.init_log_writer(nn_params.get('log_dir'))
            if init_sess:
                self.sess.run(tf.global_variables_initializer())
            
        self.step = 0



    def init_log_writer(self, log_dir):
        import datetime
        import os
        now = datetime.datetime.now()
        log = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
        self.log_writer = tf.summary.FileWriter(log)
        self.log_writer.add_graph(self.graph)

    def debug(self, varname='scaler/input_shift/input_shift'):
        with self.graph.as_default():
            #display(tf.global_variables())

            variables_names = [v.name for v in tf.trainable_variables() if v.name==varname]
            # print("-")
            values = self.sess.run(variables_names)
            for k,v in zip(variables_names, values):
                print(k, v)
            

    def get_params(self):
        params = {
               'actions': self.actions,
               'lim_stepsize': self.lim_stepsize,
               'V': self.V,
               'T': self.T,
               'consume': self.consume,
               'period_length': self.period_length,
               'agent_name': self.agent_name,
               'state_variables': self.state_variables,
               'normalized': self.normalized,
               'limit_base': self.limit_base,
               'agent_type': 'NN_Agent',
               'nn_params': self.nn_params
            }

        extra = {
            'mu_in':     list(list(self.scaler_params.values())[0].values),
            'sigma_in':  list(list(self.scaler_params.values())[1].values),
            'mu_out':    list(list(self.scaler_params.values())[2].values),
            'sigma_out': list(list(self.scaler_params.values())[3].values),
            'step':      self.step,
            }
        
        return params, extra

    def copy(self, new_name=None):
        params, extra = self.get_params()

        new_agent = RLAgent_NN(**params)
        if isinstance(new_name, str):
            new_agent.agent_name = new_name
        new_agent.samples = self.samples.copy()
        if hasattr(new_agent, "model"):
            new_agent.model = self.model.copy()

        for k, v in extra.items():
            setattr(new_agent, k, v)

        return new_agent

    def save(self, path=".", outfile_agent=None, outfile_model=None, outfile_samples=None, overwrite=False, ignore_samples=False):
        if outfile_agent is None:
            outfile_agent = self.agent_name
        if outfile_model is None:
            outfile_model = self.agent_name
        if outfile_samples is None:
            outfile_samples = self.agent_name

        # append file type
        if outfile_agent.split(".")[-1] != 'json':
            outfile_agent = '{}.json'.format(outfile_agent)
        if outfile_model.split(".")[-1] != 'chk':
            outfile_model = '{}.chk'.format(outfile_model)
        if outfile_samples.split(".")[-1] != 'csv':
            outfile_samples = '{}.csv'.format(outfile_samples)

        params, extra = self.get_params()
        params.update(extra)

        if not os.path.exists(path):
            os.makedirs(path)

        ### save agent to disk
        outfile_agent = os.path.join(path, outfile_agent)
        if os.path.isfile(outfile_agent) and overwrite is False:
            print("File '{}' exists! Do not overwrite!".format(outfile_agent))
        else:
            with open(outfile_agent, 'w') as f_out:
                f_out.write(json.dumps(params, default=lambda df: json.loads(df.to_json())) + "\n")
            print("Saved agent: '{}'".format(outfile_agent))

        ### save agent model to disk
        outfile_model = os.path.join(path, outfile_model)
        if os.path.isfile(outfile_model) and overwrite is False:
            print("File '{}' exists! Do not overwrite!".format(outfile_model))
        else:
            with self.graph.as_default():
                self.saver.save(self.sess, outfile_model)
            print("Saved model: '{}'".format(outfile_model))

        ### save samples to disk
        if ignore_samples:
            print("ignoring samples")
        else:
            outfile_samples = os.path.join(path, outfile_samples)
            if os.path.isfile(outfile_samples) and overwrite is False:
                print("File '{}'  exists! Do not overwrite!".format(outfile_samples))
            else:
                self.samples.to_csv(outfile_samples)
                print("Saved samples: '{}'".format(outfile_samples))


    def load(agent_name=None, path='.', infile_agent=None, infile_model=None, infile_samples=None, ignore_samples=False):

        if agent_name is None:
            assert isinstance(infile_agent, str), "Bad parameter 'infile_agent', given: {}".format(infile_agent)
            assert isinstance(infile_model, str), "Bad parameter 'infile_model', given: {}".format(infile_model)
            assert isinstance(infile_samples, str), "Bad parameter 'infile_samples', given: {}".format(infile_samples)
        else:
            infile_agent = "{}.json".format(agent_name)
            infile_model = "{}.chk".format(agent_name)
            infile_samples = "{}.csv".format(agent_name)

        with open(os.path.join(path, infile_agent), 'r') as f:
            data = json.load(f)
        
        ql = RLAgent_NN(
            actions=data['actions'],
            lim_stepsize=safe_list_get(data, idx='lim_stepsize', default=0.1),
            V=data['V'],
            T=data['T'],
            consume=safe_list_get(data, idx='consume', default='volume'),
            period_length=data['period_length'],
            agent_name=data['agent_name'],
            state_variables=data['state_variables'] or ['volume', 'time'],
            normalized=data['normalized'],
            limit_base=data['limit_base'],
            nn_params=data['nn_params'],
            init_sess=False
            )

        if ignore_samples:
            ql.samples = pd.DataFrame()
            # print("No samples loaded! Parameter 'ignore_samples'==True")
        else:
            ql.samples = pd.read_csv(os.path.join(path, infile_samples), parse_dates=['timestamp'], index_col=0)
        
        ### RESTORE model
        print("infile_model:", os.path.join(path, infile_model))

        with ql.graph.as_default():
            ql.saver.restore(sess=ql.sess, save_path=os.path.join(path, infile_model))
        
        ql.scaler_params = {
            ql.mu_in:     pd.Series(data.get('mu_in')),
            ql.sigma_in:  pd.Series(data.get('sigma_in')),
            ql.mu_out:    pd.Series(data.get('mu_out')),
            ql.sigma_out: pd.Series(data.get('sigma_out')),
        }
        ql.step = data.get('step')

        return ql

    def predict(self, states, verbose=False):
        df = pd.DataFrame(states)

        preds = pd.DataFrame([])
        for a in self.actions:
            d = df.copy()
            d['action'] = float(a)
            
            feed_dict = {
                self.inputs:    d,
                self.keep_prob: 1.0,
                **self.scaler_params,
            }

            preds[a] = self.sess.run(self.prediction, feed_dict).flatten()

            if verbose:
                display(d, preds)

        return preds


    def get_action(self, p_feat, which_min='first'):
        assert isinstance(which_min, str) and which_min in ['first', 'last']

        if self.step > 0:
            if isinstance(p_feat, list):
                p_feat = pd.DataFrame(p_feat).T
            df = pd.concat([pd.DataFrame(p_feat)]*len(self.actions), ignore_index=True)
            df['action'] = self.actions
            
            preds = np.array(self.predict(df))

            if which_min == 'first':
                # if multiple minima exist, choose first one (lowest aggression level)
                action_idx = np.nanargmin(preds)
            elif which_min == 'last':
                # if multiple minima exist, choose last one (highest aggression level)
                action_idx = np.where(preds == np.nanmin(preds))[0][-1]
        else: 
            action_idx = random.randint(0, len(self.actions)-1)
            
        return self.actions[action_idx], action_idx

    def fit(self, X, Y):
        # fit scaler parameters

        self.scaler_params = {
            self.mu_in:     np.mean(X, axis=0),
            self.sigma_in:  np.std(X, axis=0),
            self.mu_out:    np.mean(Y, axis=0),
            self.sigma_out: np.std(Y, axis=0),
        }

        # training loop
        for i in tqdm(range(self.steps), leave=False, desc="steps"):
            if self.mini_batches:
                for X_batch, Y_batch in iterate_minibatches(X, Y, min(self.minibatch_size, len(X)), shuffle=False):

                    feed_dict = {
                        self.inputs:    X_batch,
                        self.targets:   Y_batch,
                        self.keep_prob: (1.0 - self.dropout),
                        **self.scaler_params,
                    }
                    # display(X_batch.shape, X_batch.head())
                    # display(Y_batch.shape, Y_batch.head())
                    self.sess.run(self.optimize, feed_dict)
            else:
                feed_dict = {
                        self.inputs:    X,
                        self.targets:   Y,
                        self.keep_prob: (1.0 - self.dropout),
                        **self.scaler_params,
                    }
                self.sess.run(self.optimize, feed_dict)
            
        self.step += self.steps

        # summary
        summary = self.sess.run(self.summary, feed_dict)

        self.log_writer.add_summary(summary, self.step)
        self.log_writer.flush()


    def learn_fromSamples(self, new_samples=None, nb_it=5, heatmap_freq=0, verbose=False):
        if new_samples is None:
            df = self.samples.copy()
        else:
            df = pd.concat([new_samples, self.samples.sample(n=len(new_samples), axis=0)], ignore_index=True)
            print(self.samples.shape, df.shape)
            display(self.samples.head())
        df.reset_index(inplace=True)

        # d_rate = 0.95

        df.insert(loc=len(df.columns)-self.state_dim, column='done', value=abs(df['volume_n']) < 0.00001)
        df.insert(loc=len(df.columns)-self.state_dim, column='min_cost', value=df['cost'])
        df['min_cost'] = df['cost']

        for n in tqdm(range(nb_it), leave=False, desc='iterations'):
            if verbose:
                print("n", n)

            if self.step>0:
                # using previous classifier as estimate of q_n-1
                states = df[["{}_n".format(var) for var in self.state_variables]].copy()

                preds = self.predict(states)

                # preparing our new training data set
                # if done, do not add any future costs anymore.
                successor_costs = (1 - df['done']) * (preds.min(axis=1))
                df['min_cost'] = (df['cost'] + successor_costs)

            X = df[self.state_variables+['action']]
            Y = pd.DataFrame(df['min_cost'])
            
            self.fit(
                X=X, 
                Y=Y
                )

            if heatmap_freq>0 and n%heatmap_freq==0:
                self.heatmap_Q(vol_intervals=4, show_minima_count=True)

        # raise NotImplementedError("ToDo: Implement")

    def sample_from_Q(self, vol_intervals, which_min, extra_variables=None):
        df = pd.DataFrame([])
        for t in range(1, self.T+1):
            for v in np.linspace(0, self.V, num=vol_intervals+1)[1::][::-1]:

                if len(self.state_variables)>2:
                    assert extra_variables is not None, "Please provide parameter 'extra_variables', otherwise 2d plot can't be generated. All state_variables: {}".format(self.state_variables)
                
                state = self.generate_state(time_left=t,
                                        volume_left=v,
                                        extra_variables=extra_variables)
                state = pd.DataFrame(state).T

                q = np.array(self.predict(state, verbose=False))

                action, action_idx = self.get_action(state, which_min=which_min)
                
                minima_count = len(np.where(q == np.nanmin(q))[0])
                
                df_tmp = pd.DataFrame({'time': t,
                                       'state': str(state.values[0]),
                                       'volume': v,
                                       'q': np.nanmin(q),
                                       'minima_count': minima_count,
                                       'action': action}, index=["{:1.2f},{}".format(v, t)])
                df = pd.concat([df, df_tmp])

        return df



###### Tensorflow operators #######

    @define_scope
    def scaler(self):
        with tf.name_scope('input_shift'):
            self.mu_in     = tf.Variable(tf.zeros([ self.layers[0]]), name='input_shift', trainable=False)
            variable_summaries(self.mu_in)
        with tf.name_scope('input_scale'):
            self.sigma_in  = tf.Variable(tf.ones([ self.layers[0]]), name='input_scale', trainable=False)
            variable_summaries(self.sigma_in)
        with tf.name_scope('output_shift'):
            self.mu_out    = tf.Variable(tf.zeros([ self.layers[-1]]), name='output_shift', trainable=False)
            variable_summaries(self.mu_out)
        with tf.name_scope('output_scale'):
            self.sigma_out = tf.Variable(tf.ones([ self.layers[-1]]), name='output_scale', trainable=False)
            variable_summaries(self.sigma_out)

    @define_scope
    def prediction(self):
        return self.decoder * self.sigma_out + self.mu_out

    @define_scope
    def encoder(self):
        enc = (self.inputs - self.mu_in) / self.sigma_in
        
        for layer_num in range(1, len(self.layers) - 1):
            n_in = self.layers[layer_num - 1]
            n_out = self.layers[layer_num]
            enc = nn_layer(enc, n_in, n_out, 'layer%i' % layer_num, act=tf.nn.tanh, wd=self.weightdecay)
            enc = tf.nn.dropout(enc, self.keep_prob)
        return enc

    @define_scope
    def decoder(self):
        self.code = self.encoder
        dec = nn_layer(self.code, self.layers[-2], self.layers[-1], 'decoder', act=tf.identity)  #, wd=self.weightdecay)
        return dec

    @define_scope
    def optimize(self):

        # scale targets
        scaled_targets = (self.targets - self.mu_out) / self.sigma_out

        # data fit
        mse = tf.losses.mean_squared_error(self.decoder, scaled_targets)
        tf.summary.scalar('data_fit', mse)
        
        # loss function
        total_loss = mse  # + acorr_reg + activity_reg #  + weight_decay + sreg
        tf.summary.scalar('total_loss', total_loss)

        # optimization 
        optimizer = tf.train.AdamOptimizer(self.learningrate)
        train_op = tf.contrib.slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

        return train_op

    @define_scope
    def evaluate(self):

        # scale targets
        scaled_targets = (self.targets - self.mu_out) / self.sigma_out

        # test mse
        loss = tf.losses.mean_squared_error(self.decoder, scaled_targets)

        # put summary to new collection, to prevent it getting grabbed by merge_all
        return tf.summary.scalar('loss', loss, collections=['evaluation'])

    @define_scope
    def summary(self):
        return tf.summary.merge_all()

