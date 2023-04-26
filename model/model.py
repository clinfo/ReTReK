import tensorflow as tf
from kgcn import layers
from kgcn.default_model import DefaultModel
import tensorflow.contrib.keras as K


class GCN(DefaultModel):
    def build_placeholders(self, info, config, batch_size, inference_only=False, **kwargs):
        if inference_only:
            keys = ['adjs', 'features', 'is_train']
        else:
            keys = ['adjs', 'nodes', 'labels', 'mask', 'mask_label', 'mask_node', 'dropout_rate', 'is_train',
                'enabled_node_nums', 'features']
        return self.get_placeholders(info, config, batch_size, keys, **kwargs)

    def build_model(self, placeholders, info, config, batch_size, inference_only=False, **kwargs):
        adj_channel_num = info.adj_channel_num
        in_adjs = placeholders["adjs"]
        features = placeholders["features"]
        is_train = placeholders["is_train"]
        temperature = config.get("temperature", 1.)
        activation = config.get("activation", "tanh")
        if not inference_only:
            in_nodes = placeholders["nodes"]
            labels = placeholders["labels"]
            mask = placeholders["mask"]
            enabled_node_nums = placeholders["enabled_node_nums"]
        layer = features
        # layer: batch_size x graph_node_num x dim
        layer = layers.GraphConv(256, adj_channel_num)(layer, adj=in_adjs)
        layer = tf.nn.leaky_relu(layer)
        layer = K.layers.Dropout(0.3)(layer, training=is_train)

        layer = layers.GraphConv(256, adj_channel_num)(layer, adj=in_adjs)
        layer = tf.nn.leaky_relu(layer)
        layer = K.layers.Dropout(0.3)(layer, training=is_train)

        layer = layers.GraphConv(256, adj_channel_num)(layer, adj=in_adjs)
        layer = tf.nn.leaky_relu(layer)
        layer = K.layers.Dropout(0.3)(layer, training=is_train)

        layer = layers.GraphDense(256)(layer)
        layer = tf.nn.leaky_relu(layer)

        layer = layers.GraphGather()(layer)
        if activation == "tanh":
            layer = tf.nn.tanh(layer)
        elif activation == "leaky_relu":
            layer = tf.nn.leaky_relu(layer)

        layer = K.layers.Dense(info.label_dim)(layer)
        layer = layer / temperature
        prediction = tf.nn.softmax(layer, name="output")
        self.out = layer
        if not inference_only:
            # computing cost and metrics
            cost = mask * tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=layer)
            cost_opt = tf.reduce_mean(cost)
            metrics = {}
            cost_sum = tf.reduce_sum(cost)
            correct_count = mask * tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1)), tf.float32)
            metrics["correct_count"] = tf.reduce_sum(correct_count)
            return self, prediction, cost_opt, cost_sum, metrics
        else:
            return self, prediction


class GCNLarge(DefaultModel):
    def build_placeholders(self, info, config, batch_size, inference_only=False, **kwargs):
        if inference_only:
            keys = ['adjs', 'features', 'is_train']
        else:
            keys = ['adjs', 'nodes', 'labels', 'mask', 'mask_label', 'mask_node', 'dropout_rate', 'is_train',
                'enabled_node_nums', 'features']
        return self.get_placeholders(info, config, batch_size, keys, **kwargs)

    def build_model(self, placeholders, info, config, batch_size, inference_only=False, **kwargs):
        adj_channel_num = info.adj_channel_num
        in_adjs = placeholders["adjs"]
        features = placeholders["features"]
        is_train = placeholders["is_train"]
        temperature = config.get("temperature", 1.)
        activation = config.get("activation", "tanh")
        if not inference_only:
            in_nodes = placeholders["nodes"]
            labels = placeholders["labels"]
            mask = placeholders["mask"]
            enabled_node_nums = placeholders["enabled_node_nums"]
        layer = features
        # layer: batch_size x graph_node_num x dim
        layer = layers.GraphConv(512, adj_channel_num)(layer, adj=in_adjs)
        layer = tf.nn.leaky_relu(layer)
        layer = K.layers.Dropout(0.3)(layer, training=is_train)

        layer = layers.GraphConv(512, adj_channel_num)(layer, adj=in_adjs)
        layer = tf.nn.leaky_relu(layer)
        layer = K.layers.Dropout(0.3)(layer, training=is_train)

        layer = layers.GraphConv(512, adj_channel_num)(layer, adj=in_adjs)
        layer = tf.nn.leaky_relu(layer)
        layer = K.layers.Dropout(0.3)(layer, training=is_train)

        layer = layers.GraphDense(2048)(layer)
        layer = tf.nn.leaky_relu(layer)

        layer = layers.GraphGather()(layer)
        if activation == "tanh":
            layer = tf.nn.tanh(layer)
        elif activation == "leaky_relu":
            layer = tf.nn.leaky_relu(layer)

        layer = K.layers.Dense(2048)(layer)
        layer = K.layers.Dropout(0.3)(layer, training=is_train)
        layer = K.layers.Dense(info.label_dim)(layer)
        layer = layer / temperature
        prediction = tf.nn.softmax(layer, name="output")
        self.out = layer
        if not inference_only:
            # computing cost and metrics
            cost = mask * tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=layer)
            cost_opt = tf.reduce_mean(cost)
            metrics = {}
            cost_sum = tf.reduce_sum(cost)
            correct_count = mask * tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1)), tf.float32)
            metrics["correct_count"] = tf.reduce_sum(correct_count)
            return self, prediction, cost_opt, cost_sum, metrics
        else:
            return self, prediction
